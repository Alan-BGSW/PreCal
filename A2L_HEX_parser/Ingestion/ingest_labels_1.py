"""
Function-scoped calibration-label ingestion pipeline (BOTTOM-UP chunking).

Flow:  pages -> function blocks (header-gated)
            -> label ANCHORS (every place a label is named)
            -> per-anchor BOUNDED context window (anchor-then-grow)
            -> code-aware embeddings
            -> local vector store with metadata filters.

Key change vs the original:
    The old extractor used "last label wins, accumulate forever": one record
    per label, holding ALL text until the function identity changed. In a
    300-page function that produces a single multi-megabyte record that (a)
    blows past the embedding token limit (the 400 you hit) and (b) embeds to a
    useless centroid vector. The new extractor produces ONE SMALL CHUNK PER
    MENTION of a label, each carrying only a bounded neighbourhood of context.
    A label named in 3 places -> 3 chunks, all tagged with the same label_path;
    reassembly happens at the metadata layer, not at embed time.

Dependencies:
    pip install requests chromadb tiktoken FlagEmbedding

Embedding via the Bosch LLM farm:
    export API_KEY="..."
    export EMBED_DEPLOYMENT="text-embedding-3-large"   # optional override
    export EMBED_API_VERSION="2024-05-01-preview"      # optional override
"""

from __future__ import annotations

import os
import re
import time
import sys
import bisect
import hashlib
import json
import sqlite3
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

# ----------------------------------------------------------------------------
# 1. FUNCTION HEADER DETECTION + NORMALIZATION   (unchanged from your original)
# ----------------------------------------------------------------------------

_HEADER_PATTERNS = [
    re.compile(r"FC\s*:\s*([A-Za-z][A-Za-z0-9_]*)\s*/\s*([\d.]+\s*;\s*\d+)"),
    re.compile(r"\[\s*([A-Za-z][A-Za-z0-9_]*)\s+([\d.]+\s*;\s*\d+)\s*\]"),
    re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+([\d.]+\s*;\s*\d+)\b"),
]


def normalize_function_name(raw_name: str) -> str:
    """Canonical, VERSION-FREE key so old/new docs collapse to the same scope."""
    return raw_name.strip().lower()


def normalize_version(raw_version: str) -> str:
    """'1400.0.0; 2' and '1400.0.0;2' -> '1400.0.0;2'."""
    return re.sub(r"\s+", "", raw_version.strip())


def detect_function_header(page_text: str) -> tuple[str, str] | None:
    """Return (normalized_name, normalized_version) if any header form matches."""
    for line in page_text.splitlines():          # line-outermost: top-of-page wins
        for pattern in _HEADER_PATTERNS:
            m = pattern.search(line)
            if m:
                return normalize_function_name(m.group(1)), normalize_version(m.group(2))
    return None


# ----------------------------------------------------------------------------
# 2. SEGMENT PAGES INTO FUNCTION BLOCKS   (unchanged behaviour)
# ----------------------------------------------------------------------------

@dataclass
class FunctionBlock:
    normalized_name: str
    version: str
    text: str = ""
    page_span: list[int] = field(default_factory=list)


def segment_into_function_blocks(pages: Iterable[str]) -> list[FunctionBlock]:
    """Open a new block only when function IDENTITY (name, version) changes."""
    blocks: list[FunctionBlock] = []
    current: FunctionBlock | None = None

    for page_no, page_text in enumerate(pages):
        header = detect_function_header(page_text)
        if header is None:
            if current is not None:
                current.text += "\n" + page_text
                current.page_span.append(page_no)
            continue

        name, version = header
        identity = (name, version)
        current_identity = (current.normalized_name, current.version) if current else None
        if identity != current_identity:
            current = FunctionBlock(normalized_name=name, version=version)
            blocks.append(current)
        current.text += "\n" + page_text
        current.page_span.append(page_no)

    return blocks


# ----------------------------------------------------------------------------
# 3. TOKEN BUDGET   (real encoder if reachable, char-estimate fallback)
# ----------------------------------------------------------------------------

class _Tokenizer:
    """text-embedding-3-* uses the cl100k_base BPE. If tiktoken can't be loaded
    (offline / blocked), fall back to a ~4-chars-per-token estimate. The budget
    is kept conservative so the estimate's drift never reaches the hard 8191 cap."""

    def __init__(self) -> None:
        self._enc = None
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        return max(1, len(text) // 4)

    def truncate(self, text: str, n_tokens: int) -> str:
        if self._enc is not None:
            return self._enc.decode(self._enc.encode(text)[:n_tokens])
        return text[: n_tokens * 4]


_TOK = _Tokenizer()


# ----------------------------------------------------------------------------
# 4. BOTTOM-UP CHUNKING:  anchor (label mention) -> bounded grown window
# ----------------------------------------------------------------------------

# A dotted, _suffixed label path:  CoElM_SM01_I.CoElM_StMWaitTOnD_I
_LABEL_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*(?:\.[A-Za-z0-9_]+)+\b")

# Lines that should STOP window growth. Default: a numbered sub-heading like
# "36.1.2 Handling of ...". Tune this to your document, or pass boundary_re=None
# to disable boundary-stopping and rely only on neighbour-anchor + line + token caps.
_DEFAULT_BOUNDARY_RE = re.compile(r"^\s*\d+(?:\.\d+){1,}\s+\S")


@dataclass
class LabelChunk:
    """ONE mention of a label plus its bounded local context."""
    label_path: str
    normalized_function_name: str
    version: str
    doc_role: str                 # "old" | "new"  -> join DIRECTION, and id salt
    anchor_line: int              # line index (within block) where label was named
    chunk_index: int              # 0-based occurrence of this label within the block
    text: str                     # the grown window

    @property
    def chunk_id(self) -> str:
        # Include the occurrence index and text hash so repeated mentions on the
        # same line remain distinct inside a single Chroma upsert batch.
        text_hash = hashlib.sha1(self.text.encode()).hexdigest()[:12]
        raw = (
            f"{self.doc_role}|{self.normalized_function_name}|{self.version}"
            f"|{self.label_path}|{self.anchor_line}|{self.chunk_index}|{text_hash}"
        )
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def embedding_text(self) -> str:
        return (f"Label: {self.label_path}\n"
                f"Function: {self.normalized_function_name}\n"
                f"Version: {self.version}\n"
                f"{self.text}")


def _strip_furniture(lines: list[str]) -> str:
    """Drop repeated page-header furniture from a window before embedding."""
    kept = [ln for ln in lines if not any(p.search(ln) for p in _HEADER_PATTERNS)]
    return "\n".join(kept).strip()


def _grow_window(
    line_tok: list[int],
    anchor_idx: int,
    lo_limit: int,
    hi_limit: int,
    max_tokens: int,
    is_boundary: list[bool],
) -> tuple[int, int]:
    """Grow [up, down] outward from the anchor, alternately, stopping at the
    first of: limit reached, structural boundary, or token budget exhausted."""
    up = down = anchor_idx
    cur = line_tok[anchor_idx]
    up_open = up > lo_limit
    down_open = down < hi_limit

    while up_open or down_open:
        progressed = False
        if up_open:
            ni = up - 1
            if ni < lo_limit or is_boundary[ni]:
                up_open = False
            elif cur + line_tok[ni] <= max_tokens:
                up, cur, progressed = ni, cur + line_tok[ni], True
            else:
                up_open = False
        if down_open:
            nj = down + 1
            if nj > hi_limit or is_boundary[nj]:
                down_open = False
            elif cur + line_tok[nj] <= max_tokens:
                down, cur, progressed = nj, cur + line_tok[nj], True
            else:
                down_open = False
        if not progressed:
            break
    return up, down


def extract_label_chunks(
    block: FunctionBlock,
    doc_role: str,
    *,
    max_lines_before: int = 12,
    max_lines_after: int = 40,
    max_tokens: int = 6000,
    boundary_re: re.Pattern | None = _DEFAULT_BOUNDARY_RE,
) -> list[LabelChunk]:
    """Bottom-up: find every label mention (anchor), grow a bounded context
    window around each, emit one chunk per mention."""
    lines = [ln.rstrip() for ln in block.text.splitlines()]
    if not lines:
        return []

    # Precompute per-line token counts (O(1) incremental growth) and boundaries.
    line_tok = [_TOK.count(ln) for ln in lines]
    is_boundary = [bool(ln.strip()) and bool(boundary_re.search(ln)) if boundary_re else False
                   for ln in lines]

    # Anchors in document order: (line_idx, label_path). One line may anchor many.
    anchors: list[tuple[int, str]] = []
    for idx, ln in enumerate(lines):
        for lbl in _LABEL_RE.findall(ln):
            anchors.append((idx, lbl))
    if not anchors:
        return []

    anchor_lines = sorted({idx for idx, _ in anchors})   # for neighbour lookup

    occurrence: dict[str, int] = defaultdict(int)
    chunks: list[LabelChunk] = []
    for idx, lbl in anchors:
        # nearest anchor strictly above / below -> a window never swallows a neighbour's seed
        pos = bisect.bisect_left(anchor_lines, idx)
        prev_anchor = anchor_lines[pos - 1] if pos > 0 else -1
        nxt = bisect.bisect_right(anchor_lines, idx)
        next_anchor = anchor_lines[nxt] if nxt < len(anchor_lines) else len(lines)

        lo_limit = max(0, prev_anchor + 1, idx - max_lines_before)
        hi_limit = min(len(lines) - 1, next_anchor - 1, idx + max_lines_after)

        up, down = _grow_window(line_tok, idx, lo_limit, hi_limit, max_tokens, is_boundary)

        text = _strip_furniture(lines[up: down + 1])
        # Safety net: a single monster line (giant table row) can exceed budget alone.
        if _TOK.count(text) > max_tokens:
            text = _TOK.truncate(text, max_tokens)

        ci = occurrence[lbl]
        occurrence[lbl] += 1
        chunks.append(LabelChunk(
            label_path=lbl,
            normalized_function_name=block.normalized_name,
            version=block.version,
            doc_role=doc_role,
            anchor_line=idx,
            chunk_index=ci,
            text=text,
        ))
    return chunks


# ----------------------------------------------------------------------------
# 5. EMBEDDINGS  (local by default, remote fallback)
# ----------------------------------------------------------------------------

EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local").strip().lower()

# Local backend
_DEFAULT_LOCAL_BGE_M3 = Path(__file__).resolve().parents[1] / "models" / "bge-m3"
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", str(_DEFAULT_LOCAL_BGE_M3))
LOCAL_EMBED_DEVICE = os.getenv("LOCAL_EMBED_DEVICE", "auto")
LOCAL_EMBED_CHUNK_SIZE = int(os.getenv("LOCAL_EMBED_CHUNK_SIZE", "256"))
LOCAL_EMBED_MAX_LENGTH = int(os.getenv("LOCAL_EMBED_MAX_LENGTH", "2048"))
LOCAL_NORMALIZE = os.getenv("LOCAL_NORMALIZE", "1").strip().lower() not in {"0", "false", "no"}

# Remote backend (Bosch LLM farm)
API_KEY = os.getenv("API_KEY")            # stays None if unset
DEPLOYMENT = os.getenv("EMBED_DEPLOYMENT", "text-embedding-3-large")
API_VERSION = os.getenv("EMBED_API_VERSION", "2024-05-01-preview")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", "./embedding_cache.sqlite3")
EMBED_ENDPOINT = (
    "https://aoai-farm.bosch-temp.com/api/openai/deployments/"
    f"{DEPLOYMENT}/embeddings?api-version={API_VERSION}"
)

_LOCAL_MODEL = None


def _auto_detect_cuda_device() -> str:
    """Return best CUDA device if available, otherwise CPU."""
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _is_bge_m3_target(model_name: str) -> bool:
    model_tag = model_name.replace("\\", "/").lower().rstrip("/")
    return model_tag.endswith("bge-m3") or model_tag == "baai/bge-m3"


def _normalize_local_device(device_spec: str) -> str:
    device = device_spec.strip().lower()
    if device in {"", "auto"}:
        return _auto_detect_cuda_device()
    if device in {"cuda", "gpu"}:
        return "cuda:0"
    if device.isdigit():
        return f"cuda:{device}"
    return device


RESOLVED_LOCAL_DEVICE = _normalize_local_device(LOCAL_EMBED_DEVICE)
DEFAULT_LOCAL_BATCH_SIZE = 128 if RESOLVED_LOCAL_DEVICE.startswith("cuda") else 32
LOCAL_BATCH_SIZE = int(os.getenv("LOCAL_BATCH_SIZE", str(DEFAULT_LOCAL_BATCH_SIZE)))
_default_fp16 = "1" if RESOLVED_LOCAL_DEVICE.startswith("cuda") else "0"
LOCAL_EMBED_USE_FP16 = os.getenv("LOCAL_EMBED_USE_FP16", _default_fp16).strip().lower() not in {"0", "false", "no"}
LOCAL_EMBED_STRICT_DEVICE = os.getenv("LOCAL_EMBED_STRICT_DEVICE", "0").strip().lower() in {"1", "true", "yes"}


if EMBED_BACKEND == "local":
    local_model_path = Path(LOCAL_EMBED_MODEL)
    local_missing = local_model_path.is_absolute() and not local_model_path.exists()
    if local_missing and API_KEY:
        EMBED_BACKEND = "remote"
        print(
            f"[embed] local bge-m3 model not found at '{LOCAL_EMBED_MODEL}', switching to remote backend.",
            file=sys.stderr,
        )


def _embedding_cache_key(text: str) -> str:
    if EMBED_BACKEND == "local":
        raw = f"local|{LOCAL_EMBED_MODEL}|normalize={int(LOCAL_NORMALIZE)}|{text}"
    else:
        raw = f"remote|{DEPLOYMENT}|{API_VERSION}|{text}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _open_embedding_cache() -> sqlite3.Connection:
    conn = sqlite3.connect(EMBED_CACHE_PATH)
    required_cols = {"cache_key", "backend", "model", "config", "embedding_json"}

    # Legacy cache files may have an older embeddings schema.
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
    ).fetchone()
    if row is not None:
        existing_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        if not required_cols.issubset(existing_cols):
            conn.execute("DROP TABLE embeddings")
            conn.commit()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            cache_key TEXT PRIMARY KEY,
            backend TEXT NOT NULL,
            model TEXT NOT NULL,
            config TEXT NOT NULL,
            embedding_json TEXT NOT NULL
        )
        """
    )
    return conn


def _get_cached_embedding(conn: sqlite3.Connection, cache_key: str) -> list[float] | None:
    row = conn.execute(
        "SELECT embedding_json FROM embeddings WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()
    return json.loads(row[0]) if row else None


def _store_cached_embeddings(
    conn: sqlite3.Connection,
    cache_keys: list[str],
    embeddings: list[list[float]],
) -> None:
    if EMBED_BACKEND == "local":
        backend, model, config = "local", LOCAL_EMBED_MODEL, f"normalize={int(LOCAL_NORMALIZE)}"
    else:
        backend, model, config = "remote", DEPLOYMENT, API_VERSION
    conn.executemany(
        """
        INSERT OR REPLACE INTO embeddings
            (cache_key, backend, model, config, embedding_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (cache_key, backend, model, config, json.dumps(embedding))
            for cache_key, embedding in zip(cache_keys, embeddings)
        ],
    )
    conn.commit()


def _get_local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        if not _is_bge_m3_target(LOCAL_EMBED_MODEL):
            raise RuntimeError(
                "Local backend only supports bge-m3. Set LOCAL_EMBED_MODEL to a bge-m3 path/name, "
                "or set EMBED_BACKEND=remote."
            )
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "BGE-M3 local embeddings require 'FlagEmbedding'. Install it in the active environment with: "
                "pip install FlagEmbedding"
            ) from exc

        device = RESOLVED_LOCAL_DEVICE
        try:
            _LOCAL_MODEL = BGEM3FlagModel(
                LOCAL_EMBED_MODEL,
                use_fp16=LOCAL_EMBED_USE_FP16,
                devices=device,
            )
        except Exception:
            if device.startswith("cuda"):
                if LOCAL_EMBED_STRICT_DEVICE:
                    raise RuntimeError(
                        f"Failed to initialize local embedding model on requested CUDA device '{device}'. "
                        "Check CUDA-enabled torch and GPU availability, or use --device cpu."
                    )
                print(
                    "[embed] failed to initialize local model on CUDA, retrying on CPU.",
                    file=sys.stderr,
                )
                _LOCAL_MODEL = BGEM3FlagModel(
                    LOCAL_EMBED_MODEL,
                    use_fp16=False,
                    devices="cpu",
                )
            else:
                raise
    return _LOCAL_MODEL


def _embed_documents_local(texts: list[str], is_query: bool) -> list[list[float]]:
    model = _get_local_model()
    prepared = texts
    # BGE models expect an instruction prefix for best retrieval quality on queries.
    if is_query and _is_bge_m3_target(LOCAL_EMBED_MODEL):
        prepared = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

    outputs = model.encode(
        prepared,
        batch_size=LOCAL_BATCH_SIZE,
        max_length=LOCAL_EMBED_MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vectors = outputs["dense_vecs"]
    if LOCAL_NORMALIZE:
        normalized: list[list[float]] = []
        for row in vectors:
            vals = [float(x) for x in row]
            norm = sum(v * v for v in vals) ** 0.5
            if norm > 0.0:
                vals = [v / norm for v in vals]
            normalized.append(vals)
        return normalized
    return [[float(x) for x in row] for row in vectors]


def _embed_documents_remote(texts: list[str], is_query: bool, max_retries: int) -> list[list[float]]:
    import requests

    if not API_KEY:
        raise RuntimeError("API_KEY is not set. Set API_KEY in this shell, then re-run.")

    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i: i + BATCH_SIZE]
        payload = {"input": chunk}
        # payload["task_type"] = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"  # Gemini only
        for attempt in range(max_retries):
            resp = requests.post(EMBED_ENDPOINT, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                break
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            done = i
            total = len(texts)
            raise RuntimeError(
                f"Embedding failed [{resp.status_code}] after caching {done}/{total} texts: {resp.text[:600]}"
            )
        data = resp.json()["data"]
        data.sort(key=lambda d: d["index"])
        out.extend(item["embedding"] for item in data)
    return out


def embed_documents(texts: list[str], is_query: bool = False, *, max_retries: int = 4) -> list[list[float]]:
    cleaned = [t if t.strip() else " " for t in texts]   # Azure 400s on empty input
    cache_keys = [_embedding_cache_key(text) for text in cleaned]
    out: list[list[float] | None] = [None] * len(cleaned)
    missing: dict[str, str] = {}

    with _open_embedding_cache() as cache:
        for idx, (text, cache_key) in enumerate(zip(cleaned, cache_keys)):
            cached = _get_cached_embedding(cache, cache_key)
            if cached is not None:
                out[idx] = cached
            else:
                missing.setdefault(cache_key, text)

        missing_items = list(missing.items())
        chunk_size = LOCAL_EMBED_CHUNK_SIZE if EMBED_BACKEND == "local" else BATCH_SIZE
        for i in range(0, len(missing_items), chunk_size):
            batch_items = missing_items[i: i + chunk_size]
            batch_keys = [cache_key for cache_key, _ in batch_items]
            chunk = [text for _, text in batch_items]
            if EMBED_BACKEND == "local":
                batch_embeddings = _embed_documents_local(chunk, is_query=is_query)
            elif EMBED_BACKEND == "remote":
                batch_embeddings = _embed_documents_remote(chunk, is_query=is_query, max_retries=max_retries)
            else:
                raise RuntimeError("Unknown EMBED_BACKEND. Use 'local' or 'remote'.")
            _store_cached_embeddings(cache, batch_keys, batch_embeddings)

        for idx, cache_key in enumerate(cache_keys):
            if out[idx] is None:
                cached = _get_cached_embedding(cache, cache_key)
                if cached is None:
                    raise RuntimeError(f"Embedding cache miss after embedding: {cache_key}")
                out[idx] = cached

    return [embedding for embedding in out if embedding is not None]


def _default_collection_name() -> str:
    # Use a backend-specific default to avoid vector-dimension collisions.
    configured = os.getenv("COLLECTION_NAME")
    if configured:
        return configured
    if EMBED_BACKEND == "local":
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", LOCAL_EMBED_MODEL).strip("_").lower()
        return f"labels_local_{safe[:40]}"
    return "labels"


# ----------------------------------------------------------------------------
# 6. LOCAL VECTOR STORE   (Chroma persistent dir, native metadata filtering)
# ----------------------------------------------------------------------------

def store_local(
    chunks: list[LabelChunk],
    embeddings: list[list[float]],
    persist_dir: str = "./label_store",
    collection_name: str | None = None,
) -> None:
    import chromadb

    if collection_name is None:
        collection_name = _default_collection_name()

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    # Chroma enforces an internal max upsert batch size. Use it when exposed,
    # otherwise fall back to a conservative size.
    max_batch_size = None
    if hasattr(client, "get_max_batch_size"):
        try:
            max_batch_size = int(client.get_max_batch_size())
        except Exception:
            max_batch_size = None
    elif hasattr(client, "max_batch_size"):
        try:
            max_batch_size = int(client.max_batch_size)
        except Exception:
            max_batch_size = None

    override = os.getenv("CHROMA_MAX_BATCH_SIZE", "").strip()
    if override:
        try:
            max_batch_size = int(override)
        except ValueError:
            pass

    if max_batch_size is None or max_batch_size <= 0:
        max_batch_size = 5000

    total = len(chunks)
    for start in range(0, total, max_batch_size):
        end = min(start + max_batch_size, total)
        batch_chunks = chunks[start:end]
        batch_embeddings = embeddings[start:end]
        collection.upsert(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c.embedding_text() for c in batch_chunks],
            metadatas=[
                {
                    "normalized_function_name": c.normalized_function_name,  # JOIN key
                    "version": c.version,
                    "doc_role": c.doc_role,                                  # WHERE filter
                    "label_path": c.label_path,                              # reassembly key
                    "chunk_index": c.chunk_index,
                    "anchor_line": c.anchor_line,
                }
                for c in batch_chunks
            ],
        )


# ----------------------------------------------------------------------------
# 7. ORCHESTRATION
# ----------------------------------------------------------------------------

def ingest(pages: list[str], doc_role: str, persist_dir: str = "./label_store", **chunk_kwargs) -> int:
    blocks = segment_into_function_blocks(pages)
    all_chunks: list[LabelChunk] = []
    for block in blocks:
        all_chunks.extend(extract_label_chunks(block, doc_role=doc_role, **chunk_kwargs))
    if not all_chunks:
        return 0
    embeddings = embed_documents([c.embedding_text() for c in all_chunks])
    store_local(all_chunks, embeddings, persist_dir=persist_dir)
    return len(all_chunks)


if __name__ == "__main__":
    import sys
    print("Import this module and call ingest(pages, doc_role).", file=sys.stderr)
