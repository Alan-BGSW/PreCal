"""
Function-scoped calibration-label ingestion pipeline.

Flow:  pages -> function blocks (header-gated) -> per-label records
       (context-aggregated) -> code-aware embeddings -> local vector store
       with metadata filters (normalized_function_name, version, doc_role).

Dependencies:
    pip install requests chromadb

The embedding step uses the Bosch LLM farm:
    export API_KEY="..."
    export EMBED_DEPLOYMENT="text-embedding-3-large"   # optional override
    export EMBED_API_VERSION="2024-05-01-preview"      # optional override
"""

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass, field
from typing import Iterable

# ----------------------------------------------------------------------------
# 1. FUNCTION HEADER DETECTION + NORMALIZATION
# ----------------------------------------------------------------------------

# The function identity is re-declared on every page in three surface forms.
# We try the most structured forms first and fall back to the looser one.
_HEADER_PATTERNS = [
    # "Table 14394 FC : ElM_Engine / 1400.0.0; 2"
    re.compile(r"FC\s*:\s*([A-Za-z][A-Za-z0-9_]*)\s*/\s*([\d.]+\s*;\s*\d+)"),
    # "36.1.2 [ElM_Engine 1400.0.0;2] Handling of electrical machine"
    re.compile(r"\[\s*([A-Za-z][A-Za-z0-9_]*)\s+([\d.]+\s*;\s*\d+)\s*\]"),
    # "ElM_Engine 1400.0.0;2 11601 | 20159"  (top-of-page line)
    re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+([\d.]+\s*;\s*\d+)\b"),
]


def normalize_function_name(raw_name: str) -> str:
    """Canonical key used for scope filtering and old<->new joining.

    Deliberately VERSION-FREE: old and new docs must collapse to the same key
    so they can be joined. Direction (old vs new) lives in doc_role, not here.
    """
    return raw_name.strip().lower()


def normalize_version(raw_version: str) -> str:
    """Collapse spacing variants: '1400.0.0; 2' and '1400.0.0;2' -> '1400.0.0;2'."""
    return re.sub(r"\s+", "", raw_version.strip())


def detect_function_header(page_text: str) -> tuple[str, str] | None:
    """Return (normalized_name, normalized_version) if a header is found."""
    for pattern in _HEADER_PATTERNS:
        for line in page_text.splitlines():
            m = pattern.search(line)
            if m:
                return normalize_function_name(m.group(1)), normalize_version(m.group(2))
    return None


# ----------------------------------------------------------------------------
# 2. SEGMENT PAGES INTO FUNCTION BLOCKS  (boundary = header change, not page)
# ----------------------------------------------------------------------------

@dataclass
class FunctionBlock:
    normalized_name: str
    version: str
    text: str = ""              # full text of the block, spanning pages
    page_span: list[int] = field(default_factory=list)


def segment_into_function_blocks(pages: Iterable[str]) -> list[FunctionBlock]:
    """Walk pages in order; open a new block only when function IDENTITY changes.

    'Spans many pages' is a non-issue: the header re-declares identity on every
    page, so we just keep appending until the identity changes.
    """
    blocks: list[FunctionBlock] = []
    current: FunctionBlock | None = None

    for page_no, page_text in enumerate(pages):
        header = detect_function_header(page_text)
        if header is None:
            # No header detected -> assume continuation of current block.
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
# 3. ENUMERATE LABELS + AGGREGATE THEIR SCATTERED CONTEXT  (gather, not cut)
# ----------------------------------------------------------------------------

# A label path looks like  CoElM_SM01_I.CoElM_StMWaitTOnD_I  (dotted, _suffixed).
_LABEL_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*(?:\.[A-Za-z0-9_]+)+\b")


@dataclass
class LabelRecord:
    label_path: str
    normalized_function_name: str
    version: str
    context_fragments: list[str] = field(default_factory=list)

    @property
    def label_id(self) -> str:
        raw = f"{self.normalized_function_name}|{self.version}|{self.label_path}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def embedding_text(self) -> str:
        """The CONTEXT-COMPLETE serialization that actually gets embedded.

        We embed what the function SAYS ABOUT the label, not the bare path,
        because matching runs on contextual meaning across versions.
        """
        body = "\n".join(self.context_fragments)
        return f"Label: {self.label_path}\nFunction: {self.normalized_function_name}\n{body}"


def extract_label_records(block: FunctionBlock) -> list[LabelRecord]:
    """Entity-centric aggregation WITHIN one function block.

    NOTE: this is a pragmatic heuristic for PDF-extracted text. For production,
    prefer the structured CDFX/Excel source for the label set and use this doc
    only to enrich context. Entangled paragraphs are duplicated into each label
    they reference (overlap is correct here, not a bug).
    """
    lines = [ln.strip() for ln in block.text.splitlines() if ln.strip()]

    # Pass 1: find anchor lines (lines that introduce a label path).
    records: dict[str, LabelRecord] = {}
    line_owner: list[str | None] = []   # which label each line is attributed to
    last_label: str | None = None

    for ln in lines:
        found = _LABEL_RE.findall(ln)
        if found:
            for lbl in found:
                if lbl not in records:
                    records[lbl] = LabelRecord(
                        label_path=lbl,
                        normalized_function_name=block.normalized_name,
                        version=block.version,
                    )
                records[lbl].context_fragments.append(ln)
            last_label = found[-1]            # subsequent prose attaches here
            line_owner.append(last_label)
        else:
            # descriptive / mode / reference line -> attach to current anchor
            if last_label is not None:
                records[last_label].context_fragments.append(ln)
            line_owner.append(last_label)

    return list(records.values())


# ----------------------------------------------------------------------------
# 4. EMBEDDINGS  (code-aware, via Voyage API, document mode, batched)
# ----------------------------------------------------------------------------

# Bosch LLM-farm (Azure OpenAI-style) embedding deployment.
# Same route pattern as your chat call, but the path ends in /embeddings.
API_KEY = str(os.getenv("API_KEY"))
# Match Bosch farm's deployment naming style and API version by default.
DEPLOYMENT = os.getenv("EMBED_DEPLOYMENT", "askbosch-prod-farm-openai-text-embedding-3-small")
API_VERSION = os.getenv("EMBED_API_VERSION", "2024-10-21")
EMBED_ENDPOINT = (
    "https://aoai-farm.bosch-temp.com/api/openai/deployments/"
    f"{DEPLOYMENT}/embeddings?api-version={API_VERSION}"
)


def embed_documents(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Embed via the LLM farm. is_query is ignored for OpenAI models (no input_type);
    if you switch the deployment to a Gemini model, map is_query -> task_type."""
    import requests

    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    out: list[list[float]] = []
    BATCH = 128
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]
        payload = {"input": chunk}
        # For a Gemini deployment, add:
        #   payload["task_type"] = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        resp = requests.post(EMBED_ENDPOINT, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()["data"]
        # Sort by index: the API does not guarantee response order matches input order.
        data.sort(key=lambda d: d["index"])
        out.extend(item["embedding"] for item in data)
    return out


# ----------------------------------------------------------------------------
# 5. LOCAL VECTOR STORE  (Chroma persistent dir, native metadata filtering)
# ----------------------------------------------------------------------------

def store_local(
    records: list[LabelRecord],
    embeddings: list[list[float]],
    doc_role: str,                       # "old" or "new"  -> join DIRECTION
    persist_dir: str = "./label_store",
    collection_name: str = "labels",
) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    collection.upsert(
        ids=[r.label_id for r in records],
        embeddings=embeddings,
        documents=[r.embedding_text() for r in records],
        metadatas=[
            {
                "normalized_function_name": r.normalized_function_name,  # JOIN key
                "version": r.version,
                "doc_role": doc_role,                                    # WHERE filter
                "label_path": r.label_path,
            }
            for r in records
        ],
    )


# ----------------------------------------------------------------------------
# 6. ORCHESTRATION
# ----------------------------------------------------------------------------

def ingest(pages: list[str], doc_role: str, persist_dir: str = "./label_store") -> int:
    blocks = segment_into_function_blocks(pages)
    all_records: list[LabelRecord] = []
    for block in blocks:
        all_records.extend(extract_label_records(block))

    if not all_records:
        return 0

    embeddings = embed_documents([r.embedding_text() for r in all_records])
    store_local(all_records, embeddings, doc_role=doc_role, persist_dir=persist_dir)
    return len(all_records)


if __name__ == "__main__":
    import sys
    # Expecting pre-extracted page text; plug your PDF extractor (PyMuPDF/pdfplumber) here.
    # Example: ingest(extract_pages("ElM_Engine_v1400.pdf"), doc_role="old")
    print("Import this module and call ingest(pages, doc_role).", file=sys.stderr)