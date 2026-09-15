"""Retrieve top OLD calibration labels from Chroma without function filtering.

This script reuses the same embedding implementation as ingest_labels_1.py:
- EMBED_BACKEND=local   -> local model (BGEM3FlagModel for bge-m3, else sentence-transformers)
- EMBED_BACKEND=remote  -> Bosch OpenAI-compatible embedding endpoint

Then it sends retrieved context to an Anthropic model exposed via LLM Farm Vertex API.

Example:
    set EMBED_BACKEND=local
    set LOCAL_EMBED_MODEL=C:\\Users\\SHUE1KOR\\Desktop\\CalAi_V2\\A2L_HEX_parser\\models\\bge-m3
    set GENAIPLATFORM_FARM_SUBSCRIPTION_KEY=<your_key>
    set VERTEX_ANTHROPIC_MODEL=claude-haiku-4-5@20251001
    python run_retrieval_claude_no_function.py "CoElM_SM01_I.CoElM_StMWaitTOnD_I" --top-labels 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import requests
from chromadb.errors import NotFoundError

# Make sibling import reliable whether launched from repo root or Ingestion/.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from ingest_labels_1 import _default_collection_name, embed_documents  # noqa: E402


def _stage(message: str) -> None:
    print(f"[stage] {message}")


def _resolve_collection_name(client: chromadb.PersistentClient, preferred_name: str) -> str:
    existing = [c.name for c in client.list_collections()]
    if preferred_name in existing:
        return preferred_name

    if not existing:
        raise RuntimeError(
            "No Chroma collections found in the selected persist directory. "
            "Run ingestion first or pass --persist-dir to the correct store."
        )

    if len(existing) == 1:
        only = existing[0]
        print(
            f"[warn] Preferred collection '{preferred_name}' not found; using only available collection '{only}'."
        )
        return only

    names = ", ".join(existing)
    raise RuntimeError(
        f"Preferred collection '{preferred_name}' not found. "
        f"Available collections: {names}. "
        "Pass --collection-name explicitly or set COLLECTION_NAME."
    )


def _build_where(doc_role: str | None, label_path: str | None) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if doc_role:
        clauses.append({"doc_role": doc_role.strip().lower()})
    if label_path:
        # Exact match to keep behavior deterministic.
        clauses.append({"label_path": label_path.strip()})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve_chunks(
    question: str,
    *,
    persist_dir: str,
    collection_name: str | None,
    k: int,
    doc_role: str | None,
    label_path: str | None,
) -> dict[str, list[Any]]:
    _stage(f"retrieve_chunks:start persist_dir={persist_dir}")
    client = chromadb.PersistentClient(path=persist_dir)
    preferred_name = collection_name or _default_collection_name()
    _stage(f"resolve_collection:preferred={preferred_name}")
    name = _resolve_collection_name(client, preferred_name)
    try:
        col = client.get_collection(name)
    except NotFoundError as exc:
        raise RuntimeError(
            f"Collection '{name}' does not exist in persist_dir='{persist_dir}'."
        ) from exc
    _stage(f"collection:using={name}")

    query_embedding = embed_documents([question], is_query=True)[0]
    where = _build_where(doc_role, label_path)

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where

    _stage(
        "query:execute "
        f"k={k} doc_role={doc_role} "
        f"label_path={'set' if bool(label_path) else 'none'}"
    )
    return col.query(**kwargs)


def _summarize_doc(doc: str, max_len: int = 220) -> str:
    compact = " ".join((doc or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[:max_len].rstrip() + "..."


def _build_semantic_query_from_new_label(
    new_label: str,
    new_rows: list[dict[str, Any]],
    max_chunks: int = 4,
) -> str:
    parts = [
        f"New label: {new_label}",
        "Meaning/context from new document:",
    ]
    for row in new_rows[:max_chunks]:
        parts.append(_summarize_doc(row.get("document", ""), max_len=450))
    return "\n".join(parts)


def _select_top_labels(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    best_by_label: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row.get("label_path") or "").strip()
        if not label:
            continue
        existing = best_by_label.get(label)
        if existing is None or row["distance"] < existing["distance"]:
            best_by_label[label] = row

    ranked = sorted(best_by_label.values(), key=lambda r: r["distance"])
    return ranked[:top_n]


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _normalize_function_name(name: str) -> str:
    return str(name).strip().lower()


def _parse_input_labels_xlsx(
    path: str,
    label_col: str | None = None,
    function_col: str | None = None,
    allowed_functions: set[str] | None = None,
) -> list[str]:
    df = pd.read_excel(path)
    if df.empty:
        return []

    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    selected_label_col = label_col or cols_lower.get("label", df.columns[-1])
    selected_function_col = function_col or cols_lower.get("function_name")
    if selected_label_col not in df.columns:
        raise RuntimeError(
            f"Label column '{selected_label_col}' not found in Excel. "
            f"Available columns: {list(df.columns)}"
        )
    if allowed_functions and selected_function_col is None:
        raise RuntimeError(
            "Function filtering requested but no function column was found. "
            "Provide --function-col explicitly."
        )

    labels: list[str] = []
    current_function = ""
    for _, row in df.iterrows():
        raw_label = row.get(selected_label_col)
        label = "" if _is_blank(raw_label) else str(raw_label).strip()

        if selected_function_col is not None:
            raw_function = row.get(selected_function_col)
            fn = "" if _is_blank(raw_function) else str(raw_function).strip()
            if fn and fn.upper() != "NOT FOUND":
                current_function = fn
            effective_function = fn or current_function
        else:
            effective_function = ""

        if not label or label.startswith("***"):
            continue

        if allowed_functions:
            if not effective_function:
                continue
            if _normalize_function_name(effective_function) not in allowed_functions:
                continue

        labels.append(label)

    seen: set[str] = set()
    unique: list[str] = []
    for item in labels:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _format_top_label_matches(
    *,
    input_label: str,
    matches: list[dict[str, Any]],
) -> str:
    lines = [
        "=== Top Label Matches (old document) ===",
        f"Input new label: {input_label}",
        "Function filter: disabled",
        "",
    ]
    for i, row in enumerate(matches, start=1):
        label = row.get("label_path")
        dist = row.get("distance")
        snippet = _summarize_doc(row.get("document", ""), max_len=260)
        lines.append(f"{i}. {label}")
        lines.append(f"   Similarity evidence: closest semantic match in same function (distance={dist:.4f}).")
        lines.append(f"   Explanation: matched by nearby calibration context and terminology; sample context: {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


def retrieve_top_old_label_matches(
    *,
    new_label: str,
    persist_dir: str,
    collection_name: str | None,
    top_n: int = 3,
    new_seed_k: int = 8,
    old_candidate_k: int = 60,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if top_n < 1:
        raise RuntimeError("top_n must be >= 1")

    # Stage 1: pull context for the provided NEW label in the same function.
    _stage("match_flow:stage1_new_exact_label")
    new_result = retrieve_chunks(
        new_label,
        persist_dir=persist_dir,
        collection_name=collection_name,
        k=new_seed_k,
        doc_role="new",
        label_path=new_label,
    )
    new_rows = _flatten_query_result(new_result)
    _stage(f"stage1_new_exact_label:rows={len(new_rows)}")

    if not new_rows:
        # Fallback: allow semantic lookup in new docs if exact label_path is absent.
        _stage("match_flow:stage1b_new_semantic_fallback")
        new_result = retrieve_chunks(
            new_label,
            persist_dir=persist_dir,
            collection_name=collection_name,
            k=new_seed_k,
            doc_role="new",
            label_path=None,
        )
        new_rows = _flatten_query_result(new_result)
        _stage(f"stage1b_new_semantic_fallback:rows={len(new_rows)}")

    if not new_rows:
        _stage("stage1_result:no_new_seed_context_using_label_only_query")
        semantic_query = (
            f"New label: {new_label}\n"
            "Match OLD labels with the closest calibration meaning."
        )
    else:
        _stage("stage1_result:new_seed_context_found_building_semantic_query")
        semantic_query = _build_semantic_query_from_new_label(new_label, new_rows)

    # Stage 2: search ONLY OLD chunks within the same function using the new context.
    _stage("match_flow:stage2_old_candidate_search")
    old_result = retrieve_chunks(
        semantic_query,
        persist_dir=persist_dir,
        collection_name=collection_name,
        k=old_candidate_k,
        doc_role="old",
        label_path=None,
    )
    old_rows = _flatten_query_result(old_result)
    _stage(f"stage2_old_candidate_search:rows={len(old_rows)}")

    _stage(f"match_flow:stage3_label_dedup_rank_top_n={top_n}")
    top_matches = _select_top_labels(old_rows, top_n=top_n)
    _stage(f"stage3_label_dedup_rank:top_matches={len(top_matches)}")
    return top_matches, new_rows, semantic_query


def _flatten_query_result(result: dict[str, list[Any]]) -> list[dict[str, Any]]:
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    rows: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        rows.append(
            {
                "distance": float(dist),
                "label_path": meta.get("label_path"),
                "normalized_function_name": meta.get("normalized_function_name"),
                "version": meta.get("version"),
                "doc_role": meta.get("doc_role"),
                "anchor_line": meta.get("anchor_line"),
                "chunk_index": meta.get("chunk_index"),
                "document": doc,
            }
        )
    return rows


def _format_context(rows: list[dict[str, Any]], max_chars_per_chunk: int = 2000) -> str:
    blocks: list[str] = []
    for i, row in enumerate(rows, start=1):
        doc = row["document"]
        if len(doc) > max_chars_per_chunk:
            doc = doc[:max_chars_per_chunk] + "\n...[truncated]"

        header = (
            f"[Chunk {i}] label={row['label_path']} "
            f"fn={row['normalized_function_name']} "
            f"ver={row['version']} role={row['doc_role']} "
            f"dist={row['distance']:.4f}"
        )
        blocks.append(header + "\n" + doc)
    return "\n\n".join(blocks)


def _build_batch_match_rows(
    *,
    query_label: str,
    matches: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    semantic_query: str,
    formatted_output: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(matches, start=1):
        snippet = _summarize_doc(str(row.get("document") or ""), max_len=500)
        rows.append(
            {
                "query_label": query_label,
                "rank": rank,
                "match_label": row.get("label_path"),
                "distance": round(float(row.get("distance") or 0.0), 4),
                "match_function_name": row.get("normalized_function_name"),
                "match_version": row.get("version"),
                "match_doc_role": row.get("doc_role"),
                "anchor_line": row.get("anchor_line"),
                "chunk_index": row.get("chunk_index"),
                "new_seed_chunk_count": len(new_rows),
                "semantic_query": semantic_query,
                "match_snippet": snippet,
                "formatted_result": formatted_output,
            }
        )
    return rows


def _build_batch_text_block(
    *,
    query_label: str,
    formatted_output: str,
    semantic_query: str,
    new_rows: list[dict[str, Any]],
    include_context: bool,
) -> str:
    parts = [formatted_output]
    if include_context:
        parts.extend(
            [
                "",
                "=== NEW Label Seed Context ===",
                _format_context(new_rows) if new_rows else "<no new-label context found>",
                "=== End NEW Label Seed Context ===",
                "",
                "=== Semantic Query Sent To OLD Search ===",
                semantic_query,
                "=== End Semantic Query ===",
            ]
        )
    return f"### {query_label}\n" + "\n".join(parts).rstrip()


def _default_batch_output_path(input_xlsx: str, suffix: str) -> str:
    source = Path(input_xlsx)
    return str(source.with_name(source.stem + "_retrieval_results" + suffix))


def _run_single_label_pipeline(
    *,
    input_label: str,
    persist_dir: str,
    collection_name: str | None,
    top_labels: int,
    new_seed_k: int,
    old_candidate_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    top_matches, new_rows, semantic_query = retrieve_top_old_label_matches(
        new_label=input_label,
        persist_dir=persist_dir,
        collection_name=collection_name,
        top_n=top_labels,
        new_seed_k=new_seed_k,
        old_candidate_k=old_candidate_k,
    )
    formatted_output = _format_top_label_matches(
        input_label=input_label,
        matches=top_matches,
    )
    return top_matches, new_rows, semantic_query, formatted_output


def _write_batch_outputs(
    *,
    match_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]],
    output_xlsx: str,
    output_txt: str,
    text_blocks: list[str],
) -> None:
    output_xlsx_path = Path(output_xlsx)
    output_txt_path = Path(output_txt)
    output_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    matches_df = pd.DataFrame(match_rows)
    unmatched_df = pd.DataFrame(unmatched_rows)

    with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
        matches_df.to_excel(writer, index=False, sheet_name="matches")
        unmatched_df.to_excel(writer, index=False, sheet_name="unmatched")

    output_txt_path.write_text("\n\n".join(text_blocks).rstrip() + "\n", encoding="utf-8")


def _claude_endpoint() -> str:
    direct = os.getenv("CLAUDE_ENDPOINT", "").strip() or os.getenv("VERTEX_ENDPOINT", "").strip()
    if direct:
        return direct

    base = os.getenv("CLAUDE_BASE_URL", "https://aoai-farm.bosch-temp.com").rstrip("/")
    model = os.getenv("VERTEX_ANTHROPIC_MODEL", "claude-haiku-4-5@20251001")
    return f"{base}/api/google/v1/publishers/anthropic/models/{model}:rawPredict"


def _vertex_model_name() -> str:
    return os.getenv("VERTEX_ANTHROPIC_MODEL", "claude-opus-4-1@20250805").strip()


def _build_auth_headers(api_key: str, mode: str) -> dict[str, str]:
    mode_norm = (mode or "bearer").strip().lower()
    if mode_norm in {"all", "all-headers", "all-subscription-headers"}:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "api-key": api_key,
            "x-api-key": api_key,
            "Ocp-Apim-Subscription-Key": api_key,
            "subscription-key": api_key,
            "x-subscription-key": api_key,
        }
    if mode_norm == "api-key":
        return {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
    if mode_norm == "x-api-key":
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
        }
    if mode_norm in {"ocp", "ocp-apim", "ocp-apim-subscription-key"}:
        return {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": api_key,
        }
    if mode_norm in {"subscription-key", "subscription"}:
        return {
            "Content-Type": "application/json",
            "subscription-key": api_key,
        }
    if mode_norm in {"x-subscription-key", "x-subscription"}:
        return {
            "Content-Type": "application/json",
            "x-subscription-key": api_key,
        }
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _ask_claude_via_sdk(question: str, context: str, *, temperature: float, max_tokens: int) -> str:
    api_key = (
        os.getenv("GENAIPLATFORM_FARM_SUBSCRIPTION_KEY")
        or os.getenv("CLAUDE_API_KEY")
        or os.getenv("API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Set GENAIPLATFORM_FARM_SUBSCRIPTION_KEY (or CLAUDE_API_KEY/API_KEY) for Claude calls."
        )

    try:
        # Imported lazily so script still works without SDK package.
        from anthropic import AnthropicVertex  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Anthropic SDK is not available. Install it with: pip install anthropic google-genai"
        ) from exc

    base_url = os.getenv("VERTEX_BASE_URL", "https://aoai-farm.bosch-temp.com/api/google/v1").strip()
    model = _vertex_model_name()

    client = AnthropicVertex(
        access_token=api_key,
        project_id=os.getenv("VERTEX_PROJECT_ID", "_"),
        region=os.getenv("VERTEX_REGION", "_"),
        base_url=base_url,
    )

    system_prompt = (
        "You are a calibration assistant. Use only the retrieved context. "
        "If context is insufficient, say so clearly and list what is missing."
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    "Question:\n"
                    f"{question}\n\n"
                    "Retrieved context:\n"
                    f"{context}"
                ),
            }
        ],
    )

    content = getattr(response, "content", None)
    if isinstance(content, list):
        texts = [getattr(block, "text", "") for block in content if hasattr(block, "text")]
        merged = "\n".join(t for t in texts if t).strip()
        if merged:
            return merged

    # Last-resort string representation if SDK schema changes.
    return str(response)


def ask_claude(question: str, context: str, *, temperature: float, max_tokens: int) -> str:
    api_key = (
        os.getenv("GENAIPLATFORM_FARM_SUBSCRIPTION_KEY")
        or os.getenv("CLAUDE_API_KEY")
        or os.getenv("API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Set GENAIPLATFORM_FARM_SUBSCRIPTION_KEY (or CLAUDE_API_KEY/API_KEY) for Claude calls."
        )

    endpoint = _claude_endpoint()
    auth_mode = os.getenv("LLM_FARM_AUTH_MODE", "bearer")

    # Preferred path: Anthropic Vertex SDK (matches official farm sample).
    prefer_sdk = os.getenv("USE_ANTHROPIC_VERTEX_SDK", "1").strip().lower() not in {"0", "false", "no"}
    if prefer_sdk:
        try:
            return _ask_claude_via_sdk(
                question=question,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            # Fall through to raw HTTP fallback below.
            pass

    system_prompt = (
        "You are a calibration assistant. Use only the retrieved context. "
        "If context is insufficient, say so clearly and list what is missing."
    )

    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Question:\n"
                    f"{question}\n\n"
                    "Retrieved context:\n"
                    f"{context}"
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    tried_modes: list[str] = []
    candidate_modes = [
        auth_mode.strip().lower() or "bearer",
        "all-subscription-headers",
        "bearer",
        "api-key",
        "x-api-key",
        "ocp-apim-subscription-key",
        "subscription-key",
        "x-subscription-key",
    ]

    resp = None
    for mode in candidate_modes:
        if mode in tried_modes:
            continue
        tried_modes.append(mode)
        headers = _build_auth_headers(api_key, mode)
        attempt = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        resp = attempt
        if attempt.status_code == 200:
            auth_mode = mode
            break
        if attempt.status_code not in (401, 403):
            # Non-auth error: keep this response and stop retry loop.
            auth_mode = mode
            break

    assert resp is not None

    if resp.status_code != 200:
        key_preview = (api_key[:4] + "..." + api_key[-4:]) if len(api_key) >= 10 else "<too-short>"
        raise RuntimeError(
            "Claude request failed "
            f"[{resp.status_code}] at {endpoint} using auth_mode={auth_mode!r} "
            f"attempted_modes={tried_modes} key={key_preview}: "
            f"{resp.text[:1000]}"
        )

    data = resp.json()
    if isinstance(data, dict) and isinstance(data.get("content"), list):
        texts = [
            block.get("text", "")
            for block in data["content"]
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        merged = "\n".join(t for t in texts if t).strip()
        if merged:
            return merged

    # Fallback for gateways that still proxy OpenAI-style chat-completions.
    try:
        msg = data["choices"][0]["message"]["content"]
    except Exception as exc:  # defensive parse for gateway variance
        raise RuntimeError(f"Unexpected Claude response schema: {json.dumps(data)[:1200]}") from exc

    if isinstance(msg, list):
        # Some gateways return content as OpenAI-style content parts.
        return "\n".join(part.get("text", "") for part in msg if isinstance(part, dict)).strip()
    return str(msg).strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve top OLD labels for one NEW label or for all labels listed in an Excel file."
        )
    )
    parser.add_argument("question", nargs="?", help="Input NEW label path for single-label mode.")
    parser.add_argument("--input-xlsx", default=None,
                        help="Excel file containing labels to process in batch mode.")
    parser.add_argument("--label-col", default=None,
                        help="Override the label column name in --input-xlsx.")
    parser.add_argument("--function-col", default=None,
                        help="Excel function column name. Defaults to function_name.")
    parser.add_argument("--functions", nargs="+", default=None,
                        help=(
                            "Only process labels belonging to these function names. "
                            "Pass one or many names separated by space."
                        ))
    parser.add_argument("--persist-dir", default="./label_store", help="Chroma persistent path.")
    parser.add_argument("--collection-name", default=None, help="Override Chroma collection name.")
    parser.add_argument("--k", type=int, default=8, help="Seed K for NEW-label context retrieval.")
    parser.add_argument("--doc-role", choices=["old", "new"], default=None,
                        help="Filter by document role.")
    parser.add_argument("--label-path", default=None,
                        help="Filter by exact label_path.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Claude sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=900,
                        help="Claude max output tokens.")
    parser.add_argument("--model", default=None,
                        help="Override Vertex Anthropic model for this run.")
    parser.add_argument("--show-context", action="store_true",
                        help="Print NEW seed context and OLD candidate context.")
    parser.add_argument("--top-labels", type=int, default=3,
                        help="Number of OLD labels to return (default: 3).")
    parser.add_argument("--old-candidate-k", type=int, default=60,
                        help="Number of OLD chunks to retrieve before label dedup/ranking.")
    parser.add_argument("--output-xlsx", default=None,
                        help="Batch mode: output Excel path. Defaults to <input>_retrieval_results.xlsx.")
    parser.add_argument("--output-txt", default=None,
                        help="Batch mode: output text path. Defaults to <input>_retrieval_results.txt.")
    return parser.parse_args()


def main() -> int:
    _stage("main:start")
    args = _parse_args()

    if args.model:
        os.environ["VERTEX_ANTHROPIC_MODEL"] = args.model.strip()

    if args.k < 1:
        raise SystemExit("--k must be >= 1")
    if args.top_labels < 1:
        raise SystemExit("--top-labels must be >= 1")
    if args.old_candidate_k < 1:
        raise SystemExit("--old-candidate-k must be >= 1")

    if args.input_xlsx:
        allowed_functions: set[str] | None = None
        if args.functions:
            normalized = [_normalize_function_name(x) for x in args.functions if str(x).strip()]
            allowed_functions = {x for x in normalized if x}
            if not allowed_functions:
                raise SystemExit("--functions was provided but no valid function names were found.")
            _stage(f"main:function_filter count={len(allowed_functions)} values={sorted(allowed_functions)}")

        labels = _parse_input_labels_xlsx(
            args.input_xlsx,
            label_col=args.label_col,
            function_col=args.function_col,
            allowed_functions=allowed_functions,
        )
        _stage(f"main:batch_input labels={len(labels)}")
        if not labels:
            _stage("main:complete_no_batch_labels")
            if allowed_functions:
                print("No labels found in the provided Excel file for the selected functions.")
            else:
                print("No labels found in the provided Excel file.")
            return 2

        output_xlsx = args.output_xlsx or _default_batch_output_path(args.input_xlsx, ".xlsx")
        output_txt = args.output_txt or _default_batch_output_path(args.input_xlsx, ".txt")

        match_rows: list[dict[str, Any]] = []
        unmatched_rows: list[dict[str, Any]] = []
        text_blocks: list[str] = []

        for index, input_label in enumerate(labels, start=1):
            _stage(f"batch:label {index}/{len(labels)} label={input_label}")
            try:
                top_matches, new_rows, semantic_query, formatted_output = _run_single_label_pipeline(
                    input_label=input_label,
                    persist_dir=args.persist_dir,
                    collection_name=args.collection_name,
                    top_labels=args.top_labels,
                    new_seed_k=args.k,
                    old_candidate_k=args.old_candidate_k,
                )
            except Exception as exc:
                unmatched_rows.append(
                    {
                        "query_label": input_label,
                        "reason": str(exc),
                    }
                )
                text_blocks.append(f"### {input_label}\nERROR: {exc}")
                continue

            if not top_matches:
                unmatched_rows.append(
                    {
                        "query_label": input_label,
                        "reason": "No OLD-label matches found",
                    }
                )
                text_blocks.append(
                    _build_batch_text_block(
                        query_label=input_label,
                        formatted_output="No OLD-label matches found for this label.",
                        semantic_query=semantic_query,
                        new_rows=new_rows,
                        include_context=args.show_context,
                    )
                )
                continue

            match_rows.extend(
                _build_batch_match_rows(
                    query_label=input_label,
                    matches=top_matches,
                    new_rows=new_rows,
                    semantic_query=semantic_query,
                    formatted_output=formatted_output,
                )
            )
            text_blocks.append(
                _build_batch_text_block(
                    query_label=input_label,
                    formatted_output=formatted_output,
                    semantic_query=semantic_query,
                    new_rows=new_rows,
                    include_context=args.show_context,
                )
            )

        _write_batch_outputs(
            match_rows=match_rows,
            unmatched_rows=unmatched_rows,
            output_xlsx=output_xlsx,
            output_txt=output_txt,
            text_blocks=text_blocks,
        )

        print(f"[done] processed labels: {len(labels)}")
        print(f"[done] matched rows: {len(match_rows)}")
        print(f"[done] unmatched labels: {len(unmatched_rows)}")
        print(f"[done] Excel -> {output_xlsx}")
        print(f"[done] TXT   -> {output_txt}")
        _stage("main:complete_batch_success")
        return 0 if match_rows else 2

    if not args.question and not args.label_path:
        raise SystemExit("Provide either a single label argument or --input-xlsx.")

    input_label = args.label_path or str(args.question).strip()
    _stage(f"main:input_label label={input_label}")
    top_matches, new_rows, semantic_query, formatted_output = _run_single_label_pipeline(
        input_label=input_label,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        top_labels=args.top_labels,
        new_seed_k=args.k,
        old_candidate_k=args.old_candidate_k,
    )

    if not top_matches:
        _stage("main:complete_no_matches")
        print("No OLD-label matches found for the given NEW label.")
        return 2

    if args.show_context:
        print("=== NEW Label Seed Context ===")
        print(_format_context(new_rows))
        print("=== End NEW Label Seed Context ===\n")
        print("=== Semantic Query Sent To OLD Search ===")
        print(semantic_query)
        print("=== End Semantic Query ===\n")

    print(formatted_output)

    _stage("main:complete_success")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
