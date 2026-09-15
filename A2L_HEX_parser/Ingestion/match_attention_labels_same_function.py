"""Match function-grouped labels against Chroma with same-function filtering.

Input:
- Step-5 grouped Excel output (Needs Attention Grouped), or any Excel file with
  label/function columns.

For each input (label, function):
1. Find the query label in one doc role (default: old).
2. Build a query embedding from that label's stored chunk embeddings.
3. Query top-N candidates constrained to the same function name and target doc role
   (default: new).
4. Return top-K unique label matches ranked by cosine distance/similarity.

Example:
    python Ingestion/match_attention_labels_same_function.py \
      --input-xlsx ./needs_attention_labels_grouped.xlsx \
      --persist-dir ./Ingestion/label_store \
      --query-role old \
      --target-role new \
      --top-k 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd

# Make sibling imports reliable when launched from repo root or Ingestion/.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from ingest_labels_1 import _default_collection_name  # noqa: E402


def _normalize_function_name(name: str) -> str:
    return name.strip().lower()


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _build_where(
    *,
    function_name: str | None = None,
    doc_role: str | None = None,
    label_path: str | None = None,
) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []

    if function_name:
        clauses.append({"normalized_function_name": _normalize_function_name(function_name)})
    if doc_role:
        clauses.append({"doc_role": doc_role.strip().lower()})
    if label_path:
        clauses.append({"label_path": label_path.strip()})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _iter_input_label_function_pairs(
    df: pd.DataFrame,
    *,
    label_col: str | None,
    function_col: str | None,
) -> list[tuple[str, str]]:
    if df.empty:
        return []

    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    selected_label_col = label_col
    if selected_label_col is None:
        selected_label_col = cols_lower.get("label", df.columns[0])

    selected_function_col = function_col
    if selected_function_col is None:
        selected_function_col = cols_lower.get("function_name")

    out: list[tuple[str, str]] = []

    if selected_function_col is not None:
        current_fn = ""
        for _, row in df.iterrows():
            raw_label = row.get(selected_label_col)
            raw_fn = row.get(selected_function_col)

            label = "" if _is_blank(raw_label) else str(raw_label).strip()
            fn = "" if _is_blank(raw_fn) else str(raw_fn).strip()

            if fn and fn.upper() != "NOT FOUND":
                current_fn = fn

            if not label:
                continue

            # Group header row in step-5 output: *** FunctionName Version ***
            if label.startswith("***"):
                continue

            effective_fn = fn or current_fn
            if not effective_fn or effective_fn.upper() == "NOT FOUND":
                continue

            out.append((label, effective_fn))
    else:
        raise ValueError(
            "No function column found. Provide --function-col for your input file."
        )

    # Remove duplicates while preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _average_embedding(embeddings: list[list[float]]) -> list[float] | None:
    if not embeddings:
        return None
    dim = len(embeddings[0])
    if dim == 0:
        return None

    sums = [0.0] * dim
    for vec in embeddings:
        if len(vec) != dim:
            continue
        for i, v in enumerate(vec):
            sums[i] += float(v)

    count = float(len(embeddings))
    if count == 0.0:
        return None
    return [v / count for v in sums]


def _get_query_embedding_from_store(
    col: Any,
    *,
    label_path: str,
    function_name: str,
    query_role: str,
    max_chunks: int,
) -> list[float] | None:
    where = _build_where(
        function_name=function_name,
        doc_role=query_role,
        label_path=label_path,
    )
    if where is None:
        return None

    data = col.get(
        where=where,
        include=["embeddings", "documents", "metadatas"],
        limit=max_chunks,
    )
    embeddings = data.get("embeddings") or []
    if embeddings:
        return _average_embedding(embeddings)
    return None


def _choose_unique_label_results(result: dict[str, list[Any]], top_k: int) -> list[dict[str, Any]]:
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

    rows.sort(key=lambda r: r["distance"])

    picked: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for row in rows:
        lbl = str(row.get("label_path") or "")
        if not lbl or lbl in seen_labels:
            continue
        picked.append(row)
        seen_labels.add(lbl)
        if len(picked) >= top_k:
            break
    return picked


def _cosine_similarity_from_distance(distance: float) -> float:
    # Chroma returns cosine distance when collection is hnsw cosine:
    # distance = 1 - cosine_similarity.
    return 1.0 - distance


def match_labels(
    *,
    input_xlsx: str,
    persist_dir: str,
    collection_name: str | None,
    query_role: str,
    target_role: str,
    top_k: int,
    candidate_pool: int,
    max_query_chunks: int,
    label_col: str | None,
    function_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_in = pd.read_excel(input_xlsx)
    pairs = _iter_input_label_function_pairs(
        df_in,
        label_col=label_col,
        function_col=function_col,
    )

    client = chromadb.PersistentClient(path=persist_dir)
    name = collection_name or _default_collection_name()
    col = client.get_collection(name)

    rows: list[dict[str, Any]] = []
    not_found_rows: list[dict[str, Any]] = []

    for query_label, function_name in pairs:
        query_embedding = _get_query_embedding_from_store(
            col,
            label_path=query_label,
            function_name=function_name,
            query_role=query_role,
            max_chunks=max_query_chunks,
        )

        if query_embedding is None:
            not_found_rows.append(
                {
                    "query_label": query_label,
                    "query_function_name": function_name,
                    "query_role": query_role,
                    "target_role": target_role,
                    "reason": "Exact label not found in query role under same function",
                    "query_mode": "not_found",
                }
            )
            continue

        query_mode = "stored_embedding"

        where = _build_where(function_name=function_name, doc_role=target_role)
        if where is None:
            continue

        result = col.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k, candidate_pool),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        picked = _choose_unique_label_results(result, top_k=top_k)

        if not picked:
            not_found_rows.append(
                {
                    "query_label": query_label,
                    "query_function_name": function_name,
                    "query_role": query_role,
                    "target_role": target_role,
                    "reason": "No same-function candidates found for target role",
                    "query_mode": query_mode,
                }
            )
            continue

        for rank, row in enumerate(picked, start=1):
            snippet = str(row["document"] or "").strip().replace("\r", "")
            if len(snippet) > 800:
                snippet = snippet[:800] + "\n...[truncated]"

            dist = float(row["distance"])
            rows.append(
                {
                    "query_label": query_label,
                    "query_function_name": function_name,
                    "query_role": query_role,
                    "target_role": target_role,
                    "rank": rank,
                    "match_label": row["label_path"],
                    "match_function_name": row["normalized_function_name"],
                    "match_version": row["version"],
                    "match_doc_role": row["doc_role"],
                    "distance": dist,
                    "cosine_similarity": _cosine_similarity_from_distance(dist),
                    "anchor_line": row["anchor_line"],
                    "chunk_index": row["chunk_index"],
                    "query_mode": query_mode,
                    "match_snippet": snippet,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(not_found_rows)


def _default_output_path(input_xlsx: str) -> str:
    p = Path(input_xlsx)
    stem = p.stem + "_same_function_top5_matches"
    return str(p.with_name(stem + ".xlsx"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match Step-5 grouped labels against Chroma using same function filter "
            "and target doc role (default: new)."
        )
    )
    parser.add_argument("--input-xlsx", required=True, help="Path to grouped labels Excel file.")
    parser.add_argument(
        "--persist-dir",
        default="./label_store",
        help="Chroma persist directory used during ingestion.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Chroma collection name. Defaults to ingest_labels_1._default_collection_name().",
    )
    parser.add_argument(
        "--query-role",
        default="old",
        choices=["old", "new"],
        help="Doc role from which query label embedding is sourced.",
    )
    parser.add_argument(
        "--target-role",
        default="new",
        choices=["old", "new"],
        help="Doc role to retrieve candidates from.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top K unique labels to return.")
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=50,
        help="Initial candidate pool before collapsing by unique label.",
    )
    parser.add_argument(
        "--max-query-chunks",
        type=int,
        default=30,
        help="Maximum chunks to read for an exact query label in query role.",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Override label column name if your Excel schema is custom.",
    )
    parser.add_argument(
        "--function-col",
        default=None,
        help="Override function column name if your Excel schema is custom.",
    )
    parser.add_argument(
        "--output-xlsx",
        default=None,
        help="Output Excel path. Defaults to <input>_same_function_top5_matches.xlsx",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_xlsx = args.output_xlsx or _default_output_path(args.input_xlsx)

    matches_df, missing_df = match_labels(
        input_xlsx=args.input_xlsx,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        query_role=args.query_role,
        target_role=args.target_role,
        top_k=max(1, int(args.top_k)),
        candidate_pool=max(1, int(args.candidate_pool)),
        max_query_chunks=max(1, int(args.max_query_chunks)),
        label_col=args.label_col,
        function_col=args.function_col,
    )

    os.makedirs(str(Path(output_xlsx).parent), exist_ok=True)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        matches_df.to_excel(writer, index=False, sheet_name="matches")
        missing_df.to_excel(writer, index=False, sheet_name="unmatched_queries")

    output_csv = str(Path(output_xlsx).with_suffix(".csv"))
    matches_df.to_csv(output_csv, index=False)

    print(f"[done] matches: {len(matches_df)}")
    print(f"[done] unmatched queries: {len(missing_df)}")
    print(f"[done] wrote: {output_xlsx}")
    print(f"[done] wrote: {output_csv}")


if __name__ == "__main__":
    main()
