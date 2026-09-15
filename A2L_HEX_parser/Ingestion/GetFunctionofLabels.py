"""Find function names for labels by searching in NEW-document embedding space.

Input Excel format (needs_attention_labels_grouped_9_.xlsx):
    Columns: function_full_name | function_name | function_version | label
    - Group header rows look like: *** ACVP_FPT  P2572_V20.0.0 ***
    - function_name is only filled on the header row; data rows have NaN.
    - Labels are in the 'label' column on data rows.

What this script does for each label in the Excel:
     1. Embeds the label text as a query vector.
     2. Searches ChromaDB in NEW-document space (doc_role='new').
     3. If an exact label_path match is present in retrieved candidates,
         extracts function name(s) from metadata.
     4. Writes found and not-found rows to an Excel output file.

Usage:
    python GetFunctionofLabels.py \
        --input-xlsx ./needs_attention_labels_grouped_9_.xlsx \
        --persist-dir ./label_store \
        --top-k 5 \
        --output-xlsx ./matched_results.xlsx

Arguments:
    --input-xlsx    : Path to the grouped labels Excel file (required)
    --persist-dir   : ChromaDB persist directory created during ingestion
                      (default: ./label_store)
    --collection-name : ChromaDB collection name (auto-detected if omitted)
    --top-k         : Number of nearest candidates to inspect per query label
                      (default: 50)
    --output-xlsx   : Output Excel path (default: auto-generated from input name)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd

# ---------------------------------------------------------------------------
# Allow running from repo root or Ingestion/ directory.
# Adjust this path if your ingest_labels_1.py lives elsewhere.
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
INGESTION_DIR = THIS_DIR / "Ingestion"
for p in [str(THIS_DIR), str(INGESTION_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ingest_labels_1 import _default_collection_name, embed_documents
except ImportError:
    # Fallback: define a no-arg callable that returns a fixed name.
    # Replace "label_chunks" with whatever your actual collection is named.
    def _default_collection_name() -> str:  # type: ignore[misc]
        return "label_chunks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_function_name(name: str) -> str:
    return name.strip().lower()


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        try:
            import math
            if math.isnan(value):
                return True
        except Exception:
            pass
    return str(value).strip() == ""


def _build_where(
    *,
    function_name: str | None = None,
    doc_role: str | None = None,
    label_path: str | None = None,
) -> dict[str, Any] | None:
    """Build a ChromaDB $where filter dict from the given constraints."""
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


def _flatten_query_result(result: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Flatten a ChromaDB query result into row dictionaries sorted by distance."""
    docs  = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    rows = [
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
        for doc, meta, dist in zip(docs, metas, dists)
    ]

    rows.sort(key=lambda r: r["distance"])

    return rows


def _cosine_similarity(distance: float) -> float:
    """Convert ChromaDB cosine distance → cosine similarity (distance = 1 - similarity)."""
    return 1.0 - distance


# ---------------------------------------------------------------------------
# Excel parsing
# ---------------------------------------------------------------------------

def _parse_grouped_excel_labels(path: str) -> list[str]:
    """
    Parse the grouped Excel file and return a deduplicated list of labels.

    The file format:
        function_full_name  | function_name | function_version | label
        ACVP_FPT P2572_...  | ACVP_FPT      | P2572_V20.0.0    | *** ACVP_FPT ... ***
        NaN                 | NaN           | NaN              | ACVP_swtCmpbltyFflp_C
        ...

    function_name (if present) is ignored in this no-filter mode.
    """
    df = pd.read_excel(path)

    # Identify columns — tolerant of minor naming differences.
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    label_col = cols_lower.get("label", df.columns[-1])

    labels: list[str] = []
    for _, row in df.iterrows():
        raw_label = row.get(label_col)

        label = "" if _is_blank(raw_label) else str(raw_label).strip()

        # Skip blank labels and group header rows.
        if not label or label.startswith("***"):
            continue

        labels.append(label)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for item in labels:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return unique


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

def match_labels(
    *,
    input_xlsx: str,
    persist_dir: str,
    collection_name: str | None,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        For every label in the input Excel:
            - Embed the label text as query embedding.
            - Query candidates in target_role='new'.
            - Keep rows only if an exact label_path match exists in retrieved candidates.
            - Write function names from metadata.

    Returns:
        matches_df   : One row per (query_label, rank) with match details.
        not_found_df : Rows where no embedding or no candidates were found.
    """
    labels = _parse_grouped_excel_labels(input_xlsx)
    print(f"[info] Loaded {len(labels)} unique labels from Excel.")

    client = chromadb.PersistentClient(path=persist_dir)
    name   = collection_name or _default_collection_name()
    col    = client.get_collection(name)
    print(f"[info] Using ChromaDB collection: '{name}' at '{persist_dir}'")

    match_rows: list[dict[str, Any]]     = []
    not_found_rows: list[dict[str, Any]] = []

    for idx, query_label in enumerate(labels, start=1):
        if idx % 100 == 0 or idx == len(labels):
            print(f"[progress] {idx}/{len(labels)} labels processed...")

        query_embedding = embed_documents([query_label], is_query=True)[0]

        result = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"doc_role": "new"},
            include=["documents", "metadatas", "distances"],
        )

        rows = _flatten_query_result(result)
        exact_rows = [
            r for r in rows
            if str(r.get("label_path") or "") == query_label
        ]

        if not exact_rows:
            not_found_rows.append({
                "query_label": query_label,
                "reason": "No exact label match found in top-k NEW candidates",
            })
            continue

        by_function: dict[str, dict[str, Any]] = {}
        for row in exact_rows:
            fn = str(row.get("normalized_function_name") or "<missing>")
            current = by_function.get(fn)
            if current is None or float(row["distance"]) < float(current["distance"]):
                by_function[fn] = row

        for rank, (fn, row) in enumerate(
            sorted(by_function.items(), key=lambda kv: float(kv[1]["distance"])),
            start=1,
        ):
            snippet = str(row.get("document") or "").strip().replace("\r", "")
            if len(snippet) > 800:
                snippet = snippet[:800] + "\n...[truncated]"

            dist = float(row["distance"])
            match_rows.append({
                "query_label": query_label,
                "rank": rank,
                "exact_match_label": row.get("label_path"),
                "function_name": fn,
                "version": row.get("version"),
                "doc_role": row.get("doc_role"),
                "cosine_similarity": round(_cosine_similarity(dist), 4),
                "distance": round(dist, 4),
                "anchor_line": row.get("anchor_line"),
                "chunk_index": row.get("chunk_index"),
                "match_snippet": snippet,
            })

    return pd.DataFrame(match_rows), pd.DataFrame(not_found_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_output_path(input_xlsx: str) -> str:
    p = Path(input_xlsx)
    return str(p.with_name(p.stem + "_matched_results.xlsx"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each label in the grouped Excel file, find its matching labels "
            "in the NEW software document embedding space and report function names from metadata."
        )
    )
    parser.add_argument(
        "--input-xlsx",
        required=True,
        help="Path to the grouped labels Excel file (e.g. needs_attention_labels_grouped_9_.xlsx).",
    )
    parser.add_argument(
        "--persist-dir",
        default="./label_store",
        help="ChromaDB persist directory (default: ./label_store).",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="ChromaDB collection name. Auto-detected from ingest_labels_1 if omitted.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of NEW-space candidates inspected per label (default: 50).",
    )
    parser.add_argument(
        "--output-xlsx",
        default=None,
        help="Output Excel path. Defaults to <input>_matched_results.xlsx.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_xlsx = args.output_xlsx or _default_output_path(args.input_xlsx)

    matches_df, missing_df = match_labels(
        input_xlsx=args.input_xlsx,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        top_k=max(1, args.top_k),
    )

    os.makedirs(str(Path(output_xlsx).parent), exist_ok=True)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        matches_df.to_excel(writer, index=False, sheet_name="matches")
        missing_df.to_excel(writer, index=False, sheet_name="unmatched")

    output_csv = str(Path(output_xlsx).with_suffix(".csv"))
    matches_df.to_csv(output_csv, index=False)

    print(f"\n[done] Matched:   {len(matches_df)} rows")
    print(f"[done] Unmatched: {len(missing_df)} labels")
    print(f"[done] Excel  ->  {output_xlsx}")
    print(f"[done] CSV    ->  {output_csv}")


if __name__ == "__main__":
    main()