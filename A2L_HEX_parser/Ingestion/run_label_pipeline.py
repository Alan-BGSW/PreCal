from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from extract_labels_llm_to_excel import extract_label_rows_from_pdf, rows_to_dataframe
from match_labels_excel import (
    build_function_breakdown_sheet,
    build_matched_sheet,
    build_summary_sheet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full pipeline: extract labels from source and destination PDFs, "
            "match them, and write one Excel workbook with five sheets."
        )
    )
    parser.add_argument("--source-pdf", required=True, help="Path to source PDF document.")
    parser.add_argument("--destination-pdf", required=True, help="Path to destination PDF document.")
    parser.add_argument("--output-xlsx", required=True, help="Path to output Excel file (.xlsx).")
    parser.add_argument("--max-input-tokens", type=int, default=2800, help="Approx input token budget per chunk.")
    parser.add_argument("--overlap-lines", type=int, default=6, help="Line overlap between chunks.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--max-output-tokens", type=int, default=1200, help="LLM max output tokens per chunk.")
    parser.add_argument(
        "--low-sim-threshold",
        type=float,
        default=70.0,
        help="Minimum semantic context similarity required for a match.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_xlsx)

    print("[pipeline] Step 1/2: extracting Source labels...")
    source_rows = extract_label_rows_from_pdf(
        args.source_pdf,
        max_input_tokens=args.max_input_tokens,
        overlap_lines=args.overlap_lines,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    source_df = rows_to_dataframe(source_rows)
    print(f"[pipeline] Source labels extracted: {len(source_df)}")

    print("[pipeline] Step 1/2: extracting Destination labels...")
    destination_rows = extract_label_rows_from_pdf(
        args.destination_pdf,
        max_input_tokens=args.max_input_tokens,
        overlap_lines=args.overlap_lines,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    destination_df = rows_to_dataframe(destination_rows)
    print(f"[pipeline] Destination labels extracted: {len(destination_df)}")

    print("[pipeline] Step 2/2: matching labels and computing metrics...")
    matched_df = build_matched_sheet(
        source_df,
        destination_df,
        low_similarity_threshold=args.low_sim_threshold,
    )
    summary_df = build_summary_sheet(
        matched_df,
        low_similarity_threshold=args.low_sim_threshold,
    )
    function_breakdown_df = build_function_breakdown_sheet(
        matched_df,
        low_similarity_threshold=args.low_sim_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        source_df.to_excel(writer, sheet_name="Source", index=False)
        destination_df.to_excel(writer, sheet_name="Destination", index=False)
        matched_df.to_excel(writer, sheet_name="Matched", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        function_breakdown_df.to_excel(writer, sheet_name="Function Breakdown", index=False)

    semantic_matches = int((matched_df["Status"] == "Semantic Match").sum()) if not matched_df.empty else 0

    print(f"[done] Workbook written: {output_path}")
    print("[done] Sheets: Source, Destination, Matched, Summary, Function Breakdown")
    print(f"[done] Semantic-matched labels (>= {args.low_sim_threshold}%): {semantic_matches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
