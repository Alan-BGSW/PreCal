from __future__ import annotations

import argparse
import difflib
import math
from pathlib import Path

import pandas as pd


_BGE_MODEL = None
_BGE_MODEL_UNAVAILABLE = False


def _normalize(value: object) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.strip().split()).lower()


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    return round(difflib.SequenceMatcher(a=a, b=b).ratio() * 100.0, 2)


def _normalize_semantic_text(value: str) -> str:
    text = _normalize(value)
    replacements = {
        "standard value": "value",
        "start value": "value",
        "default value": "value",
        "initial value": "value",
        "shall": "must",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _get_bge_model():
    global _BGE_MODEL
    global _BGE_MODEL_UNAVAILABLE

    if _BGE_MODEL_UNAVAILABLE:
        return None
    if _BGE_MODEL is not None:
        return _BGE_MODEL

    try:
        from FlagEmbedding import BGEM3FlagModel

        _BGE_MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    except Exception:
        _BGE_MODEL_UNAVAILABLE = True
        _BGE_MODEL = None
    return _BGE_MODEL


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


def _embed_texts(texts: list[str]) -> dict[str, list[float]]:
    model = _get_bge_model()
    if model is None:
        return {}

    unique = []
    seen = set()
    for text in texts:
        norm = _normalize_semantic_text(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        unique.append(norm)

    if not unique:
        return {}

    try:
        encoded = model.encode(unique, batch_size=16, max_length=512)
        dense = encoded.get("dense_vecs", [])
    except Exception:
        return {}

    out: dict[str, list[float]] = {}
    for idx, text in enumerate(unique):
        vec = dense[idx].tolist() if hasattr(dense[idx], "tolist") else list(dense[idx])
        out[text] = _unit(vec)
    return out


def _semantic_similarity(
    a: str,
    b: str,
    emb_a: dict[str, list[float]],
    emb_b: dict[str, list[float]],
) -> float:
    na = _normalize_semantic_text(a)
    nb = _normalize_semantic_text(b)
    if not na and not nb:
        return 100.0
    if not na or not nb:
        return 0.0

    va = emb_a.get(na)
    vb = emb_b.get(nb)
    if va is not None and vb is not None:
        dot = sum(x * y for x, y in zip(va, vb))
        score = max(0.0, min(1.0, dot)) * 100.0
        return round(score, 2)

    return _similarity(na, nb)


def _validate_columns(df: pd.DataFrame, sheet_name: str) -> None:
    missing: list[str] = []
    if "Label Name" not in df.columns:
        missing.append("Label Name")
    if "Function Name" not in df.columns and "Function Component" not in df.columns:
        missing.append("Function Name or Function Component")
    if "Context" not in df.columns and "Description" not in df.columns:
        missing.append("Context or Description")
    if missing:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing required columns: {', '.join(missing)}"
        )


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "Function Component": "Function Name",
        "Value": "Label Value",
        "Value Type": "Value Structure",
        "Suffix": "Suffix Identified",
    }
    for source_col, target_col in aliases.items():
        if source_col in out.columns and target_col not in out.columns:
            out[target_col] = out[source_col]
    if "Context" not in out.columns and "Description" in out.columns:
        out["Context"] = out["Description"]
    return out


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = _canonicalize_columns(df)
    for col in [
        "Description",
        "Label Value",
        "Value Structure",
        "Suffix Identified",
        "Suffix Category",
        "Section Page(s)",
    ]:
        if col not in out.columns:
            out[col] = ""

    out["Function Name"] = out["Function Name"].fillna("").astype(str).str.strip()
    out["Label Name"] = out["Label Name"].fillna("").astype(str).str.strip()
    out["Context"] = out["Context"].fillna("").astype(str).str.strip()
    for col in [
        "Description",
        "Label Value",
        "Value Structure",
        "Suffix Identified",
        "Suffix Category",
        "Section Page(s)",
    ]:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["_function_norm"] = out["Function Name"].map(_normalize)
    out["_label_norm"] = out["Label Name"].map(_normalize)
    out["_key"] = out["_label_norm"]
    out = out[out["_label_norm"] != ""]
    return out


def _pair_rows_by_context(
    source_rows: pd.DataFrame,
    dest_rows: pd.DataFrame,
    source_emb: dict[str, list[float]],
    dest_emb: dict[str, list[float]],
) -> list[tuple[int, int, float]]:
    if source_rows.empty or dest_rows.empty:
        return []

    scored_pairs: list[tuple[float, int, int, float]] = []
    for s_idx, s_row in source_rows.iterrows():
        s_ctx = str(s_row.get("Context", ""))
        s_fn = _normalize(str(s_row.get("Function Name", "")))
        for d_idx, d_row in dest_rows.iterrows():
            d_ctx = str(d_row.get("Context", ""))
            d_fn = _normalize(str(d_row.get("Function Name", "")))
            ctx_score = _semantic_similarity(s_ctx, d_ctx, source_emb, dest_emb)
            same_function_bonus = 2.0 if s_fn and s_fn == d_fn else 0.0
            scored_pairs.append((ctx_score + same_function_bonus, int(s_idx), int(d_idx), ctx_score))

    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    used_source: set[int] = set()
    used_dest: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for _, s_idx, d_idx, ctx_score in scored_pairs:
        if s_idx in used_source or d_idx in used_dest:
            continue
        used_source.add(s_idx)
        used_dest.add(d_idx)
        matches.append((s_idx, d_idx, ctx_score))

    return matches


def build_matched_sheet(
    source_df: pd.DataFrame,
    destination_df: pd.DataFrame,
    *,
    low_similarity_threshold: float,
) -> pd.DataFrame:
    source = _prepare(source_df)
    dest = _prepare(destination_df)
    source_emb = _embed_texts(source["Context"].tolist())
    dest_emb = _embed_texts(dest["Context"].tolist())

    source = source.reset_index(drop=True)
    dest = dest.reset_index(drop=True)
    scored_pairs: list[tuple[float, int, int, float]] = []

    for s_idx, s_row in source.iterrows():
        s_ctx = str(s_row.get("Context", ""))
        s_fn = _normalize(str(s_row.get("Function Name", "")))
        s_suffix = _normalize(str(s_row.get("Suffix Identified", "")))

        for d_idx, d_row in dest.iterrows():
            d_ctx = str(d_row.get("Context", ""))
            d_fn = _normalize(str(d_row.get("Function Name", "")))
            d_suffix = _normalize(str(d_row.get("Suffix Identified", "")))

            ctx_score = _semantic_similarity(s_ctx, d_ctx, source_emb, dest_emb)
            bonus = 0.0
            if s_fn and d_fn and s_fn == d_fn:
                bonus += 1.5
            if s_suffix and d_suffix and s_suffix == d_suffix:
                bonus += 1.0
            scored_pairs.append((ctx_score + bonus, int(s_idx), int(d_idx), ctx_score))

    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    used_source: set[int] = set()
    used_dest: set[int] = set()
    rows: list[dict[str, object]] = []

    for _, s_idx, d_idx, ctx_sim in scored_pairs:
        if s_idx in used_source or d_idx in used_dest:
            continue
        if ctx_sim < low_similarity_threshold:
            continue

        used_source.add(s_idx)
        used_dest.add(d_idx)

        s = source.iloc[s_idx]
        d = dest.iloc[d_idx]
        rows.append(
            {
                "Source Function Component": s["Function Name"],
                "Destination Function Component": d["Function Name"],
                "Source Label": s["Label Name"],
                "Destination Label": d["Label Name"],
                "Source Description": s.get("Description", ""),
                "Destination Description": d.get("Description", ""),
                "Source Value": s.get("Label Value", ""),
                "Destination Value": d.get("Label Value", ""),
                "Source Value Type": s.get("Value Structure", ""),
                "Destination Value Type": d.get("Value Structure", ""),
                "Source Suffix": s.get("Suffix Identified", ""),
                "Destination Suffix": d.get("Suffix Identified", ""),
                "Source Page(s)": s.get("Section Page(s)", ""),
                "Destination Page(s)": d.get("Section Page(s)", ""),
                "Context Similarity %": ctx_sim,
                "Status": "Semantic Match",
                "Context Flag": "OK",
            }
        )

    matched_df = pd.DataFrame(rows)
    if matched_df.empty:
        return pd.DataFrame(
            columns=[
                "Source Function Component",
                "Destination Function Component",
                "Source Label",
                "Destination Label",
                "Source Description",
                "Destination Description",
                "Source Value",
                "Destination Value",
                "Source Value Type",
                "Destination Value Type",
                "Source Suffix",
                "Destination Suffix",
                "Source Page(s)",
                "Destination Page(s)",
                "Context Similarity %",
                "Status",
                "Context Flag",
            ]
        )

    matched_df = matched_df.sort_values(
        by=["Context Similarity %", "Source Label", "Destination Label"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return matched_df


def build_summary_sheet(matched_df: pd.DataFrame, *, low_similarity_threshold: float) -> pd.DataFrame:
    if matched_df.empty:
        rows = [
            {"Metric": "Total Semantic Matches", "Value": 0},
            {"Metric": "Average Context Similarity %", "Value": 0.0},
            {"Metric": f"Threshold Used (>= {low_similarity_threshold}%)", "Value": low_similarity_threshold},
        ]
        return pd.DataFrame(rows)

    staged = matched_df.copy()
    staged["Context Similarity %"] = pd.to_numeric(staged["Context Similarity %"], errors="coerce")
    matched_count = len(staged)
    avg_similarity = round(staged["Context Similarity %"].mean(), 2) if not staged.empty else 0.0

    rows = [
        {"Metric": "Total Semantic Matches", "Value": matched_count},
        {"Metric": "Average Context Similarity %", "Value": avg_similarity},
        {"Metric": f"Threshold Used (>= {low_similarity_threshold}%)", "Value": low_similarity_threshold},
    ]
    return pd.DataFrame(rows)


def build_function_breakdown_sheet(matched_df: pd.DataFrame, *, low_similarity_threshold: float) -> pd.DataFrame:
    function_col = "Function Component"

    staged = matched_df.copy()
    if "Source Function Component" in staged.columns and "Destination Function Component" in staged.columns:
        staged[function_col] = staged["Source Function Component"].fillna("").astype(str).str.strip()
        empty_source = staged[function_col] == ""
        staged.loc[empty_source, function_col] = (
            staged.loc[empty_source, "Destination Function Component"].fillna("").astype(str).str.strip()
        )
    elif "Function Name" in staged.columns:
        staged[function_col] = staged["Function Name"].fillna("").astype(str).str.strip()
    else:
        staged[function_col] = ""

    if staged.empty:
        return pd.DataFrame(
            columns=[
                function_col,
                "Semantic Matches",
                "Average Context Similarity %",
            ]
        )

    staged["Context Similarity %"] = pd.to_numeric(staged["Context Similarity %"], errors="coerce")
    sim_avg = (
        staged.groupby(function_col, as_index=False)["Context Similarity %"]
        .mean()
        .rename(columns={"Context Similarity %": "Average Context Similarity %"})
    )
    sim_avg["Average Context Similarity %"] = sim_avg["Average Context Similarity %"].round(2)

    label_col = "Source Label" if "Source Label" in staged.columns else "Label Name"
    counts = staged.groupby(function_col, as_index=False)[label_col].count()
    counts = counts.rename(columns={label_col: "Semantic Matches"})

    out = counts.merge(sim_avg, on=function_col, how="left")
    out["Average Context Similarity %"] = out["Average Context Similarity %"].fillna(0.0)

    out = out[
        [
            function_col,
            "Semantic Matches",
            "Average Context Similarity %",
        ]
    ].sort_values(by=[function_col]).reset_index(drop=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Matched sheet by comparing Source and Destination label sheets in an Excel file."
    )
    parser.add_argument("--input-xlsx", required=True, help="Input Excel file path.")
    parser.add_argument(
        "--output-xlsx",
        default=None,
        help="Output Excel file path. Defaults to overwrite input file.",
    )
    parser.add_argument("--source-sheet", default="Source", help="Source sheet name.")
    parser.add_argument("--destination-sheet", default="Destination", help="Destination sheet name.")
    parser.add_argument("--matched-sheet", default="Matched", help="Matched sheet name to write.")
    parser.add_argument("--summary-sheet", default="Summary", help="Summary sheet name to write.")
    parser.add_argument(
        "--function-breakdown-sheet",
        default="Function Breakdown",
        help="Function-wise breakdown sheet name to write.",
    )
    parser.add_argument(
        "--low-sim-threshold",
        type=float,
        default=70.0,
        help="Minimum semantic context similarity required for a match.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_xlsx)
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    output_path = Path(args.output_xlsx) if args.output_xlsx else input_path

    all_sheets = pd.read_excel(input_path, sheet_name=None)
    if args.source_sheet not in all_sheets:
        raise ValueError(f"Sheet not found: {args.source_sheet}")
    if args.destination_sheet not in all_sheets:
        raise ValueError(f"Sheet not found: {args.destination_sheet}")

    source_df = all_sheets[args.source_sheet]
    dest_df = all_sheets[args.destination_sheet]

    _validate_columns(source_df, args.source_sheet)
    _validate_columns(dest_df, args.destination_sheet)

    matched_df = build_matched_sheet(
        source_df,
        dest_df,
        low_similarity_threshold=args.low_sim_threshold,
    )
    summary_df = build_summary_sheet(matched_df, low_similarity_threshold=args.low_sim_threshold)
    function_breakdown_df = build_function_breakdown_sheet(
        matched_df,
        low_similarity_threshold=args.low_sim_threshold,
    )

    all_sheets[args.matched_sheet] = matched_df
    all_sheets[args.summary_sheet] = summary_df
    all_sheets[args.function_breakdown_sheet] = function_breakdown_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    common_count = int((matched_df["Status"] == "Semantic Match").sum()) if not matched_df.empty else 0
    low_sim_count = int((matched_df.get("Context Flag") == "Low Similarity").sum()) if not matched_df.empty else 0

    print(f"[done] Wrote workbook: {output_path}")
    print(f"[done] Semantic-matched labels: {common_count}")
    print(f"[done] Low similarity labels (< {args.low_sim_threshold}%): {low_sim_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
