#!/usr/bin/env python3
"""
apply_frm_start_values.py
=========================
Apply Start Values extracted from an FRM Excel file to calibration outputs.

Input Excel is the workbook created by extract_frm.py, normally with a
"Start Values" sheet and columns like:
    Label name | Start Value | Unit | IF Condition | ...

This script:
  1. Reads Label name + Start Value from the FRM Excel.
  2. Updates matching labels in a CDFX copy when --cdfx is provided.
  3. If --cdfx is not provided, generates a temporary CDFX from A2L+HEX.
  4. Patches the HEX using the existing cdfx_to_a2l_hex.patch_hex_from_cdfx.
  5. Writes an Excel log showing updated/missing/skipped labels.

Examples:
    python apply_frm_start_values.py --frm-xlsx frm_start_values.xlsx --a2l dest.a2l --hex updated.hex

    python apply_frm_start_values.py --frm-xlsx frm_start_values.xlsx --a2l dest.a2l --hex updated.hex \
        --cdfx updated.CDFX --out-cdfx updated_with_frm.CDFX --out-hex updated_with_frm.hex
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import lxml.etree as ET
import pandas as pd

from a2l_hex_to_cdfx import write_cdfx_from_a2l_hex
from cdfx_to_a2l_hex import patch_hex_from_cdfx


VALUE_TAGS = {"V", "VT"}


@dataclass
class FrmValue:
    label: str
    start_value: str
    unit: str = ""
    condition: str = ""


@dataclass
class ApplyLog:
    label: str
    start_value: str
    status: str
    detail: str = ""
    values_written: int = 0
    old_values_sample: str = ""
    new_values_sample: str = ""


def _norm_col(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _pick_col(columns: Iterable[object], *wanted: str) -> object | None:
    normalized = {_norm_col(c): c for c in columns}
    for name in wanted:
        col = normalized.get(_norm_col(name))
        if col is not None:
            return col
    return None


def read_frm_values(xlsx_path: Path) -> list[FrmValue]:
    excel = pd.ExcelFile(xlsx_path)
    sheet_name = "Start Values" if "Start Values" in excel.sheet_names else excel.sheet_names[0]
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str).fillna("")

    label_col = _pick_col(df.columns, "Label name", "Label", "label_name")
    value_col = _pick_col(df.columns, "Start Value", "Standard Value", "start_value")
    unit_col = _pick_col(df.columns, "Unit")
    condition_col = _pick_col(df.columns, "IF Condition", "Condition")

    if label_col is None or value_col is None:
        raise ValueError("FRM Excel must contain 'Label name' and 'Start Value' columns.")

    values: list[FrmValue] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        label = str(row[label_col]).strip()
        start_value = str(row[value_col]).strip()
        if not label or not start_value:
            continue
        if label in seen:
            continue
        seen.add(label)
        values.append(FrmValue(
            label=label,
            start_value=start_value,
            unit=str(row[unit_col]).strip() if unit_col is not None else "",
            condition=str(row[condition_col]).strip() if condition_col is not None else "",
        ))
    return values


def _local_name(elem: ET._Element) -> str:
    return ET.QName(elem).localname if isinstance(elem.tag, str) else ""


def _children(elem: ET._Element, name: str) -> list[ET._Element]:
    return [child for child in elem if _local_name(child) == name]


def _first_child(elem: ET._Element, name: str) -> ET._Element | None:
    children = _children(elem, name)
    return children[0] if children else None


def _short_name(sw: ET._Element) -> str | None:
    sn = _first_child(sw, "SHORT-NAME")
    if sn is None or not sn.text:
        return None
    return sn.text.strip()


def _build_cdfx_index(root: ET._Element) -> dict[str, ET._Element]:
    out: dict[str, ET._Element] = {}
    for elem in root.iter():
        if _local_name(elem) != "SW-INSTANCE":
            continue
        label = _short_name(elem)
        if label:
            out[label] = elem
    return out


def _value_container(sw: ET._Element) -> ET._Element | None:
    sw_value_cont = _first_child(sw, "SW-VALUE-CONT")
    if sw_value_cont is None:
        return None
    return _first_child(sw_value_cont, "SW-VALUES-PHYS")


def _value_elements(sw: ET._Element) -> list[ET._Element]:
    container = _value_container(sw)
    if container is None:
        return []
    return [elem for elem in container.iter() if _local_name(elem) in VALUE_TAGS]


def _clean_start_value(raw: str) -> str:
    value = raw.strip()
    value = re.sub(r"^\(\s*all\s*\)\s*", "", value, flags=re.IGNORECASE).strip()
    return value


def _split_explicit_values(value: str) -> list[str]:
    if not value:
        return []
    if re.search(r"[,;|]", value):
        return [part.strip() for part in re.split(r"[,;|]", value) if part.strip()]
    parts = value.split()
    if len(parts) > 1 and all(_looks_numeric(part) for part in parts):
        return parts
    return [value]


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _new_values_for_elements(start_value: str, count: int) -> list[str]:
    cleaned = _clean_start_value(start_value)
    explicit_values = _split_explicit_values(cleaned)
    if not explicit_values:
        return []
    if len(explicit_values) == count:
        return explicit_values
    if len(explicit_values) == 1:
        return explicit_values * count
    if len(explicit_values) < count:
        return explicit_values + [explicit_values[-1]] * (count - len(explicit_values))
    return explicit_values[:count]


def _sample(values: list[str], limit: int = 8) -> str:
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f", ... ({len(values)} total)"


def apply_values_to_cdfx(frm_values: list[FrmValue], cdfx_path: Path) -> list[ApplyLog]:
    parser = ET.XMLParser(remove_blank_text=False)
    tree = ET.parse(str(cdfx_path), parser)
    root = tree.getroot()
    index = _build_cdfx_index(root)

    logs: list[ApplyLog] = []
    for item in frm_values:
        sw = index.get(item.label)
        if sw is None:
            logs.append(ApplyLog(item.label, item.start_value, "missing_in_cdfx"))
            continue

        elems = _value_elements(sw)
        if not elems:
            logs.append(ApplyLog(item.label, item.start_value, "skipped", "No V/VT value nodes found"))
            continue

        new_values = _new_values_for_elements(item.start_value, len(elems))
        if not new_values:
            logs.append(ApplyLog(item.label, item.start_value, "skipped", "Empty start value after cleanup"))
            continue

        old_values = [(elem.text or "").strip() for elem in elems]
        for elem, value in zip(elems, new_values):
            elem.text = value

        status = "unchanged" if old_values == new_values else "updated"
        logs.append(ApplyLog(
            label=item.label,
            start_value=item.start_value,
            status=status,
            values_written=len(new_values),
            old_values_sample=_sample(old_values),
            new_values_sample=_sample(new_values),
        ))

    tree.write(str(cdfx_path), encoding="utf-8", xml_declaration=True)
    return logs


def write_log(logs: list[ApplyLog], patch_info: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_log = pd.DataFrame([asdict(row) for row in logs])
    status_counts = (
        df_log.groupby("status", dropna=False).size().reset_index(name="count")
        if not df_log.empty else pd.DataFrame(columns=["status", "count"])
    )
    df_patch = pd.DataFrame([{
        "patched_labels_in_hex": patch_info.get("patched", 0),
        "axes_patched": patch_info.get("axes_patched", 0),
        "skipped_no_char": patch_info.get("skipped_no_char", 0),
        "skipped_no_layout": patch_info.get("skipped_no_layout", 0),
        "errors_total": patch_info.get("errors_total", 0),
        "warnings_total": patch_info.get("warnings_total", 0),
        "out_hex": patch_info.get("out_hex", ""),
    }])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_log.to_excel(writer, index=False, sheet_name="FRM Apply Log")
        status_counts.to_excel(writer, index=False, sheet_name="Summary")
        df_patch.to_excel(writer, index=False, sheet_name="HEX Patch Summary")
        pd.DataFrame({"warning": patch_info.get("warnings", [])}).to_excel(
            writer, index=False, sheet_name="Warnings")
        pd.DataFrame({"error": patch_info.get("errors", [])}).to_excel(
            writer, index=False, sheet_name="Errors")


def _default_out_hex(hex_path: Path) -> Path:
    return hex_path.with_name(hex_path.stem + "_frm_start_values" + hex_path.suffix)


def _default_out_cdfx(cdfx_path: Path | None, out_hex: Path) -> Path:
    if cdfx_path is not None:
        return cdfx_path.with_name(cdfx_path.stem + "_frm_start_values.CDFX")
    return out_hex.with_suffix(".CDFX")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Apply FRM Excel Start Values to CDFX and patched HEX outputs."
    )
    ap.add_argument("--frm-xlsx", required=True, type=Path, help="Excel created by extract_frm.py")
    ap.add_argument("--a2l", required=True, type=Path, help="Destination A2L used for HEX patching")
    ap.add_argument("--hex", required=True, type=Path, help="HEX file to patch")
    ap.add_argument("--cdfx", type=Path, default=None, help="Optional CDFX to update before HEX patching")
    ap.add_argument("--out-hex", type=Path, default=None, help="Output HEX path")
    ap.add_argument("--out-cdfx", type=Path, default=None, help="Output CDFX path; defaults to *_frm_start_values.CDFX when --cdfx is provided")
    ap.add_argument("--log-xlsx", type=Path, default=None, help="Excel apply log path")
    args = ap.parse_args(argv)

    for path_arg, label in ((args.frm_xlsx, "FRM Excel"), (args.a2l, "A2L"), (args.hex, "HEX")):
        if not path_arg.exists():
            raise FileNotFoundError(f"{label} not found: {path_arg}")
    if args.cdfx is not None and not args.cdfx.exists():
        raise FileNotFoundError(f"CDFX not found: {args.cdfx}")

    out_hex = args.out_hex or _default_out_hex(args.hex)
    out_cdfx = args.out_cdfx or (_default_out_cdfx(args.cdfx, out_hex) if args.cdfx else None)
    log_xlsx = args.log_xlsx or out_hex.with_name(out_hex.stem + "_frm_apply_log.xlsx")

    frm_values = read_frm_values(args.frm_xlsx)
    if not frm_values:
        raise ValueError("No FRM start values found in the Excel file.")

    with tempfile.TemporaryDirectory(prefix="frm_start_values_") as tmp_dir:
        if args.cdfx is not None:
            working_cdfx = out_cdfx or _default_out_cdfx(args.cdfx, out_hex)
            working_cdfx.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.cdfx, working_cdfx)
        else:
            working_cdfx = out_cdfx or Path(tmp_dir) / "frm_start_values_working.CDFX"
            write_cdfx_from_a2l_hex(args.a2l, args.hex, working_cdfx)

        logs = apply_values_to_cdfx(frm_values, working_cdfx)
        patch_info = patch_hex_from_cdfx(args.a2l, args.hex, working_cdfx, out_hex)
        write_log(logs, patch_info, log_xlsx)

    updated = sum(1 for row in logs if row.status == "updated")
    unchanged = sum(1 for row in logs if row.status == "unchanged")
    missing = sum(1 for row in logs if row.status == "missing_in_cdfx")
    skipped = sum(1 for row in logs if row.status == "skipped")

    print(f"FRM labels read: {len(frm_values)}")
    print(f"CDFX labels updated: {updated}")
    print(f"CDFX labels unchanged: {unchanged}")
    print(f"Missing in CDFX: {missing}")
    print(f"Skipped: {skipped}")
    print(f"HEX labels patched: {patch_info.get('patched', 0)}")
    print(f"Output HEX: {out_hex}")
    if args.cdfx is not None or args.out_cdfx is not None:
        print(f"Output CDFX: {working_cdfx}")
    print(f"Log Excel: {log_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())