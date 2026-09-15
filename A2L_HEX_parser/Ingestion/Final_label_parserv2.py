from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf as fitz

LABEL_SPLITTER_RE = re.compile(
    r"^(?P<module>[A-Za-z0-9]+)_(?:(?P<instance>[A-Za-z0-9_]+)\.)?"
    r"(?P<prefix>ti|fac|rat|pwr|u|vol|i|cur|st|b|pct|val|num|tq|n)?"
    r"(?P<name>[A-Za-z0-9]+)_(?P<suffix>CA|C|T|M)$",
    flags=re.IGNORECASE,
)

NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
AXIS_DEFINITION_RE = re.compile(
    r"(?P<axis>[XY]):\s*(?:(?P<signal>[A-Za-z0-9_./-]+)\s*)?\[\s*(?P<unit>[^\]]+?)\s*\]\s*\[\s*(?P<size>\d+)\s*\]",
    flags=re.IGNORECASE,
)
AXIS_SIMPLE_VALUES_RE = re.compile(
    rf"\b(?P<axis>[XY])\s*:?[\s,;]+(?P<vals>{NUM}(?:[\s,;]+{NUM})+)\b",
    flags=re.IGNORECASE,
)
VALUE_TABLE_REF_RE = re.compile(r"\bTable\s+(?P<id>\d+)\b", flags=re.IGNORECASE)

START_MARKER_RE = re.compile(r"\b(?:start|standard|default|initial)\s*value\b\s*:??", flags=re.IGNORECASE)
BROADCAST_TOKEN_RE = re.compile(r"\b(?:\(all\)|all)\b", flags=re.IGNORECASE)
ARRAY_WITH_UNIT_RE = re.compile(
    rf"(?:(?:\(all\)|all)\s*)?(?P<vals>{NUM}(?:[\s,;]+{NUM})+)\s*\[(?P<unit>[^\]]*)\]",
    flags=re.IGNORECASE,
)
SCALAR_WITH_UNIT_RE = re.compile(
    rf"(?:(?:\(all\)|all)\s*)?(?P<val>{NUM})\s*\[(?P<unit>[^\]]*)\]",
    flags=re.IGNORECASE,
)

SECTION_HEADING_RE = re.compile(
    r"^\s*\d+(?:\.\d+){0,6}\s+(?:\[(?P<module1>[A-Za-z0-9_]+)\]|(?P<module2>[A-Za-z0-9_]+))"
)
TABLE_TRIGGER_RE = re.compile(r"\[apph_labtab\]", flags=re.IGNORECASE)
TABLE_HEADER_RE = re.compile(r"label\s*name.*description.*start\s*value", flags=re.IGNORECASE)
TABLE_ID_RE = re.compile(r"\bTable\s+\d+\b", flags=re.IGNORECASE)
APPH_TABLE_END_RE = re.compile(
    r"^Caution\b.*\btables\s+above\b|^\d+(?:\.\d+)*\s+System\s+Constants",
    flags=re.IGNORECASE,
)

LIMIT_BETWEEN_RE = re.compile(rf"between\s+(?P<low>{NUM})\s+(?:and|to)\s+(?P<high>{NUM})", re.IGNORECASE)
LIMIT_MIN_MAX_RE = re.compile(
    rf"min\s*[:=]?\s*(?P<low>{NUM})\s*[,; ]+max\s*[:=]?\s*(?P<high>{NUM})",
    re.IGNORECASE,
)
LIMIT_GT_RE = re.compile(
    rf"(?:must\s+be\s+)?(?:greater\s+than|at\s+least|>=|>)\s*(?P<low>{NUM})",
    re.IGNORECASE,
)
LIMIT_LT_RE = re.compile(
    rf"(?:must\s+be\s+)?(?:less\s+than|at\s+most|<=|<)\s*(?P<high>{NUM})",
    re.IGNORECASE,
)

LABEL_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?_[A-Za-z0-9_]+\b")

OBJECT_CLASS_BY_SUFFIX = {
    "C": "VALUE",
    "T": "CURVE",
    "M": "MAP",
    "CA": "VAL_BLK",
}


@dataclass
class CalibrationRecord:
    module: str
    label_name: str
    object_class: str
    physical_prefix: str
    context: str
    start_value: float | list[float] | None
    unit: str | None
    is_broadcast: bool
    limits: dict[str, Any]
    axes: list[dict[str, Any]] | None
    source_reference: dict[str, Any]
    value_table_id: str | None


@dataclass
class ValueTableRecord:
    table_id: str
    label_name: str
    object_class: str
    source_page: int
    axes: dict[str, dict[str, Any]]
    values: dict[str, list[float]]
    raw_text: str | None


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _extract_numbers(text: str) -> list[float]:
    return [float(v) for v in re.findall(NUM, text)]


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    cleaned = unit.strip()
    if cleaned in {"", "-", "−", "--"}:
        return "dimensionless"
    return cleaned


def _extract_pages(document_path: Path) -> list[tuple[int, str]]:
    if document_path.suffix.lower() == ".pdf":
        doc = fitz.open(document_path)
        try:
            return [(page.number + 1, page.get_text("text")) for page in doc]
        finally:
            doc.close()

    text = document_path.read_text(encoding="utf-8", errors="ignore")
    return [(1, text)]


def _extract_table_id(lines: list[str], index: int) -> str | None:
    start = max(0, index - 3)
    end = min(len(lines), index + 2)
    for i in range(start, end):
        match = TABLE_ID_RE.search(lines[i])
        if match:
            return match.group(0)
    return None


def _find_label_in_line(line: str) -> str | None:
    for token in LABEL_TOKEN_RE.findall(line):
        candidate = token.strip().strip(".,;:()[]{}<>\"'|")
        if candidate.lower().startswith("timeouteem_"):
            continue
        if candidate.lower() == "eem_fidxxxxok_c":
            continue
        if LABEL_SPLITTER_RE.match(candidate):
            return candidate
    return None


def _is_section_heading(line: str) -> bool:
    return bool(SECTION_HEADING_RE.match(line))

def _is_lookahead_stop_line(line: str) -> bool:
    """
    Determines if a line is a hard stop for the row-continuation lookahead.
    This prevents the parser from consuming parts of the next section.
    """
    if _is_section_heading(line):
        return True
    # Stop if it looks like an Axis definition (e.g. "X: ...", "Y: ...")
    if re.search(r"^\s*[XY]:\s*", line, re.IGNORECASE):
        return True
    # Stop if we hit a table header/title block (e.g. "Table 4982")
    if re.search(r"^Table\s+\d+\b", line, re.IGNORECASE):
        return True
    # Stop if we hit caution markers or other system blocks
    if re.search(r"^Caution\b|^\d+\s+System\s+Constants", line, re.IGNORECASE):
        return True
    # Stop on explicit date stamps
    if line.startswith("2025-") or line.startswith("2026-"):
        return True
    return False



def _extract_parent_module(line: str) -> str | None:
    match = SECTION_HEADING_RE.match(line)
    if match:
        bracketed = match.group("module1")
        if bracketed and "_" in bracketed and len(bracketed) > 3:
            return bracketed
        plain = match.group("module2")
        if plain and "_" in plain and not plain.lower().startswith("dsd"):
            return plain

    versioned_heading = re.match(
        r"^\s*\d+(?:\.\d+){1,6}\s+\[(?P<module>[A-Za-z0-9_]+)(?:\s|\])",
        line,
    )
    if versioned_heading and "_" in versioned_heading.group("module") and len(versioned_heading.group("module")) > 3:
        return versioned_heading.group("module")
    return None


def _parse_table_sections(page_text: str) -> list[tuple[int, int]]:
    lines = page_text.splitlines()
    triggers: list[int] = []
    for i, raw in enumerate(lines):
        line = _normalize_ws(raw)
        if TABLE_TRIGGER_RE.search(line):
            triggers.append(i)

    if not triggers:
        return []

    intervals: list[tuple[int, int]] = []
    for trig in triggers:
        start = max(0, trig - 2)
        end = len(lines)
        for j in range(trig + 1, len(lines)):
            line = _normalize_ws(lines[j])
            if APPH_TABLE_END_RE.search(line):
                end = j
                break
            if _is_section_heading(line) and j > trig + 2:
                end = j
                break
        intervals.append((start, end))

    intervals.sort()
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _calibration_continuation_end(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if re.search(r"System\s+Constants\s*-\s*Parameters|^\s*\d+\s+System\s+Constants", line, re.IGNORECASE):
            return index
    return len(lines)


def _extract_start_value_and_unit(row_text: str) -> tuple[float | list[float] | None, str | None, bool]:
    marker_match = START_MARKER_RE.search(row_text)
    segment = row_text[marker_match.end():] if marker_match else row_text
    if marker_match and re.search(r"\bsee\s+table\b|\btable\s+\d+\b", segment, re.IGNORECASE):
        return None, None, bool(BROADCAST_TOKEN_RE.search(segment))
    if not marker_match:
        segment = AXIS_DEFINITION_RE.sub(" ", segment)
        segment = AXIS_SIMPLE_VALUES_RE.sub(" ", segment)
    is_broadcast = bool(BROADCAST_TOKEN_RE.search(segment))

    array_matches = list(ARRAY_WITH_UNIT_RE.finditer(segment))
    if array_matches:
        match = array_matches[-1]
        values = _extract_numbers(match.group("vals"))
        unit = _normalize_unit(match.group("unit"))
        return values, unit, is_broadcast

    scalar_matches = list(SCALAR_WITH_UNIT_RE.finditer(segment))
    if scalar_matches:
        match = scalar_matches[-1]
        value = float(match.group("val"))
        unit = _normalize_unit(match.group("unit"))
        return value, unit, is_broadcast

    if marker_match:
        values = _extract_numbers(segment)
        if len(values) > 1:
            return values, None, is_broadcast
        if len(values) == 1:
            return values[0], None, is_broadcast

    return None, None, is_broadcast


def _extract_row_base_context_verbatim(label: str, row_lines_raw: list[str]) -> str:
    merged = "\n".join(line.rstrip() for line in row_lines_raw)
    label_pos = merged.find(label)
    tail = merged[label_pos + len(label):] if label_pos >= 0 else merged
    return tail.lstrip(" |-:")


def _extract_axis_breakpoints_after_definition(
    text: str,
    axis_match: re.Match[str],
    next_start: int | None,
) -> list[float] | None:
    tail_end = next_start if next_start is not None else len(text)
    tail = text[axis_match.end():tail_end]
    candidate = re.search(rf"(?P<vals>{NUM}(?:[\s,;]+{NUM})+)", tail)
    if not candidate:
        return None
    points = _extract_numbers(candidate.group("vals"))
    return points or None


def _extract_axes_from_row(row_text: str) -> list[dict[str, Any]] | None:
    definition_matches = list(AXIS_DEFINITION_RE.finditer(row_text))
    if not definition_matches:
        return None

    axes: list[dict[str, Any]] = []
    for idx, axis_match in enumerate(definition_matches):
        next_start = definition_matches[idx + 1].start() if idx + 1 < len(definition_matches) else None
        axis = axis_match.group("axis").upper()
        signal = axis_match.group("signal")
        unit = _normalize_unit(axis_match.group("unit"))
        size = int(axis_match.group("size"))
        breakpoints = _extract_axis_breakpoints_after_definition(row_text, axis_match, next_start)
        if breakpoints and size > 0 and len(breakpoints) > size:
            breakpoints = breakpoints[:size]

        axes.append(
            {
                "axis": axis,
                "signal": signal if signal else None,
                "unit": unit,
                "size": size,
                "breakpoints": breakpoints,
            }
        )

    return axes


def _extract_explicit_limits(description: str) -> tuple[float | None, float | None, str | None, str | None]:
    between_match = LIMIT_BETWEEN_RE.search(description)
    if between_match:
        low = float(between_match.group("low"))
        high = float(between_match.group("high"))
        return low, high, "explicit_between", f"Found phrase 'between {low} and {high}'"

    min_max_match = LIMIT_MIN_MAX_RE.search(description)
    if min_max_match:
        low = float(min_max_match.group("low"))
        high = float(min_max_match.group("high"))
        return low, high, "explicit_min_max", f"Found phrase 'min {low} max {high}'"

    gt_match = LIMIT_GT_RE.search(description)
    if gt_match:
        low = float(gt_match.group("low"))
        return low, None, "explicit_greater_than", f"Found lower-bound phrase with {low}"

    lt_match = LIMIT_LT_RE.search(description)
    if lt_match:
        high = float(lt_match.group("high"))
        return None, high, "explicit_less_than", f"Found upper-bound phrase with {high}"

    return None, None, None, None


def _infer_implicit_limits(prefix: str | None, unit: str | None) -> tuple[float | None, float | None, str, str]:
    normalized_prefix = (prefix or "").lower()
    normalized_unit = (unit or "").strip().lower()

    if normalized_unit in {"%", "percent"} or normalized_prefix in {"pct", "rat"}:
        return 0.0, 100.0, "implicit_percent_ratio", "Unit/prefix indicates percentage or ratio scale"
    if normalized_prefix == "fac" and normalized_unit in {"", "-", "dimensionless"}:
        return 0.0, 1.0, "implicit_factor", "fac with dimensionless unit is normalized"
    if normalized_prefix == "ti":
        return 0.0, None, "implicit_non_negative", "time prefix implies non-negative duration"
    if normalized_prefix == "b":
        return 0.0, 1.0, "implicit_boolean", "boolean prefix implies 0/1 domain"
    return None, None, "unbounded", "no explicit or implicit domain rule matched"


def _resolve_limits(description: str, prefix: str | None, unit: str | None) -> dict[str, Any]:
    lower, upper, limit_type, rationale = _extract_explicit_limits(description)
    return {
        "lower": lower,
        "upper": upper,
        "type": limit_type or "missing",
        "rationale": rationale or "No explicit limit in source text",
    }


def _extract_value_table_ref(row_text: str) -> str | None:
    marker_match = START_MARKER_RE.search(row_text)
    segment = row_text[marker_match.end():] if marker_match else row_text
    match = VALUE_TABLE_REF_RE.search(segment)
    if not match:
        return None
    return f"Table {match.group('id')}"


def _parse_calibration_row(
    label: str,
    row_lines: list[str],
    row_lines_raw: list[str] | None,
    page_no: int,
    parent_module: str | None,
    calibration_table_id: str | None,
) -> CalibrationRecord | None:
    match = LABEL_SPLITTER_RE.match(label)
    if not match:
        return None

    row_text = _normalize_ws(" ".join(row_lines))
        # --- START OF WORKAROUND FIX ---
    # Manually rebuild the context to ensure correct order and exclude start values.
    context_lines = []
    # The first line contains the label, so we take the part after the label.
    first_line_text = (row_lines_raw or row_lines)[0]
    label_pos = first_line_text.find(label)
    if label_pos != -1:
        # Get the description part on the same line as the label
        description_part = first_line_text[label_pos + len(label):].lstrip(" |-:")
        if description_part:
            context_lines.append(description_part.strip())

    # Process the rest of the lines
    for line in (row_lines_raw or row_lines)[1:]:
        # Stop if we hit a line that defines the start value, to avoid including it
        if START_MARKER_RE.search(line):
            break
        context_lines.append(line.strip())
    
    context = " ".join(context_lines).strip()
    # --- END OF WORKAROUND FIX ---

    start_value, unit, is_broadcast = _extract_start_value_and_unit(row_text)

    suffix = match.group("suffix").upper()
    object_class = OBJECT_CLASS_BY_SUFFIX.get(suffix, "UNKNOWN")
    axes = _extract_axes_from_row(row_text) if object_class in {"CURVE", "MAP", "VAL_BLK"} else None

    limits = _resolve_limits(context, match.group("prefix"), unit)
    value_table_id = _extract_value_table_ref(row_text)

    return CalibrationRecord(
        module=match.group("module"),
        label_name=label,
        object_class=object_class,
        physical_prefix=(match.group("prefix") or "").lower(),
        context=context,
        start_value=start_value,
        unit=unit,
        is_broadcast=is_broadcast,
        limits=limits,
        axes=axes,
        source_reference={
            "table_id": calibration_table_id,
            "page": page_no,
            "parent_module": parent_module,
        },
        value_table_id=value_table_id,
    )


def _is_numeric_line(text: str) -> bool:
    return bool(re.match(r"^\s*[-+0-9eE.,; ]+\s*$", text))


def _extract_series_values(lines: list[str], key: str) -> list[float]:
    key_u = key.upper()
    for i, raw in enumerate(lines):
        line = _normalize_ws(raw)
        upper = line.upper()
        if upper == key_u or upper.startswith(key_u + " "):
            values: list[float] = []
            rest = line[len(key):].strip()
            if rest:
                values.extend(_extract_numbers(rest))
            j = i + 1
            while j < len(lines):
                nxt = _normalize_ws(lines[j])
                nxt_u = nxt.upper()
                if not nxt:
                    break
                if nxt_u in {"X", "Y", "VAL"}:
                    break
                if nxt_u.startswith("TABLE "):
                    break
                if _find_label_in_line(nxt) and values:
                    break
                if not _is_numeric_line(nxt):
                    break
                values.extend(_extract_numbers(nxt))
                j += 1
            return values
    return []


def _parse_map_yx_segments(block_lines: list[str], x_size: int | None, y_size: int | None) -> tuple[list[float], list[float], list[float]]:
    if not y_size or y_size <= 0:
        return [], [], []

    yx_indices: list[int] = []
    for i, raw in enumerate(block_lines):
        if _normalize_ws(raw).upper() == "Y/X":
            yx_indices.append(i)

    if not yx_indices:
        return [], [], []

    x_values: list[float] = []
    y_values: list[float] = []
    z_by_y: dict[float, list[float]] = {}

    for seg_idx, start in enumerate(yx_indices):
        end = yx_indices[seg_idx + 1] if seg_idx + 1 < len(yx_indices) else len(block_lines)
        tokens: list[float] = []
        for line in block_lines[start + 1:end]:
            text = _normalize_ws(line)
            if not text:
                continue
            if text.upper().startswith("TABLE "):
                break
            if _find_label_in_line(text):
                break
            if not _is_numeric_line(text):
                continue
            tokens.extend(_extract_numbers(text))

        if not tokens:
            continue

        if x_size and x_size > 0:
            col_count = x_size
        else:
            den = y_size + 1
            col_count = (len(tokens) - y_size) // den if len(tokens) >= y_size else -1
        if col_count <= 0 or len(tokens) <= col_count:
            continue

        seg_x = tokens[:col_count]
        body = tokens[col_count:]

        idx = 0
        while idx < len(body):
            if idx + 1 + col_count > len(body):
                break
            yv = body[idx]
            idx += 1
            row_vals = body[idx: idx + col_count]
            idx += col_count
            if len(row_vals) != col_count:
                break
            if yv not in z_by_y:
                y_values.append(yv)
                z_by_y[yv] = row_vals

        if not x_values:
            x_values = seg_x

    if x_size and len(x_values) > x_size:
        x_values = x_values[:x_size]

    z_flat: list[float] = []
    if y_size and len(y_values) > y_size:
        y_values = y_values[:y_size]

    for yv in y_values:
        row_vals = z_by_y.get(yv, [])
        if x_size and len(row_vals) > x_size:
            row_vals = row_vals[:x_size]
        z_flat.extend(row_vals)

    return x_values, y_values, z_flat


def _is_table_noise_line(line: str) -> bool:
    text = _normalize_ws(line)
    if not text:
        return True
    if re.match(r"^\d{4}-\d{2}-\d{2}\s*\|\s*\[PVER\]", text):
        return True
    if "Robert Bosch GmbH reserves all rights" in text:
        return True
    if re.search(r"all rights of disposal such as copying and passing on to third parties", text, re.IGNORECASE):
        return True
    if re.match(r"^\d+\s*\|\s*\d+$", text):
        return True
    if re.match(r"^[A-Za-z0-9_]+\s+\d+\.\d+\.\d+;?\d*$", text):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+System\s+Constants", text, re.IGNORECASE):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+Systemconstants", text, re.IGNORECASE):
        return True
    return False


def _extract_value_tables(pages: list[tuple[int, str]]) -> list[ValueTableRecord]:
    all_blocks: list[tuple[str, int, list[str], str]] = []
    all_lines: list[str] = []
    line_pages: list[int] = []
    for page_no, page_text in pages:
        page_lines = page_text.splitlines()
        all_lines.extend(page_lines)
        line_pages.extend([page_no] * len(page_lines))

    starts: list[int] = []
    table_ids: list[str] = []
    for index, raw in enumerate(all_lines):
        match = TABLE_ID_RE.search(raw)
        if match:
            starts.append(index)
            table_ids.append(match.group(0))

    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(all_lines)
        raw_block_lines = [x.rstrip() for x in all_lines[start:end] if x.strip()]
        raw_block_lines = [x for x in raw_block_lines if not _is_table_noise_line(x)]
        block = [_normalize_ws(x) for x in raw_block_lines if _normalize_ws(x)]
        if block:
            all_blocks.append((table_ids[idx], line_pages[start], block, "\n".join(raw_block_lines)))

    value_tables: list[ValueTableRecord] = []

    for table_id, page_no, block_lines, raw_text in all_blocks:
        label_name: str | None = None
        object_class = "UNKNOWN"
        for line in block_lines:
            lbl = _find_label_in_line(line)
            if not lbl:
                continue
            m = LABEL_SPLITTER_RE.match(lbl)
            if not m:
                continue
            suffix = m.group("suffix").upper()
            cls = OBJECT_CLASS_BY_SUFFIX.get(suffix, "UNKNOWN")
            if cls in {"CURVE", "MAP", "VAL_BLK"}:
                label_name = lbl
                object_class = cls
                break

        if not label_name:
            continue

        text = "\n".join(block_lines)
        axes: dict[str, dict[str, Any]] = {}
        for m in AXIS_DEFINITION_RE.finditer(text):
            axis = m.group("axis").upper()
            axes[axis] = {
                "signal": m.group("signal") if m.group("signal") else None,
                "unit": _normalize_unit(m.group("unit")),
                "size": int(m.group("size")),
            }

        x_values = _extract_series_values(block_lines, "X")
        y_values = _extract_series_values(block_lines, "Y")
        val_values = _extract_series_values(block_lines, "VAL")

        if object_class == "MAP":
            x_size = axes.get("X", {}).get("size") if axes.get("X") else None
            y_size = axes.get("Y", {}).get("size") if axes.get("Y") else None
            mx, my, mz = _parse_map_yx_segments(block_lines, x_size, y_size)
            if mx:
                x_values = mx
            if my:
                y_values = my
            if mz:
                val_values = mz

        if x_values and "X" not in axes:
            axes["X"] = {"signal": None, "unit": None, "size": len(x_values)}
        if y_values and "Y" not in axes:
            axes["Y"] = {"signal": None, "unit": None, "size": len(y_values)}

        value_tables.append(
            ValueTableRecord(
                table_id=table_id,
                label_name=label_name,
                object_class=object_class,
                source_page=page_no,
                axes=axes,
                values={
                    "X": x_values,
                    "Y": y_values,
                    "VAL": val_values,
                },
                raw_text=raw_text,
            )
        )

    return value_tables


def _is_continuation_noise_line(line: str) -> bool:
    text = _normalize_ws(line)
    if not text:
        return True
    if _is_label_header_line(text):
        return True
    if re.match(r"^\d{4}-\d{2}-\d{2}\s*\|\s*\[PVER\]", text):
        return True
    if "Robert Bosch GmbH reserves all rights" in text:
        return True
    if re.search(r"all rights of disposal such as copying and passing on to third parties", text, re.IGNORECASE):
        return True
    if re.match(r"^\d+\s*\|\s*\d+$", text):
        return True
    if re.match(r"^[A-Za-z0-9_]+\s+\d+\.\d+\.\d+;?\d*\s+\d+\s*\|\s*\d+$", text):
        return True
    if re.match(r"^[A-Za-z0-9_]+\s+\d+\.\d+\.\d+;?\d*$", text):
        return True
    return False


def _merge_record_axes(record: CalibrationRecord, axes_update: list[dict[str, Any]]) -> None:
    if not axes_update:
        return
    if not record.axes:
        record.axes = axes_update
        return

    existing_by_axis: dict[str, dict[str, Any]] = {}
    for axis in record.axes:
        name = str(axis.get("axis", "")).upper()
        if name:
            existing_by_axis[name] = axis

    for incoming in axes_update:
        axis_name = str(incoming.get("axis", "")).upper()
        if not axis_name:
            continue
        target = existing_by_axis.get(axis_name)
        if target is None:
            record.axes.append(incoming)
            existing_by_axis[axis_name] = incoming
            continue

        if not target.get("signal") and incoming.get("signal"):
            target["signal"] = incoming.get("signal")
        if not target.get("unit") and incoming.get("unit"):
            target["unit"] = incoming.get("unit")
        if not target.get("size") and incoming.get("size"):
            target["size"] = incoming.get("size")
        if not target.get("breakpoints") and incoming.get("breakpoints"):
            target["breakpoints"] = incoming.get("breakpoints")


def _merge_unlabeled_continuation_into_record(record: CalibrationRecord, continuation_lines_raw: list[str]) -> None:
    if not continuation_lines_raw:
        return

    kept_lines: list[str] = []
    for raw in continuation_lines_raw:
        normalized = _normalize_ws(raw)
        if _is_continuation_noise_line(raw):
            continue
        if _is_section_heading(normalized) or APPH_TABLE_END_RE.search(normalized):
            continue
        kept_lines.append(raw.rstrip())

    if not kept_lines:
        return

    addition = "\n".join(kept_lines).strip()
    if addition and addition not in record.context:
        if record.context.strip():
            record.context = f"{record.context.rstrip()}\n{addition}"
        else:
            record.context = addition

    parsed_text = _normalize_ws(" ".join(kept_lines))
    if parsed_text:
        if record.start_value is None or not record.unit:
            start_value, unit, is_broadcast = _extract_start_value_and_unit(parsed_text)
            if record.start_value is None and start_value is not None:
                record.start_value = start_value
            if not record.unit and unit:
                record.unit = unit
            record.is_broadcast = record.is_broadcast or is_broadcast

        if record.value_table_id is None:
            table_ref = _extract_value_table_ref(parsed_text)
            if table_ref:
                record.value_table_id = table_ref

        if record.object_class in {"CURVE", "MAP", "VAL_BLK"}:
            continuation_axes = _extract_axes_from_row(parsed_text)
            if continuation_axes:
                _merge_record_axes(record, continuation_axes)


def parse_document(document_path: Path, max_row_lines: int = 20) -> tuple[list[CalibrationRecord], list[ValueTableRecord]]:
    pages = _extract_pages(document_path)
    parsed_rows: list[CalibrationRecord] = []
    apph_table_open = False
    active_parent_module: str | None = None
    open_table_tail_record: CalibrationRecord | None = None

    for page_no, page_text in pages:
        lines = page_text.splitlines()
        normalized_lines = [_normalize_ws(line) for line in lines]
        for line in normalized_lines:
            candidate = _extract_parent_module(line)
            if candidate:
                active_parent_module = candidate

        sections = _parse_table_sections(page_text)
        if not sections and apph_table_open:
            end = len(normalized_lines)
            for index, line in enumerate(normalized_lines):
                if APPH_TABLE_END_RE.search(line) or _is_section_heading(line):
                    end = index
                    break
            sections = [(0, end)] if end else []
        
        if not sections:
            apph_table_open = False
            open_table_tail_record = None
            continue

        page_tail_record = open_table_tail_record if apph_table_open else None

        for start, end in sections:
            parent_module: str | None = None
            for i in range(start, -1, -1):
                candidate = _extract_parent_module(normalized_lines[i])
                if candidate:
                    parent_module = candidate
                    break
            if parent_module is None:
                parent_module = active_parent_module

            calibration_table_id = _extract_table_id(normalized_lines, start)
            section_rows: list[CalibrationRecord] = []
            section_tail_record: CalibrationRecord | None = page_tail_record if start == 0 else None
            
            cursor = start
            while cursor < end:
                line = normalized_lines[cursor]
                label = _find_label_in_line(line)
                
                if not label:
                    cursor += 1
                    continue

                # --- START OF FIX ---
                # This block handles the multi-line row structure
                row_lines = [line]
                row_lines_raw = [lines[cursor]]
                lookahead = cursor + 1

                # Greedily grab subsequent lines until we hit a new label or a stop condition
                while lookahead < end and len(row_lines) < max_row_lines:
                    next_line = normalized_lines[lookahead]
                    if not next_line.strip(): # Skip empty lines
                        lookahead += 1
                        continue
                    
                    # --- THE FINAL FIX ---
                    # 1. Stop if the next line is a new label
                    if _find_label_in_line(next_line):
                        break
                        
                    # 2. Stop if the next line is any other structural boundary
                    if _is_lookahead_stop_line(next_line):
                        break
                    # ----------------------

                    
                    row_lines.append(next_line)
                    row_lines_raw.append(lines[lookahead])
                    lookahead += 1
                    # --- END OF FIX ---

                record = _parse_calibration_row(
                    label=label,
                    row_lines=row_lines,
                    row_lines_raw=row_lines_raw,
                    page_no=page_no,
                    parent_module=parent_module,
                    calibration_table_id=calibration_table_id,
                )
                if record:
                    section_rows.append(record)
                    section_tail_record = record
                
                cursor = lookahead
            
            parsed_rows.extend(section_rows)
            page_tail_record = section_tail_record

        apph_table_open = bool(sections) and sections[-1][1] == len(normalized_lines)
        open_table_tail_record = page_tail_record if apph_table_open else None

    best_by_label: dict[str, CalibrationRecord] = {}

    def score(row: CalibrationRecord) -> int:
        s = len(row.context)
        s += 100 if row.axes else 0
        s += 1000 if row.start_value is not None else 0
        s += 10 if row.unit else 0
        s += 500 if row.value_table_id else 0
        s += 6 if row.source_reference.get("table_id") else 0
        s += 4 if row.source_reference.get("parent_module") else 0
        return s

    for row in parsed_rows:
        existing = best_by_label.get(row.label_name)
        if existing is None or score(row) > score(existing):
            best_by_label[row.label_name] = row

    value_tables = _extract_value_tables(pages)
    
    return sorted(best_by_label.values(), key=lambda r: (r.module, r.label_name)), value_tables



def _backfill_split_start_values(
    records: dict[str, CalibrationRecord],
    pages: list[tuple[int, str]],
) -> None:
    document_text = "\n".join(text for _, text in pages)
    for label, record in records.items():
        if record.start_value is not None:
            continue
        position = document_text.find(label)
        if position < 0:
            continue
        window = document_text[position:position + 5000]
        marker = START_MARKER_RE.search(window)
        value_text = window[marker.start():] if marker else window
        strict_value = re.search(
            rf"(?:start|standard|default|initial)\s*value\b\s*:?\s*(?:\(all\)|all)?\s*(?P<value>{NUM})\s*\[(?P<unit>[^\]]*)\]",
            value_text,
            flags=re.IGNORECASE,
        )
        if strict_value:
            value = float(strict_value.group("value"))
            unit = _normalize_unit(strict_value.group("unit"))
            broadcast = bool(BROADCAST_TOKEN_RE.search(strict_value.group(0)))
        else:
            value, unit, broadcast = _extract_start_value_and_unit(value_text)
        if value is not None:
            record.start_value = value
            record.unit = unit
            record.is_broadcast = broadcast


def _serialize_start_value(value: float | list[float] | None) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _as_sheet_rows(cal_rows: list[CalibrationRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in cal_rows:
        rows.append(
            {
                "module": r.module,
                "label_name": r.label_name,
                "object_class": r.object_class,
                "physical_prefix": r.physical_prefix,
                "context": r.context,
                "start_value": _serialize_start_value(r.start_value),
                "unit": r.unit or "",
                "is_broadcast": str(r.is_broadcast).lower(),
                "limit_lower": r.limits.get("lower"),
                "limit_upper": r.limits.get("upper"),
                "limit_type": r.limits.get("type"),
                "limit_rationale": r.limits.get("rationale"),
                "axes": "" if r.axes is None else json.dumps(r.axes, ensure_ascii=True),
                "calibration_table_id": r.source_reference.get("table_id"),
                "calibration_page": r.source_reference.get("page"),
                "parent_module": r.source_reference.get("parent_module"),
                "value_table_id": r.value_table_id,
            }
        )
    return rows


def _index_value_tables(value_tables: list[ValueTableRecord]) -> tuple[dict[str, ValueTableRecord], dict[str, ValueTableRecord]]:
    def quality(vt: ValueTableRecord) -> int:
        score = 0
        score += len(vt.values.get("X", []))
        score += len(vt.values.get("Y", []))
        score += len(vt.values.get("VAL", []))
        score += 10 if vt.axes else 0
        return score

    by_id: dict[str, ValueTableRecord] = {}
    by_label: dict[str, ValueTableRecord] = {}
    for vt in value_tables:
        if vt.table_id:
            old = by_id.get(vt.table_id)
            if old is None or quality(vt) > quality(old):
                by_id[vt.table_id] = vt
        if vt.label_name:
            old = by_label.get(vt.label_name)
            if old is None or quality(vt) > quality(old):
                by_label[vt.label_name] = vt
    return by_id, by_label


def _pick_value_table(row: CalibrationRecord, by_id: dict[str, ValueTableRecord], by_label: dict[str, ValueTableRecord]) -> ValueTableRecord | None:
    if row.value_table_id:
        return by_id.get(row.value_table_id)

    if re.search(rf"\[\s*{re.escape(row.label_name)}\s*\]", row.context):
        return by_label.get(row.label_name)

    if re.search(r"\bsee\s+table\b", row.context, re.IGNORECASE):
        return by_label.get(row.label_name)

    return None


def _line_starts_with_label(line: str) -> str | None:
    match = re.match(r"^\s*(?P<label>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)?_[A-Za-z0-9_]+)\b", line)
    if not match:
        return None
    candidate = match.group("label").strip()
    if LABEL_SPLITTER_RE.match(candidate):
        return candidate
    return None


def _is_label_header_line(line: str) -> bool:
    normalized = _normalize_ws(line).lower()
    if normalized in {"label name", "description", "label name description"}:
        return True
    if TABLE_HEADER_RE.search(normalized):
        return True
    return False


def _extract_label_scoped_table_text(table: ValueTableRecord, label_name: str) -> str:
    if not table.raw_text:
        return ""

    lines = [line.rstrip() for line in table.raw_text.splitlines()]
    if not lines:
        return ""

    table_title = lines[0] if lines and TABLE_ID_RE.search(lines[0]) else ""
    segments: list[list[str]] = []
    i = 0

    while i < len(lines):
        current = lines[i]
        starts_label = _line_starts_with_label(current)
        bracket_ref = bool(re.search(rf"\[\s*{re.escape(label_name)}\s*\]", current))
        if starts_label != label_name and not bracket_ref:
            i += 1
            continue

        segment: list[str] = []
        if table_title and current != table_title:
            segment.append(table_title)
        segment.append(current)

        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if _is_label_header_line(nxt):
                j += 1
                continue

            nxt_starts_label = _line_starts_with_label(nxt)
            if nxt_starts_label and nxt_starts_label != label_name:
                break

            segment.append(nxt)
            j += 1

        segments.append(segment)
        i = j

    if not segments:
        return ""

    unique_parts: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        part = "\n".join(line for line in segment if line.strip()).strip()
        if not part or part in seen:
            continue
        seen.add(part)
        unique_parts.append(part)

    return "\n\n".join(unique_parts).strip()


def _is_descriptive_apph_table(table: ValueTableRecord) -> bool:
    raw = (table.raw_text or "").lower()
    if not raw:
        return False
    if "[apph_labtab]" in raw:
        return True
    if "calibration parameters for" in raw:
        return True
    return False


def _append_value_table_text_to_context(records: list[CalibrationRecord], value_tables: list[ValueTableRecord]) -> None:
    by_id, by_label = _index_value_tables(value_tables)
    by_label_all: dict[str, list[ValueTableRecord]] = {}
    for table in value_tables:
        if table.label_name:
            by_label_all.setdefault(table.label_name, []).append(table)
    for label_tables in by_label_all.values():
        label_tables.sort(key=lambda t: (t.source_page, t.table_id))

    for row in records:
        label_tables = by_label_all.get(row.label_name, [])
        has_descriptive_table = any(_is_descriptive_apph_table(table) for table in label_tables)
        linked_tables: list[ValueTableRecord] = []
        linked = _pick_value_table(row, by_id, by_label)
        if linked is not None:
            if (
                not has_descriptive_table
                or _is_descriptive_apph_table(linked)
            ):
                linked_tables.append(linked)

        if (
            re.search(rf"\[\s*{re.escape(row.label_name)}\s*\]", row.context)
            or re.search(r"\bsee\s+table\b", row.context, re.IGNORECASE)
            or row.value_table_id is not None
        ):
            for table in label_tables:
                if has_descriptive_table and not _is_descriptive_apph_table(table):
                    continue
                has_structured_content = bool(table.axes) or any(
                    bool(table.values.get(key, [])) for key in ("X", "Y", "VAL")
                )
                if (
                    not has_structured_content
                    and table.table_id != row.value_table_id
                    and not _is_descriptive_apph_table(table)
                ):
                    continue
                if all(table.table_id != existing.table_id for existing in linked_tables):
                    linked_tables.append(table)

        for table in linked_tables:
            if not table.raw_text:
                continue
            table_text = _extract_label_scoped_table_text(table, row.label_name)
            if not table_text or table_text in row.context:
                continue
            if row.context.strip():
                row.context = f"{row.context.rstrip()}\n\n{table_text}"
            else:
                row.context = table_text


def _record_axis_meta(row: CalibrationRecord, axis_name: str) -> dict[str, Any]:
    if not row.axes:
        return {}
    for axis in row.axes:
        if str(axis.get("axis", "")).upper() == axis_name.upper():
            return axis
    return {}


def _record_axis_breakpoints(row: CalibrationRecord, axis_name: str) -> list[float]:
    axis = _record_axis_meta(row, axis_name)
    values = axis.get("breakpoints") if axis else None
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for v in values:
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _build_curve_sheet_rows(curve_rows: list[CalibrationRecord], value_tables: list[ValueTableRecord]) -> list[dict[str, Any]]:
    by_id, by_label = _index_value_tables(value_tables)
    out: list[dict[str, Any]] = []

    for row in curve_rows:
        vt = _pick_value_table(row, by_id, by_label)
        if vt is None:
            x_axis = _record_axis_meta(row, "X")
            x_vals = _record_axis_breakpoints(row, "X")
            if not x_vals and x_axis.get("size"):
                x_vals = [None] * int(x_axis["size"])
            if isinstance(row.start_value, list):
                y_vals = row.start_value
            elif isinstance(row.start_value, (int, float)) and x_vals:
                y_vals = [float(row.start_value)] * len(x_vals)
            else:
                y_vals = []
            max_len = max(len(x_vals), len(y_vals))
            if max_len > 0:
                for idx in range(max_len):
                    out.append(
                        {
                            "label_name": row.label_name,
                            "module": row.module,
                            "table_id": row.value_table_id,
                            "x_index": idx,
                            "x_value": x_vals[idx] if idx < len(x_vals) else None,
                            "curve_value": y_vals[idx] if idx < len(y_vals) else None,
                            "x_signal": x_axis.get("signal"),
                            "x_unit": x_axis.get("unit"),
                            "context": row.context,
                        }
                    )
                continue
            out.append(
                {
                    "label_name": row.label_name,
                    "module": row.module,
                    "table_id": row.value_table_id,
                    "x_index": "",
                    "x_value": "",
                    "curve_value": "",
                    "x_signal": "",
                    "x_unit": "",
                    "context": row.context,
                }
            )
            continue

        x_vals = vt.values.get("X", [])
        y_vals = vt.values.get("VAL", [])
        axis_meta = vt.axes.get("X", {})
        if (not x_vals and not y_vals) and row.axes:
            x_vals = _record_axis_breakpoints(row, "X")
            axis_row_meta = _record_axis_meta(row, "X")
            if not x_vals and axis_row_meta.get("size"):
                x_vals = [None] * int(axis_row_meta["size"])
            if isinstance(row.start_value, list):
                y_vals = row.start_value
            elif isinstance(row.start_value, (int, float)) and x_vals:
                y_vals = [float(row.start_value)] * len(x_vals)
            else:
                y_vals = []
            axis_meta = _record_axis_meta(row, "X")
        max_len = max(len(x_vals), len(y_vals))
        if max_len == 0:
            out.append(
                {
                    "label_name": row.label_name,
                    "module": row.module,
                    "table_id": vt.table_id,
                    "x_index": "",
                    "x_value": "",
                    "curve_value": "",
                    "x_signal": axis_meta.get("signal"),
                    "x_unit": axis_meta.get("unit"),
                    "context": row.context,
                }
            )
            continue
        for idx in range(max_len):
            out.append(
                {
                    "label_name": row.label_name,
                    "module": row.module,
                    "table_id": vt.table_id,
                    "x_index": idx,
                    "x_value": x_vals[idx] if idx < len(x_vals) else None,
                    "curve_value": y_vals[idx] if idx < len(y_vals) else None,
                    "x_signal": axis_meta.get("signal"),
                    "x_unit": axis_meta.get("unit"),
                    "context": row.context,
                }
            )

    return out


def _build_map_sheet_rows(map_rows: list[CalibrationRecord], value_tables: list[ValueTableRecord]) -> list[dict[str, Any]]:
    by_id, by_label = _index_value_tables(value_tables)
    out: list[dict[str, Any]] = []

    for row in map_rows:
        vt = _pick_value_table(row, by_id, by_label)
        if vt is None:
            x_vals = _record_axis_breakpoints(row, "X")
            y_vals = _record_axis_breakpoints(row, "Y")
            x_meta = _record_axis_meta(row, "X")
            y_meta = _record_axis_meta(row, "Y")
            if not x_vals and x_meta.get("size"):
                x_vals = [None] * int(x_meta["size"])
            if not y_vals and y_meta.get("size"):
                y_vals = [None] * int(y_meta["size"])
            if isinstance(row.start_value, list):
                z_vals = row.start_value
            elif isinstance(row.start_value, (int, float)) and x_vals and y_vals:
                z_vals = [float(row.start_value)] * (len(x_vals) * len(y_vals))
            else:
                z_vals = []
            if x_vals and y_vals:
                k = 0
                for yi, yv in enumerate(y_vals):
                    for xi, xv in enumerate(x_vals):
                        out.append(
                            {
                                "label_name": row.label_name,
                                "module": row.module,
                                "table_id": row.value_table_id,
                                "x_index": xi,
                                "x_value": xv,
                                "y_index": yi,
                                "y_value": yv,
                                "z_value": z_vals[k] if k < len(z_vals) else None,
                                "x_signal": x_meta.get("signal"),
                                "x_unit": x_meta.get("unit"),
                                "y_signal": y_meta.get("signal"),
                                "y_unit": y_meta.get("unit"),
                                "context": row.context,
                            }
                        )
                        k += 1
                continue

            out.append(
                {
                    "label_name": row.label_name,
                    "module": row.module,
                    "table_id": row.value_table_id,
                    "x_index": "",
                    "x_value": "",
                    "y_index": "",
                    "y_value": "",
                    "z_value": "",
                    "x_signal": x_meta.get("signal"),
                    "x_unit": x_meta.get("unit"),
                    "y_signal": y_meta.get("signal"),
                    "y_unit": y_meta.get("unit"),
                    "context": row.context,
                }
            )
            continue

        x_vals = vt.values.get("X", [])
        y_vals = vt.values.get("Y", [])
        z_vals = vt.values.get("VAL", [])
        x_meta = vt.axes.get("X", {})
        y_meta = vt.axes.get("Y", {})

        if (not x_vals and not y_vals and not z_vals) and row.axes:
            x_vals = _record_axis_breakpoints(row, "X")
            y_vals = _record_axis_breakpoints(row, "Y")
            x_meta = _record_axis_meta(row, "X")
            y_meta = _record_axis_meta(row, "Y")
            if not x_vals and x_meta.get("size"):
                x_vals = [None] * int(x_meta["size"])
            if not y_vals and y_meta.get("size"):
                y_vals = [None] * int(y_meta["size"])
            if isinstance(row.start_value, list):
                z_vals = row.start_value
            elif isinstance(row.start_value, (int, float)) and x_vals and y_vals:
                z_vals = [float(row.start_value)] * (len(x_vals) * len(y_vals))
            else:
                z_vals = []
        if not z_vals:
            out.append(
                {
                    "label_name": row.label_name,
                    "module": row.module,
                    "table_id": vt.table_id,
                    "x_index": "",
                    "x_value": "",
                    "y_index": "",
                    "y_value": "",
                    "z_value": "",
                    "x_signal": x_meta.get("signal"),
                    "x_unit": x_meta.get("unit"),
                    "y_signal": y_meta.get("signal"),
                    "y_unit": y_meta.get("unit"),
                    "context": row.context,
                }
            )
            continue

        if x_vals and y_vals and z_vals and len(z_vals) >= len(x_vals) * len(y_vals):
            k = 0
            for yi, yv in enumerate(y_vals):
                for xi, xv in enumerate(x_vals):
                    out.append(
                        {
                            "label_name": row.label_name,
                            "module": row.module,
                            "table_id": vt.table_id,
                            "x_index": xi,
                            "x_value": xv,
                            "y_index": yi,
                            "y_value": yv,
                            "z_value": z_vals[k],
                            "x_signal": x_meta.get("signal"),
                            "x_unit": x_meta.get("unit"),
                            "y_signal": y_meta.get("signal"),
                            "y_unit": y_meta.get("unit"),
                            "context": row.context,
                        }
                    )
                    k += 1
        else:
            for i, zv in enumerate(z_vals):
                out.append(
                    {
                        "label_name": row.label_name,
                        "module": row.module,
                        "table_id": vt.table_id,
                        "x_index": i if i < len(x_vals) else None,
                        "x_value": x_vals[i] if i < len(x_vals) else None,
                        "y_index": None,
                        "y_value": None,
                        "z_value": zv,
                        "x_signal": x_meta.get("signal"),
                        "x_unit": x_meta.get("unit"),
                        "y_signal": y_meta.get("signal"),
                        "y_unit": y_meta.get("unit"),
                        "context": row.context,
                    }
                )

    return out


def _build_array_sheet_rows(array_rows: list[CalibrationRecord], value_tables: list[ValueTableRecord]) -> list[dict[str, Any]]:
    by_id, by_label = _index_value_tables(value_tables)
    out: list[dict[str, Any]] = []

    for row in array_rows:
        vt = _pick_value_table(row, by_id, by_label)
        values: list[float] = []
        table_id = row.value_table_id
        axis_signal = ""
        axis_unit = ""

        if vt is not None:
            table_id = vt.table_id
            values = vt.values.get("VAL", []) or vt.values.get("X", [])
            x_meta = vt.axes.get("X", {})
            axis_signal = str(x_meta.get("signal") or "")
            axis_unit = str(x_meta.get("unit") or "")
        elif isinstance(row.start_value, list):
            values = row.start_value

        if not values:
            out.append(
                {
                    "label_name": row.label_name,
                    "module": row.module,
                    "table_id": table_id,
                    "index": "",
                    "value": "",
                    "axis_signal": axis_signal,
                    "axis_unit": axis_unit,
                    "context": row.context,
                }
            )
            continue

        for idx, val in enumerate(values):
            out.append(
                {
                    "label_name": row.label_name,
                    "module": row.module,
                    "table_id": table_id,
                    "index": idx,
                    "value": val,
                    "axis_signal": axis_signal,
                    "axis_unit": axis_unit,
                    "context": row.context,
                }
            )

    return out


def write_workbook(records: list[CalibrationRecord], value_tables: list[ValueTableRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scalar_rows = [r for r in records if r.object_class == "VALUE"]
    curve_rows = [r for r in records if r.object_class == "CURVE"]
    map_rows = [r for r in records if r.object_class == "MAP"]
    array_rows = [r for r in records if r.object_class == "VAL_BLK"]

    df_calibration = pd.DataFrame(_as_sheet_rows(records))
    df_scalars = pd.DataFrame(_as_sheet_rows(scalar_rows))
    df_curves = pd.DataFrame(_build_curve_sheet_rows(curve_rows, value_tables))
    df_maps = pd.DataFrame(_build_map_sheet_rows(map_rows, value_tables))
    df_arrays = pd.DataFrame(_build_array_sheet_rows(array_rows, value_tables))

    if df_calibration.empty:
        df_calibration = pd.DataFrame(columns=[
            "module",
            "label_name",
            "object_class",
            "physical_prefix",
            "context",
            "start_value",
            "unit",
            "is_broadcast",
            "limit_lower",
            "limit_upper",
            "limit_type",
            "limit_rationale",
            "axes",
            "calibration_table_id",
            "calibration_page",
            "parent_module",
            "value_table_id",
        ])

    if df_scalars.empty:
        df_scalars = pd.DataFrame(columns=[
            "module",
            "label_name",
            "object_class",
            "physical_prefix",
            "context",
            "start_value",
            "unit",
            "is_broadcast",
            "limit_lower",
            "limit_upper",
            "limit_type",
            "limit_rationale",
            "axes",
            "calibration_table_id",
            "calibration_page",
            "parent_module",
            "value_table_id",
        ])

    if df_curves.empty:
        df_curves = pd.DataFrame(columns=[
            "label_name",
            "module",
            "table_id",
            "x_index",
            "x_value",
            "curve_value",
            "x_signal",
            "x_unit",
            "context",
        ])

    if df_maps.empty:
        df_maps = pd.DataFrame(columns=[
            "label_name",
            "module",
            "table_id",
            "x_index",
            "x_value",
            "y_index",
            "y_value",
            "z_value",
            "x_signal",
            "x_unit",
            "y_signal",
            "y_unit",
            "context",
        ])

    if df_arrays.empty:
        df_arrays = pd.DataFrame(columns=[
            "label_name",
            "module",
            "table_id",
            "index",
            "value",
            "axis_signal",
            "axis_unit",
            "context",
        ])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_calibration.to_excel(writer, sheet_name="Calibration", index=False)
        df_scalars.to_excel(writer, sheet_name="Scalars", index=False)
        df_curves.to_excel(writer, sheet_name="Curves", index=False)
        df_maps.to_excel(writer, sheet_name="Maps", index=False)
        df_arrays.to_excel(writer, sheet_name="Arrays", index=False)


RANGE_PHYS_RE = re.compile(
    rf"(?P<lower>{NUM})\s*\.\.\.\s*(?P<upper>{NUM})(?:\s*\[(?P<unit>[^\]]*)\])?",
    flags=re.IGNORECASE,
)


def _logical_pdf_lines(lines: list[str]) -> list[str]:
    logical: list[str] = []
    for raw in lines:
        line = _normalize_ws(raw)
        if not line:
            continue
        if logical and logical[-1].endswith(("-", "−")):
            logical[-1] = logical[-1][:-1] + line
        else:
            logical.append(line)
    return logical


def _canonical_label(value: str) -> str | None:
    candidate = value.strip().strip(".,;:()[]{}<>\"'|")
    return candidate if LABEL_SPLITTER_RE.match(candidate) else None


def _extract_physical_range_map(pages: list[tuple[int, str]]) -> dict[str, dict[str, Any]]:
    range_map: dict[str, dict[str, Any]] = {}

    lines: list[str] = []
    for _, page_text in pages:
        lines.extend(_logical_pdf_lines(page_text.splitlines()))

    details_starts = [
        i for i, line in enumerate(lines)
        if re.search(r"Parameters\s*:\s*details", line, flags=re.IGNORECASE)
    ]
    for details_start in details_starts:
        end = len(lines)
        for i in range(details_start + 1, len(lines)):
            if (
                TABLE_ID_RE.search(lines[i])
                or re.search(r"Variables\s*:\s*details", lines[i], re.IGNORECASE)
                or re.search(r"Parameters\s*:\s*overview", lines[i], re.IGNORECASE)
            ):
                end = i
                break

        for i in range(details_start + 1, end):
            label = _canonical_label(lines[i])
            if not label:
                continue

            row_tail: list[str] = []
            for candidate in lines[i + 1:end]:
                if _canonical_label(candidate):
                    break
                row_tail.append(candidate)

            range_text = " ".join(row_tail).replace("−", "-").replace("–", "-")
            range_matches = list(RANGE_PHYS_RE.finditer(range_text))
            if not range_matches:
                continue

            physical = range_matches[1] if len(range_matches) > 1 else range_matches[0]
            raw = physical.group(0).strip()
            range_map[label] = {
                "lower": float(physical.group("lower")),
                "upper": float(physical.group("upper")),
                "raw": raw,
            }

    return range_map


def _clean_explicit_context(label: str, row_lines: list[str]) -> str:
    merged = _normalize_ws(" ".join(row_lines))
    label_pos = merged.find(label)
    return _normalize_ws(merged[label_pos + len(label):].strip(" |-:")) if label_pos >= 0 else ""


def _extract_simple_calibration_rows(
    pages: list[tuple[int, str]],
    max_row_lines: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    active_main_function = ""

    for page_no, page_text in pages:
        normalized_lines = [_normalize_ws(line) for line in page_text.splitlines()]
        for line in normalized_lines:
            heading_function = _extract_parent_module(line)
            if heading_function:
                active_main_function = heading_function

        sections = _parse_table_sections(page_text)
        if not sections:
            heading_positions = [i for i, line in enumerate(normalized_lines) if _is_section_heading(line)]
            continuation_end = heading_positions[0] if heading_positions else len(normalized_lines)
            has_labels = any(_find_label_in_line(line) for line in normalized_lines[:continuation_end])
            if has_labels and continuation_end > 0:
                sections = [(0, continuation_end)]
            else:
                continue
        for start, end in sections:
            main_function = active_main_function
            for heading_index in range(start, -1, -1):
                heading_function = _extract_parent_module(normalized_lines[heading_index])
                if heading_function:
                    main_function = heading_function
                    break

            cursor = start
            while cursor < end:
                label = _find_label_in_line(normalized_lines[cursor])
                if not label:
                    cursor += 1
                    continue

                row_lines = [normalized_lines[cursor]]
                lookahead = cursor + 1
                while lookahead < end and len(row_lines) < max_row_lines:
                    next_line = normalized_lines[lookahead]
                    if (
                        _find_label_in_line(next_line)
                        or re.match(r"^[A-Za-z0-9_]+\.[A-Za-z0-9_]+", next_line)
                        or _is_section_heading(next_line)
                        or re.search(r"^Caution\b|^Table\s+\d+\b|^\d+\s+System\s+Constants", next_line, re.IGNORECASE)
                        or next_line.startswith("2025-")
                    ):
                        break
                    if not next_line:
                        lookahead += 1
                        continue
                    row_lines.append(next_line)
                    lookahead += 1

                row_text = _normalize_ws(" ".join(row_lines))
                start_value, unit, _ = _extract_start_value_and_unit(row_text)
                rows.append(
                    {
                        "label_name": label,
                        "context": _clean_explicit_context(label, row_lines),
                        "start_value": _serialize_start_value(start_value),
                        "unit": unit or "",
                        "page": page_no,
                        "main_function": main_function or "",
                    }
                )
                cursor = lookahead

    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        current = best.get(row["label_name"])
        if current is None or len(row["context"]) > len(current["context"]):
            best[row["label_name"]] = row
    return list(best.values())


def write_csv(records: list[dict[str, Any]], range_map: dict[str, dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "main_function",
        "label_name",
        "context",
        "start_value",
        "unit",
        "lower_limit",
        "upper_limit",
        "value_range_phys_raw",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            physical = _lookup_physical_range(row["label_name"], range_map)
            writer.writerow(
                {
                    "main_function": row.get("main_function", ""),
                    "label_name": row["label_name"],
                    "context": row["context"],
                    "start_value": row["start_value"],
                    "unit": row["unit"],
                    "lower_limit": physical.get("lower", ""),
                    "upper_limit": physical.get("upper", ""),
                    "value_range_phys_raw": physical.get("raw", ""),
                }
            )


def _lookup_physical_range(label: str, range_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return range_map.get(label, {})


def _precal_metadata_rows(
    records: list[CalibrationRecord],
    range_map: dict[str, dict[str, Any]],
    main_functions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        physical = _lookup_physical_range(record.label_name, range_map)
        rows.append(
            {
                "main_function": (main_functions or {}).get(record.label_name, ""),
                "label_name": record.label_name,
                "context": _precal_context(record.context),
                "start_value": _serialize_start_value(record.start_value),
                "unit": record.unit or "",
                "lower_limit": physical.get("lower", ""),
                "upper_limit": physical.get("upper", ""),
                "value_range_phys_raw": physical.get("raw", ""),
            }
        )
    return rows


def _precal_context(value: str) -> str:
    return value or ""


def _precal_curve_rows(
    records: list[CalibrationRecord],
    value_tables: list[ValueTableRecord],
    main_functions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows = _build_curve_sheet_rows(
        [record for record in records if record.object_class == "CURVE"],
        value_tables,
    )
    for row in rows:
        row["main_function"] = (main_functions or {}).get(row.get("label_name", ""), "")
        row["context"] = _precal_context(str(row.get("context", "")))
    return rows


def _precal_map_rows(
    records: list[CalibrationRecord],
    value_tables: list[ValueTableRecord],
    main_functions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows = _build_map_sheet_rows(
        [record for record in records if record.object_class == "MAP"],
        value_tables,
    )
    for row in rows:
        row["main_function"] = (main_functions or {}).get(row.get("label_name", ""), "")
        row["context"] = _precal_context(str(row.get("context", "")))
    return rows


def _precal_array_rows(
    records: list[CalibrationRecord],
    value_tables: list[ValueTableRecord],
    main_functions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows = _build_array_sheet_rows(
        [record for record in records if record.object_class == "VAL_BLK"],
        value_tables,
    )
    for row in rows:
        row["main_function"] = (main_functions or {}).get(row.get("label_name", ""), "")
        row["context"] = _precal_context(str(row.get("context", "")))
    return rows


def _precal_audit_rows(
    records: list[CalibrationRecord],
    range_map: dict[str, dict[str, Any]],
    value_tables: list[ValueTableRecord],
    main_functions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    by_id, by_label = _index_value_tables(value_tables)
    curve_rows = _build_curve_sheet_rows(
        [record for record in records if record.object_class == "CURVE"],
        value_tables,
    )
    map_rows = _build_map_sheet_rows(
        [record for record in records if record.object_class == "MAP"],
        value_tables,
    )

    audit: list[dict[str, Any]] = []
    for record in records:
        linked = _pick_value_table(record, by_id, by_label)
        curve_values = [
            row for row in curve_rows if row.get("label_name") == record.label_name
        ]
        map_values = [
            row for row in map_rows if row.get("label_name") == record.label_name
        ]
        audit.append(
            {
                "main_function": (main_functions or {}).get(record.label_name, ""),
                "label_name": record.label_name,
                "object_class": record.object_class,
                "context_present": bool(_precal_context(record.context)),
                "start_value_present": record.start_value is not None,
                "physical_range_present": bool(_lookup_physical_range(record.label_name, range_map)),
                "value_table_linked": linked is not None,
                "numeric_curve_values": sum(row.get("curve_value") is not None for row in curve_values),
                "numeric_map_values": sum(row.get("z_value") is not None for row in map_values),
                "source_page": record.source_reference.get("page"),
            }
        )
    return audit


def write_precal_workbook(
    records: list[CalibrationRecord],
    value_tables: list[ValueTableRecord],
    range_map: dict[str, dict[str, Any]],
    output_path: Path,
    main_functions: dict[str, str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_rows = _precal_metadata_rows(records, range_map, main_functions)
    scalar_rows = [
        row for row in metadata_rows
        if next((record for record in records if record.label_name == row["label_name"]), None)
        and next(record for record in records if record.label_name == row["label_name"]).object_class == "VALUE"
    ]

    calibration_columns = [
        "main_function",
        "label_name",
        "context",
        "start_value",
        "unit",
        "lower_limit",
        "upper_limit",
        "value_range_phys_raw",
    ]
    curve_columns = [
        "main_function",
        "label_name",
        "module",
        "table_id",
        "x_index",
        "x_value",
        "curve_value",
        "x_signal",
        "x_unit",
        "context",
    ]
    map_columns = [
        "main_function",
        "label_name",
        "module",
        "table_id",
        "x_index",
        "x_value",
        "y_index",
        "y_value",
        "z_value",
        "x_signal",
        "x_unit",
        "y_signal",
        "y_unit",
        "context",
    ]
    array_columns = [
        "main_function",
        "label_name",
        "module",
        "table_id",
        "index",
        "value",
        "axis_signal",
        "axis_unit",
        "context",
    ]
    audit_columns = [
        "main_function",
        "label_name",
        "object_class",
        "context_present",
        "start_value_present",
        "physical_range_present",
        "value_table_linked",
        "numeric_curve_values",
        "numeric_map_values",
        "source_page",
    ]

    frames = {
        "Calibration": pd.DataFrame(metadata_rows, columns=calibration_columns),
        "Scalars": pd.DataFrame(scalar_rows, columns=calibration_columns),
        "Curves": pd.DataFrame(_precal_curve_rows(records, value_tables, main_functions), columns=curve_columns),
        "Maps": pd.DataFrame(_precal_map_rows(records, value_tables, main_functions), columns=map_columns),
        "Arrays": pd.DataFrame(_precal_array_rows(records, value_tables, main_functions), columns=array_columns),
        "Audit": pd.DataFrame(
            _precal_audit_rows(records, range_map, value_tables, main_functions),
            columns=audit_columns,
        ),
    }

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse calibration documentation into a pre-calibration workbook."
    )
    parser.add_argument("--input-doc", required=True, help="Path to input calibration document (.pdf or text file).")
    parser.add_argument("--output-xlsx", required=True, help="Path to output Excel workbook.")
    parser.add_argument("--max-row-lines", type=int, default=20, help="Max continuation lines per calibration row.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_doc)
    output_path = Path(args.output_xlsx)

    if not input_path.exists():
        raise FileNotFoundError(f"Input document not found: {input_path}")

    pages = _extract_pages(input_path)
    range_map = _extract_physical_range_map(pages)
    records, value_tables = parse_document(
        input_path,
        max_row_lines=max(2, args.max_row_lines),
    )
    _append_value_table_text_to_context(records, value_tables)
    main_functions = {
        record.label_name: str(record.source_reference.get("parent_module") or "")
        for record in records
    }
    write_precal_workbook(records, value_tables, range_map, output_path, main_functions)

    print(f"[done] Parsed labels: {len(records)}")
    print(f"[done] Authoritative physical ranges: {len(range_map)}")
    print(f"[done] Parsed value tables: {len(value_tables)}")
    print(f"[done] Workbook path: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
