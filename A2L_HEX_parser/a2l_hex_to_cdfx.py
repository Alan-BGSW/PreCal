"""
a2l_hex_to_cdfx.py
==================
Convert an A2L + HEX pair into a minimal CDFX file that the existing
CDFX-vs-CDFX pipeline (pipeline.override_incorrect_values) can consume
without any JSON intermediate.

The generated CDFX only contains the elements the pipeline reads:

    MSRSW > SW-SYSTEMS > SW-SYSTEM > SW-INSTANCE-SPEC > SW-INSTANCE-TREE
        SW-INSTANCE
            SHORT-NAME
            CATEGORY                        (VALUE / VAL_BLK / CURVE / MAP)
            SW-VALUE-CONT
                UNIT-DISPLAY-NAME
                SW-ARRAYSIZE/V              (VAL_BLK only)
                SW-VALUES-PHYS
                    V | VT                  (scalar / 1D)
                    VG > LABEL + V*         (MAP rows)
            SW-AXIS-CONTS
                SW-AXIS-CONT*               (one for CURVE, two for MAP)
                    CATEGORY
                    SW-VALUES-PHYS > V*

This is enough for `extract_labels_with_c` and the `_update_*_cdfx`
functions in pipeline.py to treat the generated file exactly like a real
CDFX written by INCA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union

import lxml.etree as ET

from parser import CalibrationExtractor


def _fmt(v: Any) -> str:
    """Stable text representation; pipeline normalises numerically anyway."""
    if v is None:
        return ""
    if isinstance(v, float):
        # Trim trailing zeros for readability while preserving precision.
        s = repr(v)
        return s
    return str(v)


def _is_numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _add_text_child(parent: ET._Element, tag: str, text: str) -> ET._Element:
    el = ET.SubElement(parent, tag)
    el.text = text
    return el


def _add_values_phys_flat(parent: ET._Element, values: Iterable[Any]) -> None:
    svp = ET.SubElement(parent, "SW-VALUES-PHYS")
    for v in values:
        tag = "V" if _is_numeric(v) else "VT"
        _add_text_child(svp, tag, _fmt(v))


def _add_axis_cont(parent_sw_instance: ET._Element,
                   axis_values: Sequence[Any],
                   unit: str = "") -> None:
    """Create (or extend) <SW-AXIS-CONTS> with one <SW-AXIS-CONT>."""
    conts = parent_sw_instance.find("SW-AXIS-CONTS")
    if conts is None:
        conts = ET.SubElement(parent_sw_instance, "SW-AXIS-CONTS")
    ac = ET.SubElement(conts, "SW-AXIS-CONT")
    _add_text_child(ac, "CATEGORY", "STD_AXIS")
    if unit:
        u = ET.SubElement(ac, "UNIT-DISPLAY-NAME")
        u.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        u.text = unit
    _add_values_phys_flat(ac, axis_values)


def _reshape_map_rows(value: Any,
                      axes: Sequence[Sequence[Any]]) -> List[List[Any]]:
    """
    Normalise parser.extract_one MAP `value` into row-major rows where each
    row corresponds to one y-axis label.

    parser.py produces ROW_DIR as (ny, nx) and COLUMN_DIR as (nx, ny).
    """
    if not isinstance(value, list) or not value:
        return []
    if not all(isinstance(r, list) for r in value):
        return []
    nx = len(axes[0]) if len(axes) >= 1 else 0
    ny = len(axes[1]) if len(axes) >= 2 else 0
    if ny and len(value) == ny:
        return [list(r) for r in value]
    if nx and len(value) == nx and value and ny and len(value[0]) == ny:
        # transpose (nx, ny) -> (ny, nx)
        return [[value[c][r] for c in range(nx)] for r in range(ny)]
    return [list(r) for r in value]


def _build_sw_instance(parent: ET._Element, entry: dict) -> None:
    label = entry.get("label")
    if not label or "error" in entry:
        return
    ctype = entry.get("type", "VALUE")
    value = entry.get("value")
    unit = entry.get("unit") or ""
    axes = entry.get("axes") or []

    sw = ET.SubElement(parent, "SW-INSTANCE")
    _add_text_child(sw, "SHORT-NAME", label)
    _add_text_child(sw, "CATEGORY", ctype)

    vc = ET.SubElement(sw, "SW-VALUE-CONT")

    if ctype == "VAL_BLK":
        flat = value if isinstance(value, list) else [value]
        sa = ET.SubElement(vc, "SW-ARRAYSIZE")
        _add_text_child(sa, "V", str(len(flat)))

    if unit:
        u = ET.SubElement(vc, "UNIT-DISPLAY-NAME")
        u.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        u.text = unit

    if ctype == "MAP":
        rows = _reshape_map_rows(value, axes)
        y_axis = list(axes[1]) if len(axes) > 1 else []
        x_axis = list(axes[0]) if len(axes) >= 1 else []
        svp = ET.SubElement(vc, "SW-VALUES-PHYS")
        for ri, row in enumerate(rows):
            vg = ET.SubElement(svp, "VG")
            y_label = y_axis[ri] if ri < len(y_axis) else ri
            _add_text_child(vg, "LABEL", _fmt(y_label))
            for cell in row:
                tag = "V" if _is_numeric(cell) else "VT"
                _add_text_child(vg, tag, _fmt(cell))
        # axes: x first, then y (matches CDFX SW-AXIS-CONT order)
        if x_axis:
            _add_axis_cont(sw, x_axis)
        if y_axis:
            _add_axis_cont(sw, y_axis)

    elif ctype == "CURVE":
        z = value if isinstance(value, list) else [value]
        x_axis = list(axes[0]) if axes else []
        _add_values_phys_flat(vc, z)
        if x_axis:
            _add_axis_cont(sw, x_axis)

    elif ctype == "VAL_BLK":
        flat = value if isinstance(value, list) else [value]
        _add_values_phys_flat(vc, flat)

    else:  # VALUE (scalar) — default fall-through for any other char_type
        if isinstance(value, list):
            # parser sometimes returns 1-element list for VALUE with COMPU_METHOD
            scalar = value[0] if value else 0
        else:
            scalar = value
        _add_values_phys_flat(vc, [scalar])


def write_cdfx_from_a2l_hex(a2l_path: Union[str, Path],
                            hex_path: Union[str, Path],
                            out_cdfx: Union[str, Path]) -> dict:
    """
    Run the A2L+HEX extractor and emit a minimal CDFX at `out_cdfx`.

    Returns a small status dict: {labels, written, warnings, out_cdfx}.
    """
    out_path = Path(out_cdfx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = CalibrationExtractor(str(hex_path), str(a2l_path))
    entries = extractor.extract_all()

    root = ET.Element("MSRSW")
    _add_text_child(root, "SHORT-NAME", Path(a2l_path).stem or "GENERATED")
    _add_text_child(root, "CATEGORY", "CDF21")

    systems = ET.SubElement(root, "SW-SYSTEMS")
    system = ET.SubElement(systems, "SW-SYSTEM")
    _add_text_child(system, "SHORT-NAME", "generated-system")
    spec = ET.SubElement(system, "SW-INSTANCE-SPEC")
    tree_el = ET.SubElement(spec, "SW-INSTANCE-TREE")
    _add_text_child(tree_el, "SHORT-NAME", "GENERATED-TREE")
    _add_text_child(tree_el, "CATEGORY", "NO_VCD")

    written = 0
    for label in entries.keys():
        entry = entries[label]
        if not isinstance(entry, dict):
            continue
        if "error" in entry:
            continue
        _build_sw_instance(tree_el, entry)
        written += 1

    tree = ET.ElementTree(root)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)

    return {
        "labels": len(entries),
        "written": written,
        "warnings": extractor.warnings[:10],
        "out_cdfx": str(out_path),
    }
