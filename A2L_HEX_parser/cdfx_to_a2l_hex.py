"""
cdfx_to_a2l_hex.py
==================
Inverse of `a2l_hex_to_cdfx` / `parser.CalibrationExtractor`.

Given an A2L descriptor, an original HEX file, and an updated CDFX, produce a
new HEX file whose calibration regions reflect the CDFX values. The A2L is
not modified (it only describes structure); the original HEX is loaded so
non-calibration bytes remain intact.

Supported per-label:
    VALUE, VAL_BLK, CURVE, MAP (same coverage as the forward extractor).
    COMPU_METHOD: IDENTICAL, LINEAR, RAT_FUNC linear sub-case,
                  TAB_VERB / TAB_NOINTP (reverse lookup),
                  TAB_INTP (nearest-pair fallback).
    Axes: COM_AXIS via AXIS_PTS_REF (STD_AXIS / FIX_AXIS are left untouched).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import lxml.etree as ET
from intelhex import IntelHex

from parser import (
    DTYPE_TABLE,
    A2LParser,
    AxisDescr,
    AxisPts,
    Characteristic,
    CompuMethod,
    CompuTab,
    RecordLayout,
)


NAMESPACE = {"autosar": "http://autosar.org/schema/r4.0"}


# ---------------------------------------------------------------------------
# Inverse compu-method
# ---------------------------------------------------------------------------
class _Inverter:
    """Physical -> raw conversion (inverse of parser.Converter.to_physical)."""

    def __init__(self,
                 compu_methods: Dict[str, CompuMethod],
                 compu_tabs: Dict[str, CompuTab]) -> None:
        self.cms = compu_methods
        self.cts = compu_tabs

    def to_raw(self, phys: Any, conversion_name: str) -> Any:
        if not conversion_name or conversion_name == "NO_COMPU_METHOD":
            return phys
        cm = self.cms.get(conversion_name)
        if cm is None:
            return phys

        ct = cm.conversion_type
        if ct == "IDENTICAL":
            return phys

        if ct == "LINEAR" and len(cm.coeffs_linear) == 2:
            # forward: phys = a*raw + b   =>   raw = (phys - b) / a
            a, b = cm.coeffs_linear
            if a == 0:
                return phys
            return (float(phys) - b) / a

        if ct == "RAT_FUNC" and len(cm.coeffs) == 6:
            # Parser forward (linear sub-case, see parser.Converter.to_physical):
            #   phys = (raw*e - c) / b    -- only applied when a=d=f=0 AND b!=0,
            # otherwise parser returns `raw` unchanged. Mirror the SAME gate
            # here so labels parser left untouched are written back untouched.
            #   inverse:  raw = (phys*b + c) / e   (b != 0 implied; e!=0 also
            #   needed to avoid a div/0, but real-world coeffs satisfy both).
            a, b, c, d, e, f = cm.coeffs
            if a == 0 and d == 0 and f == 0 and b != 0 and e != 0:
                return (float(phys) * b + c) / e
            return phys

        if ct in ("TAB_VERB", "TAB_NOINTP"):
            tab = self.cts.get(cm.compu_tab_ref)
            if tab is None:
                return phys
            for in_val, out_val in tab.pairs:
                if _equalish(out_val, phys):
                    return in_val
            return phys  # unknown enum -> leave raw region untouched downstream

        if ct == "TAB_INTP":
            tab = self.cts.get(cm.compu_tab_ref)
            if tab is None or not tab.pairs:
                return phys
            try:
                target = float(phys)
            except (TypeError, ValueError):
                return phys
            best = None
            best_d = float("inf")
            for in_val, out_val in tab.pairs:
                try:
                    d = abs(float(out_val) - target)
                except (TypeError, ValueError):
                    continue
                if d < best_d:
                    best_d = d
                    best = in_val
            return best if best is not None else phys

        return phys


def _equalish(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return str(a) == str(b)
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
_INT_BOUNDS = {
    "B": (0, 255),
    "b": (-128, 127),
    "H": (0, 65535),
    "h": (-32768, 32767),
    "I": (0, 4294967295),
    "i": (-2147483648, 2147483647),
    "Q": (0, (1 << 64) - 1),
    "q": (-(1 << 63), (1 << 63) - 1),
}


def _pack_one(raw: Any, dtype: str, endian: str) -> bytes:
    fmt, _size = DTYPE_TABLE.get(dtype, ("H", 2))
    if fmt in _INT_BOUNDS:
        try:
            ival = int(round(float(raw)))
        except (TypeError, ValueError):
            ival = 0
        lo, hi = _INT_BOUNDS[fmt]
        if ival < lo:
            ival = lo
        elif ival > hi:
            ival = hi
        return struct.pack(f"{endian}{fmt}", ival)
    # float types: e / f / d
    try:
        fval = float(raw)
    except (TypeError, ValueError):
        fval = 0.0
    return struct.pack(f"{endian}{fmt}", fval)


# ---------------------------------------------------------------------------
# CDFX value readers (mirror the selectors used in pipeline.py)
# ---------------------------------------------------------------------------
def _text_list(elems: Sequence[ET._Element]) -> List[str]:
    out: List[str] = []
    for el in elems:
        if el.text is not None and el.text.strip() != "":
            out.append(el.text.strip())
    return out


def _ordered_v_vt(container: ET._Element) -> List[str]:
    """Return text of V/VT children in document order (skips empties).

    CDFX rows may interleave numeric (V) and enum-string (VT) cells; reading
    them as two separate findall() calls loses positional order, which
    silently mangles tables with mixed cell kinds (e.g. TAB_VERB CURVE/MAPs).
    """
    out: List[str] = []
    if container is None:
        return out
    for child in container:
        tag = ET.QName(child).localname if isinstance(child.tag, str) else None
        if tag in ("V", "VT") and child.text is not None and child.text.strip() != "":
            out.append(child.text.strip())
    return out


def _read_scalar(sw: ET._Element) -> Optional[str]:
    v = sw.find("SW-VALUE-CONT/SW-VALUES-PHYS/V", namespaces=NAMESPACE)
    if v is not None and v.text and v.text.strip():
        return v.text.strip()
    vt = sw.find("SW-VALUE-CONT/SW-VALUES-PHYS/VT", namespaces=NAMESPACE)
    if vt is not None and vt.text and vt.text.strip():
        return vt.text.strip()
    return None


def _read_flat(sw: ET._Element) -> List[str]:
    return _ordered_v_vt(sw.find("SW-VALUE-CONT/SW-VALUES-PHYS",
                                 namespaces=NAMESPACE))


def _read_map_rows(sw: ET._Element) -> List[List[str]]:
    rows: List[List[str]] = []
    sw_values = sw.find("SW-VALUE-CONT/SW-VALUES-PHYS", namespaces=NAMESPACE)
    if sw_values is None:
        return rows
    for vg in sw_values.findall("VG", namespaces=NAMESPACE):
        rows.append(_ordered_v_vt(vg))
    return rows


def _read_axes(sw: ET._Element) -> List[List[str]]:
    axes: List[List[str]] = []
    for ac in sw.findall(".//SW-AXIS-CONT", namespaces=NAMESPACE):
        axes.append(_ordered_v_vt(ac.find("SW-VALUES-PHYS",
                                          namespaces=NAMESPACE)))
    return axes


# ---------------------------------------------------------------------------
# Main patcher
# ---------------------------------------------------------------------------
def _write_bytes(ih: IntelHex, address: int, data: bytes) -> None:
    for i, b in enumerate(data):
        ih[address + i] = b


def _flatten_map(rows: List[List[str]], char: Characteristic) -> List[str]:
    """
    Flatten CDFX MAP rows (ny x nx, row-major as our writer emits) to the flat
    physical-array order parser.HexReader produced. parser indexes both
    ROW_DIR and COLUMN_DIR maps using physical[r*nx + c] (see CalibrationExtractor.
    extract_one MAP reshape), so a single row-major flatten matches both modes.
    """
    if len(char.axis_descr) < 2:
        # fall back to row concatenation
        return [v for row in rows for v in row]
    nx = char.axis_descr[0].max_axis_points
    ny = char.axis_descr[1].max_axis_points
    flat: List[str] = []
    for r in range(ny):
        row = rows[r] if r < len(rows) else []
        for c in range(nx):
            flat.append(row[c] if c < len(row) else "")
    return flat


def patch_hex_from_cdfx(a2l_path: Union[str, Path],
                        hex_path: Union[str, Path],
                        cdfx_path: Union[str, Path],
                        out_hex_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Patch `hex_path` with values read from `cdfx_path`, using `a2l_path` for
    structural metadata. Write the result to `out_hex_path` (Intel HEX format).

    Returns: {patched, axes_patched, skipped_no_char, skipped_no_layout,
              skipped_no_char_axis_pts, skipped_no_char_unknown,
              errors, errors_total, warnings, warnings_total, out_hex}.
    """
    a2l = A2LParser(a2l_path)
    inverter = _Inverter(a2l.compu_methods, a2l.compu_tabs)
    endian = "<" if a2l.byte_order == "MSB_LAST" else ">"

    ih = IntelHex()
    ih.loadhex(str(hex_path))

    tree = ET.parse(str(cdfx_path))
    root = tree.getroot()

    patched = 0
    skipped_no_char = 0
    skipped_no_layout = 0
    patched_labels: List[str] = []
    skipped_no_char_labels: List[str] = []
    skipped_no_char_axis_pts_labels: List[str] = []
    skipped_no_char_unknown_labels: List[str] = []
    skipped_no_layout_labels: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []
    axes_done: set[str] = set()

    for sw in root.findall(".//SW-INSTANCE", namespaces=NAMESPACE):
        sn = sw.find("SHORT-NAME", namespaces=NAMESPACE)
        if sn is None or not sn.text:
            continue
        label = sn.text.strip()

        char = a2l.characteristics.get(label)
        if char is None:
            skipped_no_char += 1
            skipped_no_char_labels.append(label)
            if label in a2l.axis_pts_map:
                skipped_no_char_axis_pts_labels.append(label)
            else:
                skipped_no_char_unknown_labels.append(label)
            continue
        rl = a2l.record_layouts.get(char.deposit)
        if rl is None:
            skipped_no_layout += 1
            skipped_no_layout_labels.append(label)
            warnings.append(f"{label}: RECORD_LAYOUT '{char.deposit}' missing")
            continue

        try:
            # ---- collect values in flat order ----
            if char.char_type == "VALUE":
                v = _read_scalar(sw)
                values: List[str] = [v] if v is not None else []
            elif char.char_type == "VAL_BLK":
                values = _read_flat(sw)
            elif char.char_type == "CURVE":
                values = _read_flat(sw)
            elif char.char_type == "MAP":
                values = _flatten_map(_read_map_rows(sw), char)
            else:
                continue

            if not values:
                continue

            # ---- pack & patch function values ----
            buf = bytearray()
            for v in values:
                try:
                    phys: Any = float(v)
                except (TypeError, ValueError):
                    phys = v  # leave as-is for verbatim tab lookup
                raw = inverter.to_raw(phys, char.conversion)
                buf += _pack_one(raw, rl.fnc_datatype, endian)
            _write_bytes(ih, char.address, bytes(buf))

            # ---- axes (COM_AXIS only) ----
            if char.char_type in ("CURVE", "MAP"):
                axes_in_cdfx = _read_axes(sw)
                for ai, ad in enumerate(char.axis_descr):
                    if ai >= len(axes_in_cdfx):
                        break
                    if ad.attribute != "COM_AXIS" or not ad.axis_pts_ref:
                        continue
                    if ad.axis_pts_ref in axes_done:
                        continue
                    ap = a2l.axis_pts_map.get(ad.axis_pts_ref)
                    if ap is None:
                        continue
                    arl = a2l.record_layouts.get(ap.deposit)
                    adtype = (arl.axis_x_datatype
                              if arl and arl.axis_x_datatype else "UWORD")
                    abuf = bytearray()
                    for v in axes_in_cdfx[ai][:ap.max_axis_points]:
                        try:
                            phys = float(v)
                        except (TypeError, ValueError):
                            phys = v
                        raw = inverter.to_raw(phys, ap.conversion)
                        abuf += _pack_one(raw, adtype, endian)
                    if abuf:
                        _write_bytes(ih, ap.address, bytes(abuf))
                        axes_done.add(ad.axis_pts_ref)

            patched += 1
            patched_labels.append(label)
        except Exception as e:  # noqa: BLE001 — collect & continue
            errors.append(f"{label}: {type(e).__name__}: {e}")

    Path(out_hex_path).parent.mkdir(parents=True, exist_ok=True)
    ih.write_hex_file(str(out_hex_path))

    return {
        "patched": patched,
        "patched_labels": patched_labels,
        "axes_patched": len(axes_done),
        "skipped_no_char": skipped_no_char,
        "skipped_no_char_labels": skipped_no_char_labels,
        "skipped_no_char_axis_pts": len(skipped_no_char_axis_pts_labels),
        "skipped_no_char_axis_pts_labels": skipped_no_char_axis_pts_labels,
        "skipped_no_char_unknown": len(skipped_no_char_unknown_labels),
        "skipped_no_char_unknown_labels": skipped_no_char_unknown_labels,
        "skipped_no_layout": skipped_no_layout,
        "skipped_no_layout_labels": skipped_no_layout_labels,
        "errors": errors[:20],
        "errors_all": errors,
        "errors_total": len(errors),
        "warnings": warnings[:20],
        "warnings_all": warnings,
        "warnings_total": len(warnings),
        "out_hex": str(out_hex_path),
    }
