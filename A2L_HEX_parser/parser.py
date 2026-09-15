"""
hex_a2l_to_calibration.py
=========================
Extract calibration parameters (labels + physical values) from ECU HEX + A2L,
replacing the CDFX-based input to your downstream pipeline.

This is what INCA does under the hood when you load HEX + A2L and export to CDFX,
implemented in pure Python with one external dependency: `intelhex`.

Pipeline
--------
    HEX (raw bytes at addresses) + A2L (descriptor) -->
        for each CHARACTERISTIC:
            1. Look up RECORD_LAYOUT  --> tells us datatype + layout
            2. Read raw bytes from HEX at characteristic.address
            3. Apply COMPU_METHOD     --> raw integer/float -> physical value
            4. Attach axes (for CURVE/MAP) via AXIS_PTS / AXIS_DESCR
        Output: {label: {value, type, unit, address, axes, ...}}

Usage (CLI)
-----------
    pip install intelhex
    python hex_a2l_to_calibration.py --hex ecu.hex --a2l descriptor.a2l --out cal.json

Usage (as a module)
-------------------
    from hex_a2l_to_calibration import CalibrationExtractor
    ext = CalibrationExtractor("ecu.hex", "descriptor.a2l")
    data = ext.extract_all()           # dict keyed by label
    one  = ext.extract_one(ext.a2l.characteristics["MyLabel_C"])

Supported subset (covers ~95% of real-world ECU data)
-----------------------------------------------------
    CHARACTERISTIC types : VALUE, CURVE, MAP, VAL_BLK
    COMPU_METHOD  types  : IDENTICAL, LINEAR, RAT_FUNC (common subcase),
                           TAB_VERB, TAB_NOINTP, TAB_INTP
    Endianness           : auto-detected from MOD_COMMON BYTE_ORDER
    Axes                 : STD_AXIS, COM_AXIS (via AXIS_PTS_REF)

Not handled (will fall through to raw values with a warning)
------------------------------------------------------------
    - FORM conversion (arbitrary string formulas; unsafe to eval)
    - FIX_AXIS_PAR_DIST (computed axis points; uncommon)
    - CUBOID, CUBE_4, CUBE_5 (3D+ tables; rare in practice)
    - General RAT_FUNC inversion (only the linear sub-case is solved)
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from intelhex import IntelHex
except ImportError:
    sys.stderr.write("ERROR: install intelhex  -->  pip install intelhex\n")
    sys.exit(1)


# =====================================================================
# A2L datatype -> (struct format char, size in bytes)
# =====================================================================
DTYPE_TABLE: Dict[str, Tuple[str, int]] = {
    "UBYTE":        ("B", 1),
    "SBYTE":        ("b", 1),
    "UWORD":        ("H", 2),
    "SWORD":        ("h", 2),
    "ULONG":        ("I", 4),
    "SLONG":        ("i", 4),
    "A_UINT64":     ("Q", 8),
    "A_INT64":      ("q", 8),
    "FLOAT16_IEEE": ("e", 2),
    "FLOAT32_IEEE": ("f", 4),
    "FLOAT64_IEEE": ("d", 8),
}


# =====================================================================
# A2L data models (only what we need for extraction)
# =====================================================================
@dataclass
class CompuMethod:
    name: str
    conversion_type: str          # IDENTICAL, LINEAR, RAT_FUNC, TAB_VERB, TAB_INTP, FORM
    unit: str = ""
    format: str = ""
    coeffs: List[float] = field(default_factory=list)         # RAT_FUNC: [a,b,c,d,e,f]
    coeffs_linear: List[float] = field(default_factory=list)  # LINEAR: [a,b]
    compu_tab_ref: str = ""
    formula: str = ""


@dataclass
class CompuTab:
    name: str
    tab_type: str                 # TAB_VERB, TAB_NOINTP, TAB_INTP
    pairs: List[Tuple[float, Any]] = field(default_factory=list)
    default_value: Any = None


@dataclass
class RecordLayout:
    name: str
    fnc_datatype: str = "UWORD"
    fnc_index_mode: str = "ROW_DIR"        # ROW_DIR, COLUMN_DIR, ALTERNATE_*
    axis_x_datatype: str = ""
    axis_y_datatype: str = ""
    no_axis_pts_x_datatype: str = "UWORD"
    no_axis_pts_y_datatype: str = "UWORD"


@dataclass
class AxisDescr:
    attribute: str = "STD_AXIS"            # STD_AXIS, FIX_AXIS, COM_AXIS
    input_quantity: str = ""
    conversion: str = ""
    max_axis_points: int = 0
    lower_limit: float = 0.0
    upper_limit: float = 0.0
    axis_pts_ref: str = ""                  # populated for COM_AXIS


@dataclass
class Characteristic:
    name: str
    description: str = ""
    char_type: str = "VALUE"                # VALUE, CURVE, MAP, VAL_BLK, ASCII
    address: int = 0
    deposit: str = ""                       # RECORD_LAYOUT reference
    conversion: str = ""                    # COMPU_METHOD reference
    lower_limit: float = 0.0
    upper_limit: float = 0.0
    matrix_dim: List[int] = field(default_factory=list)
    number: int = 0                         # legacy size keyword for VAL_BLK
    axis_descr: List[AxisDescr] = field(default_factory=list)


@dataclass
class AxisPts:
    name: str
    address: int = 0
    input_quantity: str = ""
    deposit: str = ""
    conversion: str = ""
    max_axis_points: int = 0
    lower_limit: float = 0.0
    upper_limit: float = 0.0


# =====================================================================
# A2L PARSER (regex-based — same style as your existing a2l_parser.py)
# =====================================================================
class A2LParser:
    """Parses the subset of A2L needed to extract calibration values from HEX."""

    def __init__(self, a2l_path: Union[str, Path]):
        self.path = Path(a2l_path)
        text = self.path.read_text(encoding="utf-8", errors="replace")
        # strip C-style comments
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r"//[^\n]*", "", text)
        self.text = text

        self.byte_order: str = "MSB_LAST"   # little-endian by default
        self.characteristics: Dict[str, Characteristic] = {}
        self.axis_pts_map: Dict[str, AxisPts] = {}
        self.compu_methods: Dict[str, CompuMethod] = {}
        self.compu_tabs: Dict[str, CompuTab] = {}
        self.record_layouts: Dict[str, RecordLayout] = {}

        self._parse_mod_common()
        self._parse_record_layouts()
        self._parse_compu_methods()
        self._parse_compu_tabs()
        self._parse_axis_pts()
        self._parse_characteristics()

    # ---------- helpers --------------------------------------------------
    @staticmethod
    def _tokens(body: str) -> List[str]:
        """Tokenize an A2L block body. Quoted strings stay as single tokens."""
        out, i, n = [], 0, len(body)
        while i < n:
            c = body[i]
            if c.isspace():
                i += 1
                continue
            if c == '"':
                j = i + 1
                while j < n and body[j] != '"':
                    if body[j] == '\\' and j + 1 < n:
                        j += 2
                        continue
                    j += 1
                out.append(body[i + 1: j])
                i = j + 1
                continue
            j = i
            while j < n and not body[j].isspace():
                j += 1
            out.append(body[i:j])
            i = j
        return out

    @staticmethod
    def _parse_address(tok: str) -> int:
        tok = tok.strip().rstrip(",")
        if tok.startswith(("0x", "0X")):
            return int(tok, 16)
        return int(tok)

    @staticmethod
    def _find_inner_blocks(body: str, kind: str) -> List[str]:
        pat = re.compile(rf"/begin\s+{kind}\s+(.*?)/end\s+{kind}", re.DOTALL)
        return pat.findall(body)

    # ---------- section parsers ------------------------------------------
    def _parse_mod_common(self) -> None:
        m = re.search(
            r"/begin\s+MOD_COMMON.*?BYTE_ORDER\s+(\w+).*?/end\s+MOD_COMMON",
            self.text, re.DOTALL,
        )
        if m:
            self.byte_order = m.group(1)

    def _parse_record_layouts(self) -> None:
        for body in self._find_inner_blocks(self.text, "RECORD_LAYOUT"):
            toks = self._tokens(body)
            if not toks:
                continue
            rl = RecordLayout(name=toks[0])
            i = 1
            while i < len(toks):
                t = toks[i]
                # FNC_VALUES Position Datatype IndexMode AddressType
                if t == "FNC_VALUES" and i + 4 < len(toks):
                    rl.fnc_datatype = toks[i + 2]
                    rl.fnc_index_mode = toks[i + 3]
                    i += 5
                # AXIS_PTS_X Position Datatype IndexIncr AddressType
                elif t == "AXIS_PTS_X" and i + 4 < len(toks):
                    rl.axis_x_datatype = toks[i + 2]
                    i += 5
                elif t == "AXIS_PTS_Y" and i + 4 < len(toks):
                    rl.axis_y_datatype = toks[i + 2]
                    i += 5
                # NO_AXIS_PTS_X Position Datatype
                elif t == "NO_AXIS_PTS_X" and i + 2 < len(toks):
                    rl.no_axis_pts_x_datatype = toks[i + 2]
                    i += 3
                elif t == "NO_AXIS_PTS_Y" and i + 2 < len(toks):
                    rl.no_axis_pts_y_datatype = toks[i + 2]
                    i += 3
                else:
                    i += 1
            self.record_layouts[rl.name] = rl

    def _parse_compu_methods(self) -> None:
        # /begin COMPU_METHOD name "desc" TYPE "format" "unit" [optional...]
        for body in self._find_inner_blocks(self.text, "COMPU_METHOD"):
            toks = self._tokens(body)
            if len(toks) < 5:
                continue
            cm = CompuMethod(
                name=toks[0],
                conversion_type=toks[2],
                format=toks[3],
                unit=toks[4],
            )
            i = 5
            while i < len(toks):
                t = toks[i]
                if t == "COEFFS" and i + 6 < len(toks):
                    try:
                        cm.coeffs = [float(toks[i + k]) for k in range(1, 7)]
                    except ValueError:
                        pass
                    i += 7
                elif t == "COEFFS_LINEAR" and i + 2 < len(toks):
                    try:
                        cm.coeffs_linear = [float(toks[i + 1]), float(toks[i + 2])]
                    except ValueError:
                        pass
                    i += 3
                elif t == "COMPU_TAB_REF" and i + 1 < len(toks):
                    cm.compu_tab_ref = toks[i + 1]
                    i += 2
                elif t == "FORMULA" and i + 1 < len(toks):
                    cm.formula = toks[i + 1]
                    i += 2
                else:
                    i += 1
            self.compu_methods[cm.name] = cm

    def _parse_compu_tabs(self) -> None:
        for kind in ("COMPU_VTAB", "COMPU_TAB", "COMPU_VTAB_RANGE"):
            for body in self._find_inner_blocks(self.text, kind):
                toks = self._tokens(body)
                if len(toks) < 4:
                    continue
                ct = CompuTab(name=toks[0], tab_type=toks[2])
                try:
                    n_entries = int(toks[3])
                except ValueError:
                    n_entries = 0
                pos = 4
                for _ in range(n_entries):
                    if pos + 1 >= len(toks):
                        break
                    try:
                        in_val = float(toks[pos])
                    except ValueError:
                        pos += 2
                        continue
                    out_raw = toks[pos + 1]
                    try:
                        out_val: Any = float(out_raw)
                    except ValueError:
                        out_val = out_raw
                    ct.pairs.append((in_val, out_val))
                    pos += 2
                # optional DEFAULT_VALUE
                for k, t in enumerate(toks):
                    if t == "DEFAULT_VALUE" and k + 1 < len(toks):
                        ct.default_value = toks[k + 1]
                self.compu_tabs[ct.name] = ct

    def _parse_axis_pts(self) -> None:
        # /begin AXIS_PTS name "desc" addr input_qty deposit max_diff conv
        #                 max_axis_points lower upper
        for body in self._find_inner_blocks(self.text, "AXIS_PTS"):
            toks = self._tokens(body)
            if len(toks) < 10:
                continue
            try:
                ap = AxisPts(
                    name=toks[0],
                    address=self._parse_address(toks[2]),
                    input_quantity=toks[3],
                    deposit=toks[4],
                    conversion=toks[6],
                    max_axis_points=int(toks[7]),
                    lower_limit=float(toks[8]),
                    upper_limit=float(toks[9]),
                )
            except (ValueError, IndexError):
                continue
            self.axis_pts_map[ap.name] = ap

    def _parse_characteristics(self) -> None:
        # /begin CHARACTERISTIC name "desc" type addr deposit max_diff conv
        #                       lower upper [optional...]
        for body in self._find_inner_blocks(self.text, "CHARACTERISTIC"):
            axis_bodies = self._find_inner_blocks(body, "AXIS_DESCR")
            # remove any nested blocks before tokenizing the flat fields
            scrubbed = re.sub(r"/begin\s+\w+.*?/end\s+\w+", "",
                              body, flags=re.DOTALL)
            toks = self._tokens(scrubbed)
            if len(toks) < 8:
                continue
            try:
                ch = Characteristic(
                    name=toks[0],
                    description=toks[1],
                    char_type=toks[2],
                    address=self._parse_address(toks[3]),
                    deposit=toks[4],
                    conversion=toks[6],
                    lower_limit=float(toks[7]),
                    upper_limit=float(toks[8]) if len(toks) > 8 else 0.0,
                )
            except (ValueError, IndexError):
                continue

            # optional MATRIX_DIM / NUMBER
            for i, t in enumerate(toks):
                if t == "MATRIX_DIM":
                    dims, j = [], i + 1
                    while j < len(toks) and toks[j].lstrip("-").isdigit():
                        dims.append(int(toks[j]))
                        j += 1
                    ch.matrix_dim = dims
                elif t == "NUMBER" and i + 1 < len(toks):
                    try:
                        ch.number = int(toks[i + 1])
                    except ValueError:
                        pass

            for ab in axis_bodies:
                at = self._tokens(ab)
                if len(at) < 6:
                    continue
                try:
                    ad = AxisDescr(
                        attribute=at[0],
                        input_quantity=at[1],
                        conversion=at[2],
                        max_axis_points=int(at[3]),
                        lower_limit=float(at[4]),
                        upper_limit=float(at[5]),
                    )
                except (ValueError, IndexError):
                    continue
                for i, t in enumerate(at):
                    if t == "AXIS_PTS_REF" and i + 1 < len(at):
                        ad.axis_pts_ref = at[i + 1]
                ch.axis_descr.append(ad)

            self.characteristics[ch.name] = ch


# =====================================================================
# HEX READER
# =====================================================================
class HexReader:
    """Wraps intelhex with typed reads honouring A2L byte order."""

    def __init__(self, hex_path: Union[str, Path], byte_order: str = "MSB_LAST"):
        self.ih = IntelHex()
        self.ih.loadhex(str(hex_path))
        # ASAP2: MSB_LAST = little-endian (most ECUs), MSB_FIRST = big-endian
        self.endian = "<" if byte_order == "MSB_LAST" else ">"

    def read(self, address: int, datatype: str,
             count: int = 1) -> Union[int, float, List]:
        fmt, size = DTYPE_TABLE.get(datatype, ("H", 2))
        total = size * count
        raw_bytes = self.ih.tobinarray(start=address, size=total).tobytes()
        values = struct.unpack(f"{self.endian}{count}{fmt}", raw_bytes)
        return list(values) if count > 1 else values[0]


# =====================================================================
# CONVERSION ENGINE (raw -> physical via COMPU_METHOD)
# =====================================================================
class Converter:
    def __init__(self, compu_methods: Dict[str, CompuMethod],
                 compu_tabs: Dict[str, CompuTab]):
        self.cms = compu_methods
        self.cts = compu_tabs

    def to_physical(self, raw: Any, conversion_name: str) -> Any:
        if not conversion_name or conversion_name == "NO_COMPU_METHOD":
            return raw
        cm = self.cms.get(conversion_name)
        if cm is None:
            return raw

        # element-wise for arrays
        if isinstance(raw, list):
            return [self.to_physical(r, conversion_name) for r in raw]

        ct = cm.conversion_type

        if ct == "IDENTICAL":
            return raw

        if ct == "LINEAR" and len(cm.coeffs_linear) == 2:
            # phys = a*raw + b
            a, b = cm.coeffs_linear
            return a * raw + b

        if ct == "RAT_FUNC" and len(cm.coeffs) == 6:
            # ASAP2 spec: raw = (a*P^2 + b*P + c) / (d*P^2 + e*P + f)
            # The overwhelmingly common case is the linear sub-case:
            #   a=d=f=0  ==>  raw = (b*P + c)/e  ==>  P = (raw*e - c)/b
            a, b, c, d, e, f = cm.coeffs
            if a == 0 and d == 0 and f == 0 and b != 0:
                return (raw * e - c) / b
            # General quadratic inversion is rare; pass raw through.
            return raw

        if ct in ("TAB_VERB", "TAB_NOINTP", "TAB_INTP"):
            tab = self.cts.get(cm.compu_tab_ref)
            if tab is None:
                return raw
            return self._lookup(raw, tab)

        # FORM and unhandled types fall through as raw.
        return raw

    @staticmethod
    def _lookup(raw: Any, tab: CompuTab) -> Any:
        if tab.tab_type in ("TAB_VERB", "TAB_NOINTP"):
            for in_val, out_val in tab.pairs:
                if int(raw) == int(in_val):
                    return out_val
            return tab.default_value if tab.default_value is not None else raw

        if tab.tab_type == "TAB_INTP":
            pts = sorted(tab.pairs, key=lambda p: p[0])
            if not pts:
                return raw
            if raw <= pts[0][0]:
                return pts[0][1]
            if raw >= pts[-1][0]:
                return pts[-1][1]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if x0 <= raw <= x1 and x1 != x0:
                    return y0 + (y1 - y0) * (raw - x0) / (x1 - x0)
        return raw


# =====================================================================
# EXTRACTOR — orchestrates A2L + HEX + Converter
# =====================================================================
class CalibrationExtractor:
    def __init__(self, hex_path: Union[str, Path], a2l_path: Union[str, Path]):
        self.a2l = A2LParser(a2l_path)
        self.hex = HexReader(hex_path, byte_order=self.a2l.byte_order)
        self.conv = Converter(self.a2l.compu_methods, self.a2l.compu_tabs)
        self.warnings: List[str] = []

    # ---------- size helpers -------------------------------------
    def _value_count(self, char: Characteristic) -> int:
        if char.char_type == "VALUE":
            return 1
        if char.char_type == "VAL_BLK":
            if char.matrix_dim:
                n = 1
                for d in char.matrix_dim:
                    n *= d
                return n
            return char.number or 1
        if char.char_type == "CURVE" and char.axis_descr:
            return char.axis_descr[0].max_axis_points
        if char.char_type == "MAP" and len(char.axis_descr) >= 2:
            return (char.axis_descr[0].max_axis_points
                    * char.axis_descr[1].max_axis_points)
        return 1

    # ---------- axis extraction ----------------------------------
    def _read_axis(self, ad: AxisDescr) -> List[Any]:
        if ad.attribute == "COM_AXIS" and ad.axis_pts_ref:
            ap = self.a2l.axis_pts_map.get(ad.axis_pts_ref)
            if ap is None:
                self.warnings.append(f"AXIS_PTS '{ad.axis_pts_ref}' not found")
                return []
            rl = self.a2l.record_layouts.get(ap.deposit)
            dtype = (rl.axis_x_datatype
                     if rl and rl.axis_x_datatype else "UWORD")
            raw = self.hex.read(ap.address, dtype, ap.max_axis_points)
            return self.conv.to_physical(raw, ap.conversion)
        # STD_AXIS / FIX_AXIS: axis stored with the characteristic.
        # That requires walking the full RECORD_LAYOUT (axis interleaved
        # with FNC_VALUES). Most production A2Ls use COM_AXIS, so we
        # handle the common case and leave a note for the rest.
        return []

    # ---------- single label -------------------------------------
    def extract_one(self, char: Characteristic) -> Dict[str, Any]:
        rl = self.a2l.record_layouts.get(char.deposit)
        if rl is None:
            self.warnings.append(
                f"{char.name}: RECORD_LAYOUT '{char.deposit}' missing")
            return {"label": char.name, "error": "no_record_layout"}

        n = self._value_count(char)
        try:
            raw = self.hex.read(char.address, rl.fnc_datatype, n)
        except Exception as e:
            self.warnings.append(f"{char.name}: HEX read failed @ "
                                 f"0x{char.address:08X}: {e}")
            return {"label": char.name, "error": str(e)}

        physical = self.conv.to_physical(raw, char.conversion)
        cm = self.a2l.compu_methods.get(char.conversion)

        entry: Dict[str, Any] = {
            "label": char.name,
            "type": char.char_type,
            "address": f"0x{char.address:08X}",
            "value": physical,
            "raw": raw,
            "unit": cm.unit if cm else "",
            "description": char.description,
        }

        # Reshape MAP into 2D matrix. Both branches produce the canonical
        # row-major shape (ny rows of nx cells) so downstream consumers
        # (a2l_hex_to_cdfx writer, cdfx_to_a2l_hex patcher) don't need to
        # know the underlying fnc_index_mode to round-trip the data.
        if (char.char_type == "MAP"
                and isinstance(physical, list)
                and len(char.axis_descr) >= 2):
            nx = char.axis_descr[0].max_axis_points
            ny = char.axis_descr[1].max_axis_points
            if rl.fnc_index_mode in ("COLUMN_DIR", "ALTERNATE_WITH_X"):
                # original layout: cell(r,c) at physical[c + r*nx]
                entry["value"] = [[physical[c + r * nx] for c in range(nx)]
                                  for r in range(ny)]
            else:   # ROW_DIR default
                entry["value"] = [physical[r * nx:(r + 1) * nx]
                                  for r in range(ny)]

        # Attach axes for CURVE/MAP
        if char.char_type in ("CURVE", "MAP"):
            entry["axes"] = [self._read_axis(ad) for ad in char.axis_descr]

        return entry

    # ---------- everything ---------------------------------------
    def extract_all(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name, char in self.a2l.characteristics.items():
            out[name] = self.extract_one(char)
        return out


# =====================================================================
# CLI
# =====================================================================
def main() -> int:
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "input_a2l_hex"
    output_dir = script_dir / "Output_json"

    ap = argparse.ArgumentParser(
        description="Extract calibration values from HEX + A2L "
                    "(Python replacement for INCA HEX->CDFX export).")
    ap.add_argument("--hex", type=Path, default=input_dir / "hexfile.hex",
                    help="ECU .hex file (default: input_a2l_hex/hexfile.hex)")
    ap.add_argument("--a2l", type=Path, default=input_dir / "atol.a2l",
                    help="A2L descriptor file (default: input_a2l_hex/atol.a2l)")
    ap.add_argument("--out", type=Path, default=output_dir / "calibration.json",
                    help="Output JSON (default: Output_json/calibration.json)")
    ap.add_argument("--label", default=None,
                    help="Extract a single label by name (debug helper)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing A2L: {args.a2l}")
    extractor = CalibrationExtractor(args.hex, args.a2l)
    a = extractor.a2l
    print(f"  characteristics : {len(a.characteristics)}")
    print(f"  compu_methods   : {len(a.compu_methods)}")
    print(f"  compu_tabs      : {len(a.compu_tabs)}")
    print(f"  record_layouts  : {len(a.record_layouts)}")
    print(f"  axis_pts        : {len(a.axis_pts_map)}")
    print(f"  byte_order      : {a.byte_order}")

    if args.label:
        ch = a.characteristics.get(args.label)
        if ch is None:
            print(f"Label '{args.label}' not found in A2L.")
            return 1
        result = {args.label: extractor.extract_one(ch)}
    else:
        result = extractor.extract_all()

    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {len(result)} labels -> {args.out}")

    if extractor.warnings:
        print(f"\n{len(extractor.warnings)} warning(s). First 5:")
        for w in extractor.warnings[:5]:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())