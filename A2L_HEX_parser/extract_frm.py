# #!/usr/bin/env python3
# """
# frm_start_values.py  (v4 - hybrid text + OCR)
# =============================================
# Scan an FRM calibration PDF and extract every calibration *label* with its
# *Start Value* (or *Standard Value*) + unit + IF-condition into an Excel file.

# Works on BOTH kinds of FRM PDF:
#   * Text-layer PDFs (the normal export)     -> exact labels, no OCR.
#   * Image-only / scanned / "flattened" PDFs -> automatically OCR'd.

# Why you may have seen an empty output
# -------------------------------------
# If a PDF is printed/trimmed/re-saved in a way that rasterizes the pages, it has
# NO text layer and pdfplumber returns 0 characters -> 0 rows. v4 detects that per
# page and falls back to OCR (Tesseract). OCR rows are flagged "OCR (verify)" in
# the Source column because OCR can misread characters (e.g. ElM_ vs E1M_), and a
# diagonal watermark can corrupt the rows it crosses. For exact labels, always
# prefer running on the ORIGINAL text-layer PDF.

# Usage:
#     python frm_start_values.py input.pdf
#     python frm_start_values.py input.pdf -o labels.xlsx
#     python frm_start_values.py input.pdf --ocr always     # force OCR every page
#     python frm_start_values.py input.pdf --ocr never       # text layer only
#     python frm_start_values.py input.pdf --debug           # diagnose

# Requires: pdfplumber, openpyxl
# Optional (only needed for image-only PDFs):
#     pip install pymupdf pytesseract pillow
#     plus the Tesseract engine:  Windows -> https://github.com/UB-Mannheim/tesseract/wiki
#                                 Linux   -> sudo apt install tesseract-ocr
# """
# from __future__ import annotations

# import argparse
# import re
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Iterable, Sequence


# # --------------------------------------------------------------------------- #
# # 1. Patterns
# # --------------------------------------------------------------------------- #
# LABEL_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]*[_.][A-Za-z0-9_.]*[-\u2212\u00ad]?$")
# IF_LINE_RE = re.compile(r"^IF\s*\(.*?i\s*\(", re.IGNORECASE)
# COND_NAME_RE = re.compile(r"i\s*\(\s*([A-Za-z0-9_]+)\s*\)", re.IGNORECASE)
# START_VALUE_RE = re.compile(
#     r"(?:Start|Standard)\s*Value\s*[:\uff1a]\s*(?P<val>.*?)\s*"
#     r"(?:\[\s*(?P<unit>[^\]]*?)\s*\])?\s*$",
#     re.IGNORECASE,
# )
# TABLE_PAGE_RE = re.compile(
#     r"Calibration\s+Hints?|(?:Start|Standard)\s*Value|Label\s*name|IF\s*Condition",
#     re.IGNORECASE,
# )
# FUNCTION_HEADER_RE = re.compile(
#     r"\b[A-Za-z][A-Za-z0-9_]*_[A-Za-z0-9_]*\s+"
#     r"\d+(?:\.\d+){1,3}_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*;\d+\b"
# )
# HEADER_SCAN_LINES = 12

# WRAP_CHARS = ("\u2212", "-", "\u00ad")
# CID_RE = re.compile(r"\(cid:\d+\)")

# ARTIFACT_RE = re.compile(r"0123456789\?")
# NOISE_RES = [
#     re.compile(r"^COPRSI", re.IGNORECASE),
#     re.compile(r"RobertBoschGmbH", re.IGNORECASE),
#     re.compile(r"reserve\w*all\s*right", re.IGNORECASE),
#     re.compile(r"thirdpartie", re.IGNORECASE),
#     re.compile(r"industrialpropertyright", re.IGNORECASE),
#     re.compile(r"^Label\s*name\b", re.IGNORECASE),
#     re.compile(r"\d+\.\d+\.\d+;\d+\s+\d+\s*\|"),
# ]

# # Conservative OCR repairs for well-known, systematic misreads on these docs.
# def ocr_repair(text: str) -> str:
#     text = text.replace("E1M_", "ElM_").replace("E]M_", "ElM_")
#     text = re.sub(r"\bEMO(\d)", r"EM0\1", text)         # EMO1PRSNT -> EM01PRSNT
#     text = re.sub(r"([A-Za-z0-9])_c\b", r"\1_C", text)  # trailing _c -> _C
#     return text


# # --------------------------------------------------------------------------- #
# # 2. Data model
# # --------------------------------------------------------------------------- #
# @dataclass
# class Record:
#     label: str
#     start_value: str
#     unit: str
#     condition: str = ""
#     description: str = ""
#     page: int = 0
#     reconstructed: bool = False
#     ocr: bool = False


# # --------------------------------------------------------------------------- #
# # 3. Extraction (hybrid: text layer, OCR fallback)
# # --------------------------------------------------------------------------- #
# def normalize(text: str) -> str:
#     return CID_RE.sub("", text.replace("\u2212", "-"))


# def clean_page(lines: list[str]) -> list[str]:
#     out: list[str] = []
#     for ln in lines:
#         ln = ARTIFACT_RE.sub("", ln).strip()
#         if not ln or ln == "|":
#             continue
#         if any(rx.search(ln) for rx in NOISE_RES):
#             continue
#         out.append(ln)
#     return out


# def _expand_page_indices(indices: Iterable[int], page_count: int, context: int) -> list[int]:
#     selected: set[int] = set()
#     for idx in indices:
#         start = max(0, idx - context)
#         end = min(page_count, idx + context + 1)
#         selected.update(range(start, end))
#     return sorted(selected)


# def _page_has_function_header(text: str, max_lines: int = HEADER_SCAN_LINES) -> bool:
#     lines = [line.strip() for line in normalize(text).splitlines() if line.strip()]
#     return any(FUNCTION_HEADER_RE.search(line) for line in lines[:max_lines])


# def _candidate_page_indices(path: Path, context: int = 0,
#                             require_function_header: bool = True) -> tuple[list[int], int]:
#     """Return likely calibration-hint table pages using fast text search."""
#     import fitz

#     doc = fitz.open(str(path))
#     try:
#         hits: list[int] = []
#         for idx, page in enumerate(doc):
#             text = page.get_text("text") or ""
#             if not TABLE_PAGE_RE.search(text):
#                 continue
#             has_function_header = _page_has_function_header(text)
#             if require_function_header and not has_function_header:
#                 continue
#             if not require_function_header or has_function_header:
#                 hits.append(idx)
#         return _expand_page_indices(hits, len(doc), context), len(doc)
#     finally:
#         doc.close()


# def _raw_text_pages(path: Path, engine: str, page_indices: Sequence[int] | None = None) -> list[str]:
#     if engine == "pymupdf":
#         import fitz
#         doc = fitz.open(str(path))
#         indices = page_indices if page_indices is not None else range(len(doc))
#         out = [(doc[i].get_text("text") or "") for i in indices]
#         doc.close()
#         return out
#     import pdfplumber
#     with pdfplumber.open(str(path)) as pdf:
#         indices = page_indices if page_indices is not None else range(len(pdf.pages))
#         return [(pdf.pages[i].extract_text() or "") for i in indices]


# def _ocr_pages(path: Path, which: list[int], dpi: int) -> dict[int, str]:
#     """OCR the given page indices; returns {index: text}. Raises if OCR unavailable."""
#     import io
#     import fitz
#     import pytesseract
#     from PIL import Image

#     result: dict[int, str] = {}
#     doc = fitz.open(str(path))
#     for i in which:
#         pix = doc[i].get_pixmap(dpi=dpi)
#         img = Image.open(io.BytesIO(pix.tobytes("png")))
#         result[i] = ocr_repair(pytesseract.image_to_string(img))
#     doc.close()
#     return result


# def extract_pages(path: Path, engine: str = "pdfplumber",
#                   ocr_mode: str = "auto", ocr_dpi: int = 300,
#                   scan_mode: str = "function-tables", table_context: int = 0):
#     """
#     Returns (pages, ocr_flags, page_numbers) where pages is list[list[str]] (cleaned lines)
#     and ocr_flags[i] is True if page i was produced by OCR.
#     """
#     page_indices: list[int] | None = None
#     total_pages: int | None = None
#     if scan_mode in {"function-tables", "tables"}:
#         require_function_header = scan_mode == "function-tables"
#         try:
#             page_indices, total_pages = _candidate_page_indices(
#                 path,
#                 table_context,
#                 require_function_header=require_function_header,
#             )
#         except ImportError:
#             print("  [warn] Fast table-page scan needs pymupdf. Falling back to full scan.", file=sys.stderr)
#         if page_indices == [] and scan_mode == "tables":
#             print("  [warn] No Start/Standard Value table pages found by fast scan. Falling back to full scan.", file=sys.stderr)
#             page_indices = None
#         elif page_indices == []:
#             print("  [warn] No pages found with both a function/version header and calibration-hint table markers. "
#                   "Retry with --scan-mode tables or --scan-mode full if this FRM uses a different header layout.",
#                   file=sys.stderr)
#     if total_pages is None:
#         total_pages = len(page_indices) if page_indices is not None else None

#     raw = _raw_text_pages(path, engine, page_indices)
#     ocr_flags = [False] * len(raw)
#     page_numbers = [i + 1 for i in page_indices] if page_indices is not None else list(range(1, len(raw) + 1))

#     # Decide which pages need OCR.
#     if ocr_mode == "always":
#         targets = list(range(len(raw)))
#     elif ocr_mode == "auto":
#         targets = [i for i, t in enumerate(raw) if len((t or "").strip()) < 40]
#     else:  # never
#         targets = []

#     if targets:
#         try:
#             original_targets = [page_numbers[i] - 1 for i in targets]
#             ocr_text = _ocr_pages(path, original_targets, ocr_dpi)
#             for i, t in ocr_text.items():
#                 local_idx = page_numbers.index(i + 1)
#                 raw[local_idx] = t
#                 ocr_flags[local_idx] = True
#         except ImportError:
#             print("  [warn] OCR needed for image-only pages but pymupdf/pytesseract/pillow "
#                   "not installed. Install: pip install pymupdf pytesseract pillow", file=sys.stderr)
#         except Exception as e:  # e.g. TesseractNotFoundError
#             print(f"  [warn] OCR unavailable ({type(e).__name__}). Install the Tesseract engine. "
#                   "Text-only pages will still be processed.", file=sys.stderr)

#     pages = [clean_page([ln.strip() for ln in normalize(t).splitlines() if ln.strip()]) for t in raw]
#     return pages, ocr_flags, page_numbers, total_pages


# # --------------------------------------------------------------------------- #
# # 4. Parser
# # --------------------------------------------------------------------------- #
# def _clean_val(v: str) -> str:
#     return re.sub(r"\s+", " ", v).strip()


# def parse_records(pages: list[list[str]], ocr_flags: list[bool] | None = None,
#                   page_numbers: list[int] | None = None) -> list[Record]:
#     ocr_flags = ocr_flags or [False] * len(pages)
#     page_numbers = page_numbers or list(range(1, len(pages) + 1))
#     records: list[Record] = []

#     condition = ""
#     label: str | None = None
#     desc_parts: list[str] = []
#     label_open = False
#     expect_label = False
#     recon = False

#     def reset():
#         nonlocal label, desc_parts, label_open, recon
#         label, desc_parts, label_open, recon = None, [], False, False

#     def emit(sv, page_idx):
#         records.append(Record(
#             label=label,
#             start_value=_clean_val(sv.group("val")),
#             unit=(sv.group("unit") or "").strip(),
#             condition=condition,
#             description=" ".join(desc_parts).strip(),
#             page=page_numbers[page_idx],
#             reconstructed=recon,
#             ocr=ocr_flags[page_idx],
#         ))

#     for page_idx, lines in enumerate(pages):
#         for line in lines:
#             if IF_LINE_RE.match(line):
#                 condition = " || ".join(COND_NAME_RE.findall(line))
#                 expect_label = True
#                 reset()
#                 continue

#             first_tok = line.split(None, 1)[0]
#             rest = line[len(first_tok):].strip()
#             sv = START_VALUE_RE.search(line)

#             if expect_label and label is None:
#                 expect_label = False
#                 if LABEL_TOKEN_RE.match(first_tok):
#                     label = first_tok
#                     label_open = first_tok.endswith(WRAP_CHARS)
#                     desc_parts = [rest] if (rest and sv is None) else []
#                     recon = False
#                     if sv is not None and not label_open:
#                         emit(sv, page_idx)
#                         reset()
#                 continue

#             if label is not None:
#                 if label_open:
#                     label = label.rstrip("".join(WRAP_CHARS)) + first_tok
#                     recon = True
#                     label_open = label.endswith(WRAP_CHARS)
#                     if sv is not None and not label_open:
#                         emit(sv, page_idx)
#                         reset()
#                     elif rest:
#                         desc_parts.append(rest)
#                     continue
#                 if sv is not None:
#                     emit(sv, page_idx)
#                     reset()
#                     continue
#                 desc_parts.append(line)
#                 continue

#     return records


# # --------------------------------------------------------------------------- #
# # 5. Excel writer
# # --------------------------------------------------------------------------- #
# def write_excel(records: Iterable[Record], out_path: Path) -> int:
#     from openpyxl import Workbook
#     from openpyxl.styles import Font, PatternFill, Alignment
#     from openpyxl.utils import get_column_letter

#     records = list(records)
#     wb = Workbook()
#     ws = wb.active
#     ws.title = "Start Values"

#     headers = ["Label name", "Start Value", "Unit", "IF Condition",
#                "Description", "Page", "Reconstructed?", "Source"]
#     ws.append(headers)
#     fill = PatternFill("solid", fgColor="4A4F5A")
#     font = Font(bold=True, color="FFFFFF")
#     for col in range(1, len(headers) + 1):
#         c = ws.cell(row=1, column=col)
#         c.fill, c.font = fill, font
#         c.alignment = Alignment(vertical="center")

#     for r in records:
#         ws.append([r.label, r.start_value, r.unit, r.condition, r.description,
#                    r.page, "YES" if r.reconstructed else "",
#                    "OCR (verify)" if r.ocr else "text"])

#     ws.freeze_panes = "A2"
#     ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{ws.max_row}"
#     for i, w in enumerate([42, 22, 8, 22, 60, 6, 14, 14], start=1):
#         ws.column_dimensions[get_column_letter(i)].width = w

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     wb.save(str(out_path))
#     return len(records)


# # --------------------------------------------------------------------------- #
# # 6. Diagnostics
# # --------------------------------------------------------------------------- #
# def run_debug(pages, ocr_flags, page_numbers=None, total_pages=None) -> None:
#     page_numbers = page_numbers or list(range(1, len(pages) + 1))
#     total_chars = sum(len(ln) for pg in pages for ln in pg)
#     empty_pages = sum(1 for pg in pages if not pg)
#     ocr_pages = sum(1 for f in ocr_flags if f)
#     if_hits = sum(1 for pg in pages for ln in pg if IF_LINE_RE.match(ln))
#     sv_hits = sum(1 for pg in pages for ln in pg
#                   if re.search(r"(?:Start|Standard)\s*Value", ln, re.IGNORECASE)
#                   and START_VALUE_RE.search(ln))

#     print("=" * 66)
#     print("DEBUG DIAGNOSTIC")
#     print("=" * 66)
#     if total_pages and total_pages != len(pages):
#         print(f"pages scanned ........... {len(pages)} of {total_pages}")
#     else:
#         print(f"pages ................... {len(pages)}")
#     print(f"pages produced by OCR ... {ocr_pages}")
#     print(f"empty (no-text) pages ... {empty_pages}")
#     print(f"total characters ........ {total_chars}")
#     print(f"IF(i(...)) lines ........ {if_hits}")
#     print(f"Start/Standard Value .... {sv_hits}   (expected number of rows ~ this)")
#     print("-" * 66)
#     for i, pg in enumerate(pages):
#         if any(re.search(r"(?:Start|Standard)\s*Value", ln, re.IGNORECASE) for ln in pg):
#             tag = " [OCR]" if ocr_flags[i] else ""
#             print(f"RAW LINES of first page with a value (page {page_numbers[i]}{tag}):")
#             print("-" * 66)
#             for ln in pg[:40]:
#                 print(repr(ln))
#             break
#     print("=" * 66)


# # --------------------------------------------------------------------------- #
# # 7. CLI
# # --------------------------------------------------------------------------- #
# def main(argv: list[str] | None = None) -> int:
#     ap = argparse.ArgumentParser(description="Extract calibration labels + Start Values from an FRM PDF into Excel.")
#     ap.add_argument("input", type=Path)
#     ap.add_argument("-o", "--output", type=Path, default=None)
#     ap.add_argument("--engine", choices=["pdfplumber", "pymupdf"], default="pdfplumber")
#     ap.add_argument("--ocr", choices=["auto", "always", "never"], default="auto",
#                     help="auto (default): OCR only pages with no text layer.")
#     ap.add_argument("--ocr-dpi", type=int, default=300)
#     ap.add_argument("--scan-mode", choices=["function-tables", "tables", "full"], default="function-tables",
#                     help="function-tables (default): scan pages with a function/version header and table markers; "
#                          "tables: scan pages with table markers only; full: scan every page.")
#     ap.add_argument("--table-context", type=int, default=0,
#                     help="Extra pages before/after each detected table page to include in table scan modes.")
#     ap.add_argument("--label-regex", default=None)
#     ap.add_argument("--debug", action="store_true")
#     args = ap.parse_args(argv)

#     if not args.input.exists():
#         print(f"error: file not found: {args.input}", file=sys.stderr)
#         return 2
#     if args.label_regex:
#         global LABEL_TOKEN_RE
#         LABEL_TOKEN_RE = re.compile(args.label_regex)

#     pages, ocr_flags, page_numbers, total_pages = extract_pages(
#         args.input,
#         engine=args.engine,
#         ocr_mode=args.ocr,
#         ocr_dpi=args.ocr_dpi,
#         scan_mode=args.scan_mode,
#         table_context=max(0, args.table_context),
#     )

#     if args.debug:
#         run_debug(pages, ocr_flags, page_numbers, total_pages)
#         return 0

#     out = args.output or args.input.with_name(args.input.stem + "_start_values.xlsx")
#     records = parse_records(pages, ocr_flags, page_numbers)
#     n = write_excel(records, out)

#     n_ocr = sum(1 for r in records if r.ocr)
#     n_recon = sum(1 for r in records if r.reconstructed)
#     scope = f"{len(pages)} of {total_pages}" if total_pages and total_pages != len(pages) else str(len(pages))
#     print(f"Parsed {n} labels from {scope} pages -> {out}   (engine={args.engine}, scan={args.scan_mode})")
#     if n_ocr:
#         print(f"  {n_ocr} row(s) came from OCR (Source='OCR (verify)') - check those label names.")
#     if n_recon:
#         print(f"  {n_recon} label(s) rebuilt from wrapped lines (Reconstructed?='YES').")
#     if n == 0:
#           print("\n0 rows. Run with --debug. If the function-header scan missed an unusual layout, retry with "
#               "--scan-mode tables or --scan-mode full. If 'total characters' is ~0, "
#               "the PDF has no text layer: install OCR (pip install pymupdf pytesseract pillow + Tesseract engine), "
#               "or run on the ORIGINAL text-layer PDF.")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())


#!/usr/bin/env python3
"""
frm_start_values.py  (v5 - fast page filter + hybrid text/OCR)
==============================================================
Scan an FRM calibration PDF and extract every calibration *label* with its
*Start Value* (or *Standard Value*) + unit + IF-condition into an Excel file.

Speed: only pages that look like calibration-hint tables are parsed. A page
qualifies when it has BOTH
  (a) a function/version header, e.g.  "ElM_Engine 35.16.0;1"   (often
      extracted glued: "ElM_Engine35.16.0;1"), and
  (b) a table marker: "Start Value" / "Standard Value" / "Label name".
Figure pages, system-constant tables and DFC pages are skipped.

Robustness: if the PDF has no text layer (scanned / "flattened" / trimmed),
pages are OCR'd automatically and those rows are flagged "OCR (verify)".

Usage:
    python frm_start_values.py input.pdf
    python frm_start_values.py input.pdf -o labels.xlsx
    python frm_start_values.py input.pdf --scan-mode full     # parse every page
    python frm_start_values.py input.pdf --table-context 1    # +/-1 page around hits
    python frm_start_values.py input.pdf --ocr never
    python frm_start_values.py input.pdf --debug

Requires: pdfplumber, openpyxl, pymupdf
Optional (image-only PDFs): pytesseract, pillow + the Tesseract engine
    Windows -> https://github.com/UB-Mannheim/tesseract/wiki
    Linux   -> sudo apt install tesseract-ocr
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


# --------------------------------------------------------------------------- #
# 1. Patterns
# --------------------------------------------------------------------------- #
LABEL_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]*[_.][A-Za-z0-9_.]*[-\u2212\u00ad]?$")
IF_LINE_RE = re.compile(r"^IF\s*\(.*?i\s*\(", re.IGNORECASE)
COND_NAME_RE = re.compile(r"i\s*\(\s*([A-Za-z0-9_]+)\s*\)", re.IGNORECASE)
START_VALUE_RE = re.compile(
    r"(?:Start|Standard)\s*Value\s*[:\uff1a]\s*(?P<val>.*?)\s*"
    r"(?:\[\s*(?P<unit>[^\]]*?)\s*\])?\s*$",
    re.IGNORECASE,
)

# A page is a candidate only if it carries one of these table markers.
TABLE_PAGE_RE = re.compile(
    r"(?:Start|Standard)\s*Value|Label\s*name",
    re.IGNORECASE,
)

# Function/version header, e.g. "ElM_Engine 35.16.0;1" or glued "ElM_Engine35.16.0;1".
#   name_part           -> ElM_Engine / ElM_PlausChk / ElM_StAct
#   version             -> 35.16.0  (2-4 dotted numbers)
#   ;revision           -> ;1
# NOTE: no underscore-suffix after the version (that was the v4 bug), and the
# separator is \s* because the text layer usually emits no space.
FUNCTION_HEADER_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*\s*\d+(?:\.\d+){1,3}\s*;\s*\d+"
)
HEADER_SCAN_LINES = 12          # header sits at the very top of the page

WRAP_CHARS = ("\u2212", "-", "\u00ad")
CID_RE = re.compile(r"\(cid:\d+\)")

ARTIFACT_RE = re.compile(r"0123456789\?")
NOISE_RES = [
    re.compile(r"^COPRSI", re.IGNORECASE),
    re.compile(r"RobertBoschGmbH", re.IGNORECASE),
    re.compile(r"reserve\w*all\s*right", re.IGNORECASE),
    re.compile(r"thirdpartie", re.IGNORECASE),
    re.compile(r"industrialpropertyright", re.IGNORECASE),
    re.compile(r"^Label\s*name\b", re.IGNORECASE),
    re.compile(r"\d+\.\d+\.\d+;\d+\s+\d+\s*\|"),   # page header line
]


def ocr_repair(text: str) -> str:
    """Conservative repairs for systematic OCR misreads seen on these documents."""
    text = text.replace("E1M_", "ElM_").replace("E]M_", "ElM_")
    text = re.sub(r"\bEMO(\d)", r"EM0\1", text)          # EMO1PRSNT -> EM01PRSNT
    text = re.sub(r"([A-Za-z0-9])_c\b", r"\1_C", text)   # trailing _c -> _C
    return text


# --------------------------------------------------------------------------- #
# 2. Data model
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    label: str
    start_value: str
    unit: str
    condition: str = ""
    description: str = ""
    page: int = 0
    reconstructed: bool = False
    ocr: bool = False


# --------------------------------------------------------------------------- #
# 3. Extraction
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    return CID_RE.sub("", text.replace("\u2212", "-"))


def clean_page(lines: list[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        ln = ARTIFACT_RE.sub("", ln).strip()
        if not ln or ln == "|":
            continue
        if any(rx.search(ln) for rx in NOISE_RES):
            continue
        out.append(ln)
    return out


def _expand_page_indices(indices: Iterable[int], page_count: int, context: int) -> list[int]:
    selected: set[int] = set()
    for idx in indices:
        selected.update(range(max(0, idx - context), min(page_count, idx + context + 1)))
    return sorted(selected)


def page_has_function_header(text: str, max_lines: int = HEADER_SCAN_LINES) -> bool:
    lines = [ln.strip() for ln in normalize(text).splitlines() if ln.strip()]
    return any(FUNCTION_HEADER_RE.search(ln) for ln in lines[:max_lines])


def select_pages(path: Path, scan_mode: str, context: int):
    """
    Decide which page indices to parse.

    Returns (indices_or_None, total_pages, note).
    indices_or_None is None  -> parse every page (full scan).

    Falls back safely:
      * no text layer at all      -> full scan (so OCR can run)
      * function-tables finds 0   -> retry 'tables'
      * tables finds 0            -> full scan
    """
    import fitz

    doc = fitz.open(str(path))
    try:
        total = len(doc)
        texts = [(pg.get_text("text") or "") for pg in doc]
    finally:
        doc.close()

    if scan_mode == "full":
        return None, total, "full scan"

    # Image-only / no text layer: a text scan cannot find anything -> full scan + OCR.
    if sum(len(t.strip()) for t in texts) < 40 * max(1, total) // 10:
        if sum(len(t.strip()) for t in texts) == 0:
            return None, total, "no text layer -> full scan (OCR)"

    table_hits = [i for i, t in enumerate(texts) if TABLE_PAGE_RE.search(t)]
    if not table_hits:
        return None, total, "no table markers found -> full scan"

    if scan_mode == "function-tables":
        hits = [i for i in table_hits if page_has_function_header(texts[i])]
        if hits:
            return _expand_page_indices(hits, total, context), total, "function header + table markers"
        print("  [warn] No page had BOTH a function/version header and a table marker; "
              "falling back to --scan-mode tables.", file=sys.stderr)

    return _expand_page_indices(table_hits, total, context), total, "table markers"


def _raw_text_pages(path: Path, engine: str, page_indices: Sequence[int] | None) -> list[str]:
    if engine == "pymupdf":
        import fitz
        doc = fitz.open(str(path))
        idxs = list(page_indices) if page_indices is not None else range(len(doc))
        out = [(doc[i].get_text("text") or "") for i in idxs]
        doc.close()
        return out
    import pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        idxs = list(page_indices) if page_indices is not None else range(len(pdf.pages))
        return [(pdf.pages[i].extract_text() or "") for i in idxs]


def _ocr_pages(path: Path, which: list[int], dpi: int) -> dict[int, str]:
    """OCR the given 0-based PDF page indices; returns {index: text}."""
    import io
    import fitz
    import pytesseract
    from PIL import Image

    result: dict[int, str] = {}
    doc = fitz.open(str(path))
    for i in which:
        pix = doc[i].get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        result[i] = ocr_repair(pytesseract.image_to_string(img))
    doc.close()
    return result


def extract_pages(path: Path, engine: str = "pdfplumber", ocr_mode: str = "auto",
                  ocr_dpi: int = 300, scan_mode: str = "function-tables",
                  table_context: int = 0):
    """Returns (pages, ocr_flags, page_numbers, total_pages, note)."""
    try:
        page_indices, total_pages, note = select_pages(path, scan_mode, table_context)
    except ImportError:
        print("  [warn] Fast page filter needs pymupdf (pip install pymupdf). Using full scan.", file=sys.stderr)
        page_indices, total_pages, note = None, None, "full scan (pymupdf missing)"

    raw = _raw_text_pages(path, engine, page_indices)
    if total_pages is None:
        total_pages = len(raw)
    page_numbers = [i + 1 for i in page_indices] if page_indices is not None else list(range(1, len(raw) + 1))
    ocr_flags = [False] * len(raw)

    if ocr_mode == "always":
        targets = list(range(len(raw)))
    elif ocr_mode == "auto":
        targets = [i for i, t in enumerate(raw) if len((t or "").strip()) < 40]
    else:
        targets = []

    if targets:
        try:
            pdf_indices = [page_numbers[i] - 1 for i in targets]
            ocr_text = _ocr_pages(path, pdf_indices, ocr_dpi)
            for local_i, pdf_i in zip(targets, pdf_indices):
                raw[local_i] = ocr_text[pdf_i]
                ocr_flags[local_i] = True
        except ImportError:
            print("  [warn] OCR needed but pytesseract/pillow/pymupdf missing: "
                  "pip install pymupdf pytesseract pillow", file=sys.stderr)
        except Exception as e:
            print(f"  [warn] OCR unavailable ({type(e).__name__}); install the Tesseract engine.", file=sys.stderr)

    pages = [clean_page([ln.strip() for ln in normalize(t).splitlines() if ln.strip()]) for t in raw]
    return pages, ocr_flags, page_numbers, total_pages, note


# --------------------------------------------------------------------------- #
# 4. Parser
# --------------------------------------------------------------------------- #
def _clean_val(v: str) -> str:
    return re.sub(r"\s+", " ", v).strip()


def parse_records(pages: list[list[str]], ocr_flags: list[bool] | None = None,
                  page_numbers: list[int] | None = None) -> list[Record]:
    ocr_flags = ocr_flags or [False] * len(pages)
    page_numbers = page_numbers or list(range(1, len(pages) + 1))
    records: list[Record] = []

    condition = ""
    label: str | None = None
    desc_parts: list[str] = []
    label_open = False
    expect_label = False
    recon = False
    prev_page_no: int | None = None

    def reset():
        nonlocal label, desc_parts, label_open, recon
        label, desc_parts, label_open, recon = None, [], False, False

    def emit(sv, page_idx):
        records.append(Record(
            label=label,
            start_value=_clean_val(sv.group("val")),
            unit=(sv.group("unit") or "").strip(),
            condition=condition,
            description=" ".join(desc_parts).strip(),
            page=page_numbers[page_idx],
            reconstructed=recon,
            ocr=ocr_flags[page_idx],
        ))

    for page_idx, lines in enumerate(pages):
        page_no = page_numbers[page_idx]
        # Pages were filtered: if we jumped over a gap, don't carry parser state
        # across the missing pages (it would join unrelated rows).
        if prev_page_no is not None and page_no != prev_page_no + 1:
            reset()
            condition, expect_label = "", False
        prev_page_no = page_no

        for line in lines:
            if IF_LINE_RE.match(line):
                condition = " || ".join(COND_NAME_RE.findall(line))
                expect_label = True
                reset()
                continue

            first_tok = line.split(None, 1)[0]
            rest = line[len(first_tok):].strip()
            sv = START_VALUE_RE.search(line)

            if expect_label and label is None:
                expect_label = False
                if LABEL_TOKEN_RE.match(first_tok):
                    label = first_tok
                    label_open = first_tok.endswith(WRAP_CHARS)
                    desc_parts = [rest] if (rest and sv is None) else []
                    recon = False
                    if sv is not None and not label_open:
                        emit(sv, page_idx)
                        reset()
                continue

            if label is not None:
                if label_open:
                    label = label.rstrip("".join(WRAP_CHARS)) + first_tok
                    recon = True
                    label_open = label.endswith(WRAP_CHARS)
                    if sv is not None and not label_open:
                        emit(sv, page_idx)
                        reset()
                    elif rest:
                        desc_parts.append(rest)
                    continue
                if sv is not None:
                    emit(sv, page_idx)
                    reset()
                    continue
                desc_parts.append(line)
                continue

    return records


# --------------------------------------------------------------------------- #
# 5. Excel writer
# --------------------------------------------------------------------------- #
def write_excel(records: Iterable[Record], out_path: Path) -> int:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    records = list(records)
    wb = Workbook()
    ws = wb.active
    ws.title = "Start Values"

    headers = ["Label name", "Start Value", "Unit", "IF Condition",
               "Description", "Page", "Reconstructed?", "Source"]
    ws.append(headers)
    fill = PatternFill("solid", fgColor="4A4F5A")
    font = Font(bold=True, color="FFFFFF")
    for col in range(1, len(headers) + 1):
        c = ws.cell(row=1, column=col)
        c.fill, c.font = fill, font
        c.alignment = Alignment(vertical="center")

    for r in records:
        ws.append([r.label, r.start_value, r.unit, r.condition, r.description,
                   r.page, "YES" if r.reconstructed else "",
                   "OCR (verify)" if r.ocr else "text"])

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{ws.max_row}"
    for i, w in enumerate([42, 22, 8, 22, 60, 6, 14, 14], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))
    return len(records)


# --------------------------------------------------------------------------- #
# 6. Diagnostics
# --------------------------------------------------------------------------- #
def run_debug(pages, ocr_flags, page_numbers, total_pages, note) -> None:
    total_chars = sum(len(ln) for pg in pages for ln in pg)
    ocr_pages = sum(1 for f in ocr_flags if f)
    if_hits = sum(1 for pg in pages for ln in pg if IF_LINE_RE.match(ln))
    sv_hits = sum(1 for pg in pages for ln in pg
                  if re.search(r"(?:Start|Standard)\s*Value", ln, re.IGNORECASE)
                  and START_VALUE_RE.search(ln))

    print("=" * 66)
    print("DEBUG DIAGNOSTIC")
    print("=" * 66)
    print(f"page selection .......... {note}")
    print(f"pages parsed ............ {len(pages)} of {total_pages}")
    print(f"pages produced by OCR ... {ocr_pages}")
    print(f"total characters ........ {total_chars}")
    print(f"IF(i(...)) lines ........ {if_hits}")
    print(f"Start/Standard Value .... {sv_hits}   (expected number of rows ~ this)")
    if pages:
        print(f"selected page numbers ... {page_numbers[:15]}{' ...' if len(page_numbers) > 15 else ''}")
    print("-" * 66)
    for i, pg in enumerate(pages):
        if any(re.search(r"(?:Start|Standard)\s*Value", ln, re.IGNORECASE) for ln in pg):
            tag = " [OCR]" if ocr_flags[i] else ""
            print(f"RAW LINES of first page with a value (page {page_numbers[i]}{tag}):")
            print("-" * 66)
            for ln in pg[:40]:
                print(repr(ln))
            break
    print("=" * 66)


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract calibration labels + Start Values from an FRM PDF into Excel.")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--engine", choices=["pdfplumber", "pymupdf"], default="pdfplumber")
    ap.add_argument("--ocr", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--ocr-dpi", type=int, default=300)
    ap.add_argument("--scan-mode", choices=["function-tables", "tables", "full"], default="function-tables",
                    help="function-tables (default): pages with a function/version header AND a table marker.")
    ap.add_argument("--table-context", type=int, default=0,
                    help="Also parse N pages before/after each detected table page.")
    ap.add_argument("--label-regex", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"error: file not found: {args.input}", file=sys.stderr)
        return 2
    if args.label_regex:
        global LABEL_TOKEN_RE
        LABEL_TOKEN_RE = re.compile(args.label_regex)

    pages, ocr_flags, page_numbers, total_pages, note = extract_pages(
        args.input, engine=args.engine, ocr_mode=args.ocr, ocr_dpi=args.ocr_dpi,
        scan_mode=args.scan_mode, table_context=max(0, args.table_context),
    )

    if args.debug:
        run_debug(pages, ocr_flags, page_numbers, total_pages, note)
        return 0

    out = args.output or args.input.with_name(args.input.stem + "_start_values.xlsx")
    records = parse_records(pages, ocr_flags, page_numbers)
    n = write_excel(records, out)

    n_ocr = sum(1 for r in records if r.ocr)
    n_recon = sum(1 for r in records if r.reconstructed)
    print(f"Parsed {n} labels from {len(pages)} of {total_pages} pages -> {out}")
    print(f"  page selection: {note}   engine={args.engine}")
    if n_ocr:
        print(f"  {n_ocr} row(s) from OCR (Source='OCR (verify)') - verify those label names.")
    if n_recon:
        print(f"  {n_recon} label(s) rebuilt from wrapped lines (Reconstructed?='YES').")
    if n == 0:
        print("\n0 rows. Try:  --scan-mode full --debug")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())