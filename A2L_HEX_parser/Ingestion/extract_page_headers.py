from __future__ import annotations

import sys
import re
import fitz  # PyMuPDF


_HEADER_PATTERNS = [
    # "Table 14394 FC : ElM_Engine / 1400.0.0; 2"
    re.compile(r"FC\s*:\s*([A-Za-z][A-Za-z0-9_]*)\s*/\s*[\d.]+\s*;\s*\d+"),
    # "36.1.2 [ElM_Engine 1400.0.0;2] Handling of electrical machine"
    re.compile(r"\[\s*([A-Za-z][A-Za-z0-9_]*)\s+[\d.]+\s*;\s*\d+\s*\]"),
    # "ElM_Engine 1400.0.0;2 11601 | 20159"  (top-of-page line)
    re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+[\d.]+\s*;\s*\d+\b", re.MULTILINE),
]


def extract_header_name(page_text: str) -> str | None:
    for pattern in _HEADER_PATTERNS:
        for line in page_text.splitlines():
            m = pattern.search(line)
            if m:
                return m.group(1).strip()
    return None


def extract_page_headers(pdf_path: str) -> dict[int, str]:
    result: dict[int, str] = {}
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        name = extract_header_name(text)
        if name:
            result[i] = name
    doc.close()
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_page_headers.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    headers = extract_page_headers(pdf_path)

    if not headers:
        print("No function headers detected in the PDF.")
    else:
        for page_no, name in headers.items():
            print(f"Page {page_no}: {name}")