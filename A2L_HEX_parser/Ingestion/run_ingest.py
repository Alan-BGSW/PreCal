"""Run the ingestion end to end.

    python run_ingest.py old  ElM_Engine_v1400.pdf
    python run_ingest.py new  ElM_Engine_v1410.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
import fitz  # PyMuPDF
from ingest_labels_1 import ingest


def extract_pages(pdf_path: str) -> list[str]:
    """One string per page. De-hyphenate the PDF line-wrap artifact (D−\\nV -> DV)."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        text = text.replace("\u2212", "-")        # normalize the minus glyph
        text = text.replace("-\n", "")             # join hyphenated wraps
        pages.append(text)
    doc.close()
    return pages


def _configure_local_embedding_device(requested_device: str | None) -> None:
    """Set local embedding device before ingest_labels_1 initializes the model."""
    if requested_device:
        os.environ["LOCAL_EMBED_DEVICE"] = requested_device
    os.environ.setdefault("EMBED_BACKEND", "local")


def _print_gpu_preflight(require_gpu: bool = False) -> None:
    device_setting = os.getenv("LOCAL_EMBED_DEVICE", "auto")
    backend = os.getenv("EMBED_BACKEND", "local")
    print(f"[preflight] EMBED_BACKEND={backend}")
    print(f"[preflight] LOCAL_EMBED_DEVICE={device_setting}")

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        requested_cuda = device_setting.startswith("cuda") or device_setting in {"gpu"}
        print(f"[preflight] torch.cuda.is_available={cuda_ok}")
        print(f"[preflight] torch.version.cuda={torch.version.cuda}")
        if cuda_ok:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "unknown-gpu"
            print(f"[preflight] detected_gpu={gpu_name}")
        else:
            print("[preflight] CUDA not available in current torch build/runtime.")
            if require_gpu or requested_cuda:
                raise SystemExit(
                    "GPU was requested, but CUDA is unavailable in this Python environment. "
                    "Install a CUDA-enabled torch build in Ingestion/.venv and retry."
                )
    except Exception as exc:
        print(f"[preflight] torch check skipped ({exc}).")
        if require_gpu:
            raise SystemExit(
                "GPU was required, but torch is not importable in this environment."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDF label ingestion into Chroma.")
    parser.add_argument("role", choices=["old", "new"], help="Document role for metadata.")
    parser.add_argument("pdf_path", help="Path to source PDF.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
        default="auto",
        help="Override LOCAL_EMBED_DEVICE for local embeddings.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail fast unless CUDA is available (recommended for GPU runs).",
    )
    args = parser.parse_args()

    _configure_local_embedding_device(args.device)
    if args.require_gpu:
        os.environ["LOCAL_EMBED_STRICT_DEVICE"] = "1"
    _print_gpu_preflight(require_gpu=args.require_gpu)

    role, pdf_path = args.role, args.pdf_path
    pages = extract_pages(pdf_path)
    n = ingest(pages, doc_role=role)
    print(f"ingested {n} label records from {pdf_path} as doc_role='{role}'")