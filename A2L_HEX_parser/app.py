"""
app.py — Streamlit UI for the A2L/HEX/CDFX calibration pipeline.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

import os
import json
import shutil
import tempfile
import time
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from io import StringIO
from pathlib import Path

import streamlit as st

import pipeline
from apply_frm_start_values import apply_values_to_cdfx, read_frm_values, write_log
from a2l_hex_to_cdfx import write_cdfx_from_a2l_hex
from cdfx_to_a2l_hex import patch_hex_from_cdfx
from detailed_process_logs import write_detailed_logs
from extract_frm import extract_pages, parse_records, write_excel
from log_delivery import upload_log_json


st.set_page_config(page_title="Pre Calibration tool",
                   layout="wide")

# Hide Streamlit's built-in upload size hint (e.g., "500MB per file").
st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzoneInstructions"] div:nth-child(2) {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Pre Calibration tool")
st.caption("Upload Source and destination files and run the calibration pipeline to get updated A2L + HEX files and CDFX files.")


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
SS = st.session_state
SS.setdefault("workdir", tempfile.mkdtemp(prefix="cal_pipeline_"))
SS.setdefault("dest_path", None)              # working CDFX (input or generated)
SS.setdefault("dest_original_name", None)     # filename to use for download
SS.setdefault("dest_cdfx_uploaded", False)    # true only for user-provided CDFX
SS.setdefault("dest_uploads", [])             # raw paths the user dropped for dest
SS.setdefault("dest_a2l_path", None)          # original dest A2L (if dest was A2L+HEX)
SS.setdefault("dest_hex_path", None)          # original dest HEX (if dest was A2L+HEX)
SS.setdefault("source_paths", [])             # raw paths the user dropped for sources
SS.setdefault("converted_cdfxs", [])          # CDFX files produced by A2L+HEX conversion
SS.setdefault("pipeline_result", None)
SS.setdefault("grouped_xlsx_path", None)
SS.setdefault("group_a2l_path", None)
SS.setdefault("patch_a2l_path", None)         # A2L used to patch HEX (defaults to dest_a2l_path)
SS.setdefault("patch_hex_path", None)         # original HEX to patch (defaults to dest_hex_path)
SS.setdefault("patched_hex_path", None)       # output patched HEX
SS.setdefault("patched_hex_info", None)       # patcher status dict
SS.setdefault("pipeline_cdfx_path", None)     # latest CDFX path used by pipeline
SS.setdefault("detailed_log_json_path", None) # detailed JSON log for pipeline+patch
SS.setdefault("detailed_log_xlsx_path", None) # detailed Excel log for pipeline+patch
SS.setdefault("log_upload_info", None)        # OneDrive/SharePoint upload status

SS.setdefault("updated_cdfx_bytes", None)     # snapshot of updated CDFX
SS.setdefault("updated_cdfx_name", None)      # filename for the download
SS.setdefault("frm_pdf_path", None)           # uploaded FRM/software PDF
SS.setdefault("frm_xlsx_path", None)          # extracted FRM start values workbook
SS.setdefault("frm_cdfx_path", None)          # CDFX after applying FRM start values
SS.setdefault("frm_hex_path", None)           # HEX after applying FRM start values
SS.setdefault("frm_log_path", None)           # FRM apply log workbook
SS.setdefault("frm_patch_info", None)         # patcher status after FRM step
SS.setdefault("frm_extract_info", None)       # extraction status after FRM step
SS.setdefault("frm_values_log_json_path", None)  # step-6 detailed label/value json log
SS.setdefault("frm_values_upload_info", None)    # OneDrive/SharePoint upload status for step-6 log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_upload(uploaded, target_dir: str) -> str:
    target = os.path.join(target_dir, f"{uuid.uuid4().hex}_{uploaded.name}")
    with open(target, "wb") as f:
        f.write(uploaded.getbuffer())
    return target


def _classify(paths: list[str]) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """
    Split raw uploads into (cdfx_paths, a2l_hex_pairs, ignored).
    A2L is paired with HEX by case-insensitive stem first. If any matched
    A2L/HEX files remain, leftover files are paired in upload order.
    """
    cdfx = [p for p in paths if p.lower().endswith(".cdfx")]
    a2ls = [(i, p) for i, p in enumerate(paths) if p.lower().endswith(".a2l")]
    hexs = [(i, p) for i, p in enumerate(paths) if p.lower().endswith(".hex")]

    pairs: list[tuple[str, str]] = []
    used_a, used_h = set(), set()

    hex_by_stem: dict[str, list[tuple[int, str]]] = {}
    for idx, p in hexs:
        hex_by_stem.setdefault(Path(p).stem.lower(), []).append((idx, p))

    for a_idx, a_path in a2ls:
        stem = Path(a_path).stem.lower()
        matches = hex_by_stem.get(stem, [])
        if matches:
            h_idx, h_path = matches.pop(0)
            pairs.append((a_path, h_path))
            used_a.add(a_idx)
            used_h.add(h_idx)

    leftover_a = [p for idx, p in a2ls if idx not in used_a]
    leftover_h = [p for idx, p in hexs if idx not in used_h]
    if len(leftover_a) == len(leftover_h) and leftover_a:
        pairs.extend(zip(leftover_a, leftover_h))
        leftover_a, leftover_h = [], []

    ignored = [p for p in paths
               if not p.lower().endswith(".cdfx")
               and not p.lower().endswith(".a2l")
               and not p.lower().endswith(".hex")]
    ignored.extend(leftover_a)
    ignored.extend(leftover_h)
    return cdfx, pairs, ignored


def _convert_pair(a2l: str, hex_path: str, out_dir: str) -> dict:
    """Convert one A2L+HEX pair to a minimal CDFX. Returns a status dict."""
    buf, err = StringIO(), StringIO()
    out_cdfx = os.path.join(out_dir, f"{Path(a2l).stem}_generated.CDFX")
    try:
        with redirect_stdout(buf), redirect_stderr(err):
            info = write_cdfx_from_a2l_hex(a2l, hex_path, out_cdfx)
        return {"ok": True, "stdout": buf.getvalue(), "stderr": err.getvalue(), **info}
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
            "stdout": buf.getvalue(),
            "stderr": err.getvalue(),
        }


def _read_cdfx_download_bytes(path: str | Path) -> bytes:
    raw = Path(path).read_bytes()
    doctype = (
        b'<!DOCTYPE MSRSW PUBLIC '
        b'"-//ASAM//DTD CALIBRATION DATA FORMAT:V2.1:LAI:IAI:XML //EN" '
        b'"cdf_v2.1.0.sl.dtd">\n'
    )
    if b'<!DOCTYPE' not in raw:
        if raw.lstrip().startswith(b'<?xml'):
            end = raw.find(b'?>') + 2
            raw = raw[:end] + b'\n' + doctype + raw[end:]
        else:
            raw = doctype + raw
    return raw


def _write_step6_start_values_log(
    *,
    out_path: Path,
    frm_values,
    apply_logs,
    patch_info: dict,
    extract_info: dict,
) -> None:
    """Write a detailed JSON log for step 6 with label/start-value outcomes."""
    apply_index = {row.label: row for row in apply_logs}
    rows: list[dict] = []

    for item in frm_values:
        row = apply_index.get(item.label)
        rows.append({
            "label": item.label,
            "start_value": item.start_value,
            "unit": item.unit,
            "condition": item.condition,
            "status": row.status if row else "not_processed",
            "detail": row.detail if row else "",
            "values_written": row.values_written if row else 0,
            "old_values_sample": row.old_values_sample if row else "",
            "new_values_sample": row.new_values_sample if row else "",
        })

    payload = {
        "log_type": "step6_start_values",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "labels_in_frm": len(frm_values),
            "labels_in_log": len(rows),
            "cdfx_updated": extract_info.get("cdfx_updated", 0),
            "cdfx_unchanged": extract_info.get("cdfx_unchanged", 0),
            "missing_in_cdfx": extract_info.get("missing_in_cdfx", 0),
            "skipped": extract_info.get("skipped", 0),
            "hex_labels_patched": patch_info.get("patched", 0),
        },
        "patch_info": patch_info,
        "labels": rows,
        "raw_apply_rows": [asdict(row) for row in apply_logs],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Workspace")
    st.code(SS.workdir, language=None)
    if st.button("Reset workspace", type="secondary"):
        try:
            shutil.rmtree(SS.workdir, ignore_errors=True)
        finally:
            SS.workdir = tempfile.mkdtemp(prefix="cal_pipeline_")
            SS.dest_path = None
            SS.dest_original_name = None
            SS.dest_cdfx_uploaded = False
            SS.dest_uploads = []
            SS.dest_a2l_path = None
            SS.dest_hex_path = None
            SS.source_paths = []
            SS.converted_cdfxs = []
            SS.pipeline_result = None
            SS.grouped_xlsx_path = None
            SS.group_a2l_path = None
            SS.patch_a2l_path = None
            SS.patch_hex_path = None
            SS.patched_hex_path = None
            SS.patched_hex_info = None
            SS.pipeline_cdfx_path = None
            SS.detailed_log_json_path = None
            SS.detailed_log_xlsx_path = None
            SS.log_upload_info = None
            SS.updated_cdfx_bytes = None
            SS.updated_cdfx_name = None
            SS.frm_pdf_path = None
            SS.frm_xlsx_path = None
            SS.frm_cdfx_path = None
            SS.frm_hex_path = None
            SS.frm_log_path = None
            SS.frm_patch_info = None
            SS.frm_extract_info = None
            SS.frm_values_log_json_path = None
            SS.frm_values_upload_info = None
            st.rerun()


# ---------------------------------------------------------------------------
# Step 1 — Source files (CDFX or A2L+HEX pairs)
# ---------------------------------------------------------------------------
st.header("Step 1 · Upload source files ")
st.markdown(
    " - Upload `.CDFX` files or `.a2l` + `.hex` pairs.  \n"
    " - .a2l and hex files should be uploaded together as a pair.  \n"
    " - Priority order will be based on the sequence of uploaded files."
)
src_uploads = st.file_uploader(
    "Source files (multi-select supported)",
    accept_multiple_files=True,
    type=["cdfx", "CDFX", "a2l", "A2L", "hex", "HEX"],
    key="src_upload",
)
if src_uploads:
    existing_original_names = {Path(p).name.split("_", 1)[-1] for p in SS.source_paths}
    new_paths = []
    for u in src_uploads:
        if u.name in existing_original_names:
            continue
        new_paths.append(_save_upload(u, SS.workdir))
    if new_paths:
        SS.source_paths.extend(new_paths)

src_cdfxs, src_pairs, src_ignored = (_classify(SS.source_paths)
                                     if SS.source_paths else ([], [], []))
if SS.source_paths:
    with st.expander("Detected source files", expanded=True):
        if src_pairs:
            st.write("**A2L + HEX pairs to convert:**")
            for a, h in src_pairs:
                st.write(f"- `{os.path.basename(a)}`  ↔  `{os.path.basename(h)}`")
        if src_cdfxs:
            st.write("**Direct CDFX sources:**")
            for p in src_cdfxs:
                st.write(f"- `{os.path.basename(p)}`")
        if src_ignored:
            st.warning("Unpaired/unknown source files (ignored):\n"
                       + "\n".join(f"- `{os.path.basename(p)}`" for p in src_ignored))


# ---------------------------------------------------------------------------
# Step 2 — Destination input (A2L + HEX for patching, + real CDFX to update)
# ---------------------------------------------------------------------------
st.header("Step 2 · Upload destination files")
st.markdown(
    " - Upload one matching `.a2l` + `.hex` pair (used for HEX patching and FUNCTION grouping).\n"
    " - Optionally upload the real INCA-exported destination `.CDFX` if you also want an "
    "updated CDFX download after the run."
)

# --- Destination A2L + HEX (for patching / grouping) ---
dest_uploads = st.file_uploader(
    "Destination A2L + HEX",
    accept_multiple_files=True,
    type=["a2l", "A2L", "hex", "HEX"],
    key="dest_upload",
)
if dest_uploads:
    SS.dest_uploads = [_save_upload(u, SS.workdir) for u in dest_uploads]
    SS.dest_a2l_path = None
    SS.dest_hex_path = None
    SS.patch_a2l_path = None
    SS.patch_hex_path = None
    SS.group_a2l_path = None
    cdfxs, pairs, ignored = _classify(SS.dest_uploads)

    if pairs and not cdfxs:
        a2l, hex_path = pairs[0]
        SS.dest_a2l_path = a2l
        SS.dest_hex_path = hex_path
        SS.patch_a2l_path = a2l
        SS.patch_hex_path = hex_path
        SS.group_a2l_path = a2l
        # Destination CDFX now comes from a direct upload (below), not generated
        # from A2L+HEX — so we deliberately do NOT touch SS.dest_path here.
        st.success(
            f"Destination pair detected: `{os.path.basename(a2l)}` + "
            f"`{os.path.basename(hex_path)}`"
        )
        st.info("One-click run will patch HEX and group attention labels. If a destination CDFX is uploaded, it will also be updated for download.")
    else:
        st.warning("Could not detect a usable destination pair. Upload one matching A2L+HEX pair.")
    if ignored:
        st.warning("Unpaired/unknown destination files (ignored):\n"
                   + "\n".join(f"- `{os.path.basename(p)}`" for p in ignored))
elif SS.dest_a2l_path and SS.dest_hex_path:
    st.info(
        f"Current destination pair: `{os.path.basename(SS.dest_a2l_path)}` + "
        f"`{os.path.basename(SS.dest_hex_path)}`"
    )

# --- Destination CDFX (real INCA file that actually gets updated) ---
dest_cdfx_upload = st.file_uploader(
    "Destination CDFX (optional, real INCA-exported CDFX for updated CDFX download)",
    accept_multiple_files=False,
    type=["cdfx", "CDFX"],
    key="dest_cdfx_upload",
)
if dest_cdfx_upload:
    is_new_dest_cdfx = (
        not SS.dest_cdfx_uploaded
        or SS.dest_original_name != dest_cdfx_upload.name
        or not SS.dest_path
        or not os.path.exists(SS.dest_path)
    )
    if is_new_dest_cdfx:
        SS.dest_path = _save_upload(dest_cdfx_upload, SS.workdir)
        SS.dest_original_name = dest_cdfx_upload.name
        SS.updated_cdfx_bytes = None
        SS.updated_cdfx_name = None
    SS.dest_cdfx_uploaded = True
    st.success(f"Destination CDFX uploaded: `{dest_cdfx_upload.name}`")
elif SS.dest_path and os.path.exists(SS.dest_path):
    if SS.dest_original_name:
        SS.dest_cdfx_uploaded = True
    st.info(f"Current destination CDFX: `{SS.dest_original_name or os.path.basename(SS.dest_path)}`")
    if st.button("Remove destination CDFX", type="secondary"):
        SS.dest_path = None
        SS.dest_original_name = None
        SS.dest_cdfx_uploaded = False
        SS.updated_cdfx_bytes = None
        SS.updated_cdfx_name = None
        st.rerun()
else:
    SS.dest_cdfx_uploaded = False

# ---------------------------------------------------------------------------
# Step 3 — One-click full run (convert -> update CDFX -> patch HEX)
# ---------------------------------------------------------------------------
st.header("Step 3 · Run pipeline")
st.markdown(
    " - After step is completed, click on run pipeline to execute the mapping of source and destination labels \n"
    " - The result of step 3 gives the statistical view post the label mapping. \n"
    " - It generates updated HEX + A2L, and updated CDFX only when destination CDFX is provided."
    
)

if src_pairs:
    st.info(f"Source pairs to convert during run: {len(src_pairs)}")
if src_cdfxs:
    st.info(f"Direct source CDFX files: {len(src_cdfxs)}")

if SS.dest_a2l_path and SS.dest_hex_path:
    SS.patch_a2l_path = SS.dest_a2l_path
    SS.patch_hex_path = SS.dest_hex_path
    SS.group_a2l_path = SS.dest_a2l_path
    st.success("Destination A2L+HEX is available for HEX patching and FUNCTION grouping.")

dest_ready = bool(SS.dest_a2l_path and SS.dest_hex_path)
dest_cdfx_ready = bool(SS.dest_path and os.path.exists(SS.dest_path))
sources_ready = bool(src_cdfxs or src_pairs)
run_disabled = not (dest_ready and sources_ready)

if not dest_cdfx_ready:
    st.info("Destination CDFX is optional. Without it, the run will still produce HEX/A2L and report outputs, but no CDFX download will be shown.")

if st.button("Run full pipeline", type="primary", disabled=run_disabled):
    SS.pipeline_result = None
    SS.converted_cdfxs = []
    SS.patched_hex_path = None
    SS.patched_hex_info = None
    SS.pipeline_cdfx_path = None
    SS.detailed_log_json_path = None
    SS.detailed_log_xlsx_path = None
    SS.log_upload_info = None
    SS.updated_cdfx_bytes = None
    SS.updated_cdfx_name = None
    SS.frm_xlsx_path = None
    SS.frm_cdfx_path = None
    SS.frm_hex_path = None
    SS.frm_log_path = None
    SS.frm_patch_info = None
    SS.frm_extract_info = None
    SS.frm_values_log_json_path = None
    SS.frm_values_upload_info = None

    start_ts = time.time()
    progress = st.progress(0.0, text="Starting full run...")
    phase = st.empty()
    buf, err_buf = StringIO(), StringIO()

    def _fmt_eta(seconds_left: float | None) -> str:
        if seconds_left is None:
            return "estimating..."
        if seconds_left < 60:
            return f"{int(seconds_left)}s"
        mins = int(seconds_left // 60)
        secs = int(seconds_left % 60)
        return f"{mins}m {secs}s"

    def _set_progress(frac: float, message: str) -> None:
        frac = max(0.0, min(1.0, frac))
        elapsed = time.time() - start_ts
        eta = (elapsed * (1.0 - frac) / frac) if frac > 0 else None
        progress.progress(frac, text=f"{message} | ETA: {_fmt_eta(eta)}")
        phase.caption(f"Elapsed: {int(elapsed)}s")

    try:
        total_units = 1 + len(src_pairs)
        completed_units = 0

        _set_progress(0.0, "Validating inputs")
        if not sources_ready:
            raise ValueError("No usable source files found.")
        if not dest_ready:
            raise ValueError("No usable destination A2L+HEX found.")
        completed_units += 1
        _set_progress(completed_units / max(total_units, 1), "Inputs validated")

        if src_pairs:
            out_dir = os.path.join(SS.workdir, "generated_cdfx")
            os.makedirs(out_dir, exist_ok=True)
            for a2l_path, hex_path in src_pairs:
                _set_progress(completed_units / max(total_units, 1),
                              f"Converting source pair {Path(a2l_path).stem}")
                with redirect_stdout(buf), redirect_stderr(err_buf):
                    src_res = _convert_pair(a2l_path, hex_path, out_dir)
                if not src_res["ok"]:
                    raise RuntimeError(
                        f"Source conversion failed for {os.path.basename(a2l_path)}: "
                        f"{src_res['error']}"
                    )
                SS.converted_cdfxs.append(src_res["out_cdfx"])
                completed_units += 1
                _set_progress(completed_units / max(total_units, 1),
                              f"Converted source pair {Path(a2l_path).stem}")

        effective_sources = list(SS.converted_cdfxs) + list(src_cdfxs)
        if not effective_sources:
            raise RuntimeError("No effective CDFX sources after conversion.")

        cdfx_download_enabled = bool(SS.dest_cdfx_uploaded and dest_cdfx_ready)
        pipeline_dest_path = SS.dest_path
        if not pipeline_dest_path or not os.path.exists(pipeline_dest_path):
            out_dir = os.path.join(SS.workdir, "generated_destination_cdfx")
            os.makedirs(out_dir, exist_ok=True)
            _set_progress(completed_units / max(total_units, 1), "Generating temporary destination CDFX")
            with redirect_stdout(buf), redirect_stderr(err_buf):
                dest_res = _convert_pair(SS.dest_a2l_path, SS.dest_hex_path, out_dir)
            if not dest_res["ok"]:
                raise RuntimeError(
                    f"Destination conversion failed for {os.path.basename(SS.dest_a2l_path)}: "
                    f"{dest_res['error']}"
                )
            pipeline_dest_path = dest_res["out_cdfx"]

        pipeline_units = max(1, len(effective_sources))
        patch_ready = bool(SS.patch_a2l_path and SS.patch_hex_path)
        group_ready = bool(SS.group_a2l_path)
        total_units += pipeline_units + (1 if patch_ready else 0) + (1 if group_ready else 0)

        def _pipeline_progress(evt: dict) -> None:
            evt_type = evt.get("type")
            if evt_type == "row_progress":
                src_i = int(evt.get("source_index", 1))
                src_total = int(evt.get("total_sources", pipeline_units))
                rows_done = int(evt.get("processed", 0))
                rows_total = max(1, int(evt.get("total", 1)))
                local = (src_i - 1 + (rows_done / rows_total)) / max(src_total, 1)
                frac = (completed_units + local * pipeline_units) / max(total_units, 1)
                _set_progress(frac,
                              f"Running pipeline: {evt.get('source', '')} "
                              f"({rows_done}/{rows_total} labels)")
            elif evt_type == "source_start":
                src_i = int(evt.get("source_index", 1))
                local = (src_i - 1) / max(pipeline_units, 1)
                frac = (completed_units + local * pipeline_units) / max(total_units, 1)
                _set_progress(frac, f"Running pipeline source {src_i}/{pipeline_units}")

        _set_progress(completed_units / max(total_units, 1), "Starting CDFX update pipeline")
        with redirect_stdout(buf), redirect_stderr(err_buf):
            SS.pipeline_result = pipeline.run_pipeline(
                source_files=effective_sources,
                destination_file=pipeline_dest_path,
                work_dir=SS.workdir,
                progress_callback=_pipeline_progress,
            )
        completed_units += pipeline_units
        if SS.pipeline_result and "error" not in SS.pipeline_result:
            SS.pipeline_cdfx_path = pipeline_dest_path
        _set_progress(completed_units / max(total_units, 1), "CDFX update pipeline finished")
        if (cdfx_download_enabled and pipeline_dest_path and os.path.exists(pipeline_dest_path)
                and "error" not in (SS.pipeline_result or {})):
            SS.updated_cdfx_bytes = _read_cdfx_download_bytes(pipeline_dest_path)
            SS.updated_cdfx_name = SS.dest_original_name or os.path.basename(pipeline_dest_path)

        if patch_ready and SS.pipeline_result and "error" not in SS.pipeline_result:
            out_hex = os.path.join(SS.workdir, f"{Path(SS.patch_hex_path).stem}_updated.hex")
            _set_progress(completed_units / max(total_units, 1), "Patching HEX from updated CDFX")
            with redirect_stdout(buf), redirect_stderr(err_buf):
                patch_info = patch_hex_from_cdfx(
                    a2l_path=SS.patch_a2l_path,
                    hex_path=SS.patch_hex_path,
                    cdfx_path=pipeline_dest_path,
                    out_hex_path=out_hex,
                )
            SS.patched_hex_path = patch_info["out_hex"]
            SS.patched_hex_info = patch_info
            completed_units += 1
            _set_progress(completed_units / max(total_units, 1), "HEX patch complete")

        att_path = (SS.pipeline_result or {}).get("attention_xlsx")
        if group_ready and SS.pipeline_result and "error" not in SS.pipeline_result and att_path and os.path.exists(att_path):
            grouped_out = os.path.join(SS.workdir, "needs_attention_labels_grouped.xlsx")
            _set_progress(completed_units / max(total_units, 1),
                          "Grouping attention labels by FUNCTION")
            with redirect_stdout(buf), redirect_stderr(err_buf):
                group_info = pipeline.group_attention_labels_by_function(
                    xlsx_path=att_path,
                    a2l_path=SS.group_a2l_path,
                    output_path=grouped_out,
                    updated_labels=(SS.pipeline_result or {}).get("updated_labels", []),
                    copied_labels=(SS.pipeline_result or {}).get("copied_labels", []),
                )
            SS.grouped_xlsx_path = group_info["output_path"]
            completed_units += 1
            _set_progress(completed_units / max(total_units, 1),
                          "Grouping complete")
        elif att_path and not group_ready:
            st.warning("Attention labels were produced but destination A2L was missing, so grouping was skipped.")

        if SS.pipeline_result and "error" not in SS.pipeline_result:
            detail_info = write_detailed_logs(
                pipeline_result=SS.pipeline_result,
                patch_info=SS.patched_hex_info,
                out_dir=SS.workdir,
                base_name="calibration_detailed_log",
            )
            SS.detailed_log_json_path = detail_info["json_path"]
            SS.detailed_log_xlsx_path = detail_info["xlsx_path"]
            SS.log_upload_info = upload_log_json(
                log_json_path=SS.detailed_log_json_path,
                workspace_dir=SS.workdir,
            )

        _set_progress(1.0, "Full run complete")
        st.success("Pipeline completed end-to-end.")
    except Exception as e:
        st.error(f"Full run failed: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
    finally:
        with st.expander("Pipeline stdout", expanded=False):
            st.code(buf.getvalue() or "(empty)")
        if err_buf.getvalue():
            with st.expander("Pipeline stderr"):
                st.code(err_buf.getvalue())


# ---------------------------------------------------------------------------
# Step 4 — Results & downloads
# ---------------------------------------------------------------------------
st.header("Step 4 · Results & downloads")

result = SS.pipeline_result
if not result:
    st.info("Run the full pipeline to see results here.")
else:
    if "error" in result:
        st.error(result["error"])
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Labels in destination", result["all_labels"])
        col2.metric("Labels updated", result["all_updated"])
        col3.metric("Labels copied directly", result.get("all_copied", 0))
        col4.metric("Labels still needing attention", len(result["remaining"]))

        st.subheader("Per-source breakdown")
        summary_rows = []
        for row in (result.get("summary") or []):
            if isinstance(row, dict):
                remapped = dict(row)
                if "updated" in remapped:
                    remapped["Labels updated"] = remapped.pop("updated")
                if "found_but_unchanged" in remapped:
                    remapped["Labels copied directly"] = remapped.pop("found_but_unchanged")
                if "still_need_attention" in remapped:
                    remapped["Labels_still_need_attention"] = remapped.pop("still_need_attention")
                summary_rows.append(remapped)
            else:
                summary_rows.append(row)

        st.dataframe(summary_rows, hide_index=True, width="stretch")

        if SS.updated_cdfx_bytes:
            st.download_button(
                "Download updated CDFX",
                SS.updated_cdfx_bytes,
                file_name=SS.updated_cdfx_name,
                mime="application/xml",
                type="primary",
            )

        att = result.get("attention_xlsx")
        if att and os.path.exists(att):
            with open(att, "rb") as f:
                st.download_button(
                    "Download labels needing attention (.xlsx)",
                    f.read(),
                    file_name=os.path.basename(att),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        log_path = "calibration_process.log"
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                st.download_button("Download calibration_process.log",
                                   f.read(),
                                   file_name="calibration_process.log",
                                   mime="text/plain")

        if SS.detailed_log_json_path and os.path.exists(SS.detailed_log_json_path):
            with open(SS.detailed_log_json_path, "rb") as f:
                st.download_button(
                    "Download detailed process log (.json)",
                    f.read(),
                    file_name=os.path.basename(SS.detailed_log_json_path),
                    mime="application/json",
                )
        upload_info = SS.log_upload_info or {}
        if upload_info.get("status") == "uploaded":
            st.success(upload_info.get("message", "Detailed JSON log uploaded."))
            if upload_info.get("uploaded_path"):
                st.caption(f"Uploaded to: {upload_info['uploaded_path']}")
            if upload_info.get("pending_flushed"):
                st.caption(f"Pending logs flushed: {upload_info['pending_flushed']}")
        elif upload_info.get("status") in ("queued_no_onedrive", "queued_upload_failed"):
            st.warning(upload_info.get("message", "Detailed JSON log queued for later upload."))
            if upload_info.get("queued_path"):
                st.caption(f"Queued at: {upload_info['queued_path']}")

        if SS.detailed_log_xlsx_path and os.path.exists(SS.detailed_log_xlsx_path):
            with open(SS.detailed_log_xlsx_path, "rb") as f:
                st.download_button(
                    "Download detailed process log (.xlsx)",
                    f.read(),
                    file_name=os.path.basename(SS.detailed_log_xlsx_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        if result["remaining"]:
            with st.expander(f"Remaining labels ({len(result['remaining'])})"):
                st.code("\n".join(result["remaining"]))

info = SS.patched_hex_info
if info:
    st.subheader("Updated HEX details")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Labels patched", info["patched"])
    c2.metric("COM_AXIS axes patched", info["axes_patched"])
    c3.metric("Skipped (no A2L char)", info["skipped_no_char"])
    c4.metric("Errors", info["errors_total"])
    if info["warnings"]:
        with st.expander(f"Warnings (showing {len(info['warnings'])} of "
                         f"{info['warnings_total']})"):
            for w in info["warnings"]:
                st.text(w)
    if info["errors"]:
        with st.expander(f"Errors (showing {len(info['errors'])} of "
                         f"{info['errors_total']})"):
            for e in info["errors"]:
                st.text(e)

if SS.patched_hex_path and os.path.exists(SS.patched_hex_path):
    with open(SS.patched_hex_path, "rb") as f:
        st.download_button(
            "Download updated HEX",
            f.read(),
            file_name=os.path.basename(SS.patched_hex_path),
            mime="application/octet-stream",
        )
    if SS.patch_a2l_path and os.path.exists(SS.patch_a2l_path):
        with open(SS.patch_a2l_path, "rb") as f:
            st.download_button(
                "Download A2L (unchanged, paired with updated HEX)",
                f.read(),
                file_name=os.path.basename(SS.patch_a2l_path),
                mime="text/plain",
            )


# ---------------------------------------------------------------------------
# Step 5 — Group attention labels by FUNCTION (auto from destination A2L)
# ---------------------------------------------------------------------------
st.header("Step 5 · Group attention labels by FUNCTION ")
st.markdown(
    " - Grouping is executed automatically in one-click run using the destination A2L."
)

if SS.grouped_xlsx_path and os.path.exists(SS.grouped_xlsx_path):
    st.success(f"Grouped file ready: `{os.path.basename(SS.grouped_xlsx_path)}`")
    with open(SS.grouped_xlsx_path, "rb") as f:
        st.download_button(
            "Download grouped attention labels (.xlsx)",
            f.read(),
            file_name=os.path.basename(SS.grouped_xlsx_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("Grouped attention file will appear here after one-click run (when attention labels exist).")


# ---------------------------------------------------------------------------
# Step 6 — Apply FRM/software PDF start values to generated outputs
# ---------------------------------------------------------------------------
st.header("Step 6 · Apply software PDF start values (Optional)")
st.markdown(
    " - Upload the software/FRM PDF after the full pipeline has completed.\n"
    " - The tool extracts label start values, updates the latest CDFX, and patches the updated HEX again."
)

frm_pdf_upload = st.file_uploader(
    "Software / FRM PDF",
    accept_multiple_files=False,
    type=["pdf", "PDF"],
    max_upload_size=10000,
    key="frm_pdf_upload",
)
if frm_pdf_upload:
    is_new_frm_pdf = (
        not SS.frm_pdf_path
        or not os.path.exists(SS.frm_pdf_path)
        or Path(SS.frm_pdf_path).name.split("_", 1)[-1] != frm_pdf_upload.name
    )
    if is_new_frm_pdf:
        SS.frm_pdf_path = _save_upload(frm_pdf_upload, SS.workdir)
        SS.frm_xlsx_path = None
        SS.frm_cdfx_path = None
        SS.frm_hex_path = None
        SS.frm_log_path = None
        SS.frm_patch_info = None
        SS.frm_extract_info = None
        SS.frm_values_log_json_path = None
        SS.frm_values_upload_info = None
    st.success(f"Software PDF uploaded: `{frm_pdf_upload.name}`")
elif SS.frm_pdf_path and os.path.exists(SS.frm_pdf_path):
    st.info(f"Current software PDF: `{Path(SS.frm_pdf_path).name.split('_', 1)[-1]}`")

frm_ready = bool(SS.frm_pdf_path and os.path.exists(SS.frm_pdf_path))
hex_ready_for_frm = bool(SS.patched_hex_path and os.path.exists(SS.patched_hex_path))
a2l_ready_for_frm = bool(SS.patch_a2l_path and os.path.exists(SS.patch_a2l_path))
cdfx_ready_for_frm = bool(SS.pipeline_cdfx_path and os.path.exists(SS.pipeline_cdfx_path))

if not hex_ready_for_frm or not a2l_ready_for_frm:
    st.info("Run the full pipeline first so the updated HEX and destination A2L are available.")
if not cdfx_ready_for_frm:
    st.info("No pipeline CDFX path is available yet; run the full pipeline first.")

frm_disabled = not (frm_ready and hex_ready_for_frm and a2l_ready_for_frm and cdfx_ready_for_frm)
if st.button("Extract PDF and apply start values", type="primary", disabled=frm_disabled):
    frm_progress = st.progress(0.0, text="Starting software PDF extraction...")
    frm_stdout, frm_stderr = StringIO(), StringIO()
    try:
        pdf_path = Path(SS.frm_pdf_path)
        base_hex_path = Path(SS.patched_hex_path)
        base_cdfx_path = Path(SS.pipeline_cdfx_path)
        a2l_path = Path(SS.patch_a2l_path)

        frm_xlsx = Path(SS.workdir) / "software_pdf_start_values.xlsx"
        frm_cdfx = Path(SS.workdir) / "updated_with_software_pdf.CDFX"
        frm_hex = Path(SS.workdir) / f"{base_hex_path.stem}_software_pdf.hex"
        frm_log = Path(SS.workdir) / "software_pdf_apply_log.xlsx"
        frm_values_json = Path(SS.workdir) / "software_pdf_start_values_log.json"

        frm_progress.progress(0.15, text="Extracting start values from PDF...")
        with redirect_stdout(frm_stdout), redirect_stderr(frm_stderr):
            extract_result = extract_pages(
                pdf_path,
                engine="pdfplumber",
                ocr_mode="auto",
                ocr_dpi=300,
            )
            # extract_frm v5 returns 5 fields; keep compatibility with older signatures.
            if isinstance(extract_result, tuple) and len(extract_result) >= 5:
                pages, ocr_flags, page_numbers, _total_pages, _note = extract_result[:5]
            elif isinstance(extract_result, tuple) and len(extract_result) >= 4:
                pages, ocr_flags, page_numbers, _total_pages = extract_result[:4]
            else:
                pages, ocr_flags = extract_result[:2]
                page_numbers = None

            records = parse_records(pages, ocr_flags, page_numbers)
            extracted_count = write_excel(records, frm_xlsx)

        if extracted_count == 0:
            raise RuntimeError("No start values were extracted from the software PDF.")

        frm_progress.progress(0.45, text="Updating CDFX with extracted start values...")
        shutil.copyfile(base_cdfx_path, frm_cdfx)
        frm_values = read_frm_values(frm_xlsx)
        logs = apply_values_to_cdfx(frm_values, frm_cdfx)

        frm_progress.progress(0.70, text="Patching updated HEX from software PDF values...")
        with redirect_stdout(frm_stdout), redirect_stderr(frm_stderr):
            patch_info = patch_hex_from_cdfx(
                a2l_path=a2l_path,
                hex_path=base_hex_path,
                cdfx_path=frm_cdfx,
                out_hex_path=frm_hex,
            )

        frm_progress.progress(0.90, text="Writing software PDF apply log...")
        write_log(logs, patch_info, frm_log)

        SS.frm_xlsx_path = str(frm_xlsx)
        SS.frm_cdfx_path = str(frm_cdfx)
        SS.frm_hex_path = patch_info["out_hex"]
        SS.frm_log_path = str(frm_log)
        SS.frm_patch_info = patch_info
        SS.frm_extract_info = {
            "pdf_pages": len(pages),
            "labels_extracted": extracted_count,
            "ocr_rows": sum(1 for r in records if r.ocr),
            "reconstructed_rows": sum(1 for r in records if r.reconstructed),
            "cdfx_updated": sum(1 for row in logs if row.status == "updated"),
            "cdfx_unchanged": sum(1 for row in logs if row.status == "unchanged"),
            "missing_in_cdfx": sum(1 for row in logs if row.status == "missing_in_cdfx"),
            "skipped": sum(1 for row in logs if row.status == "skipped"),
        }

        _write_step6_start_values_log(
            out_path=frm_values_json,
            frm_values=frm_values,
            apply_logs=logs,
            patch_info=patch_info,
            extract_info=SS.frm_extract_info,
        )
        SS.frm_values_log_json_path = str(frm_values_json)
        SS.frm_values_upload_info = upload_log_json(
            log_json_path=frm_values_json,
            workspace_dir=SS.workdir,
            upload_subpath=Path("CalAiLogs") / "Incoming_start_values",
        )

        frm_progress.progress(1.0, text="Software PDF values applied.")
        st.success("Software PDF start values were applied to the generated outputs.")
    except Exception as e:
        st.error(f"Software PDF step failed: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
    finally:
        with st.expander("Software PDF step stdout", expanded=False):
            st.code(frm_stdout.getvalue() or "(empty)")
        if frm_stderr.getvalue():
            with st.expander("Software PDF step stderr"):
                st.code(frm_stderr.getvalue())

if SS.frm_extract_info:
    st.subheader("Software PDF results")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Labels extracted", SS.frm_extract_info["labels_extracted"])
    f2.metric("CDFX labels updated", SS.frm_extract_info["cdfx_updated"])
    f3.metric("Missing in CDFX", SS.frm_extract_info["missing_in_cdfx"])
    f4.metric("HEX labels patched", (SS.frm_patch_info or {}).get("patched", 0))

    if SS.frm_extract_info.get("ocr_rows"):
        st.warning(
            f"{SS.frm_extract_info['ocr_rows']} extracted row(s) came from OCR. "
            "Please verify those label names in the extracted Excel."
        )

if SS.frm_xlsx_path and os.path.exists(SS.frm_xlsx_path):
    with open(SS.frm_xlsx_path, "rb") as f:
        st.download_button(
            "Download extracted software PDF start values (.xlsx)",
            f.read(),
            file_name=os.path.basename(SS.frm_xlsx_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

if SS.frm_log_path and os.path.exists(SS.frm_log_path):
    with open(SS.frm_log_path, "rb") as f:
        st.download_button(
            "Download software PDF apply log (.xlsx)",
            f.read(),
            file_name=os.path.basename(SS.frm_log_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

if SS.frm_values_log_json_path and os.path.exists(SS.frm_values_log_json_path):
    with open(SS.frm_values_log_json_path, "rb") as f:
        st.download_button(
            "Download software PDF labels + values log (.json)",
            f.read(),
            file_name=os.path.basename(SS.frm_values_log_json_path),
            mime="application/json",
        )

frm_upload = SS.frm_values_upload_info or {}
if frm_upload.get("status") == "uploaded":
    st.success(frm_upload.get("message", "Step-6 start-values log uploaded."))
    if frm_upload.get("uploaded_path"):
        st.caption(f"Uploaded to: {frm_upload['uploaded_path']}")
    if frm_upload.get("pending_flushed"):
        st.caption(f"Pending logs flushed: {frm_upload['pending_flushed']}")
elif frm_upload.get("status") in ("queued_no_onedrive", "queued_upload_failed"):
    st.warning(frm_upload.get("message", "Step-6 start-values log queued for later upload."))
    if frm_upload.get("queued_path"):
        st.caption(f"Queued at: {frm_upload['queued_path']}")

if SS.frm_cdfx_path and os.path.exists(SS.frm_cdfx_path):
    st.download_button(
        "Download CDFX with software PDF start values",
        _read_cdfx_download_bytes(SS.frm_cdfx_path),
        file_name=os.path.basename(SS.frm_cdfx_path),
        mime="application/xml",
    )

if SS.frm_hex_path and os.path.exists(SS.frm_hex_path):
    with open(SS.frm_hex_path, "rb") as f:
        st.download_button(
            "Download HEX with software PDF start values",
            f.read(),
            file_name=os.path.basename(SS.frm_hex_path),
            mime="application/octet-stream",
            type="primary",
        )
