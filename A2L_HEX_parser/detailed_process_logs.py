from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


_HEADER_DICTIONARY_ROWS = [
    {"sheet": "Summary", "header": "generated_utc", "meaning": "UTC timestamp when this detailed log file was generated."},
    {"sheet": "Summary", "header": "destination_labels", "meaning": "Total labels present in destination CDFX considered by pipeline."},
    {"sheet": "Summary", "header": "cdfx_updated", "meaning": "Count of destination labels whose values were changed from source(s)."},
    {"sheet": "Summary", "header": "cdfx_copied_unchanged", "meaning": "Count of labels found in source(s) but already equal in destination, so unchanged."},
    {"sheet": "Summary", "header": "attention_labels", "meaning": "Count of labels still unresolved after all sources."},
    {"sheet": "Summary", "header": "hex_patched", "meaning": "Number of labels successfully written into HEX."},
    {"sheet": "Summary", "header": "hex_axes_patched", "meaning": "Number of COM_AXIS axis blocks written into HEX."},
    {"sheet": "Summary", "header": "hex_skipped_no_char", "meaning": "Labels skipped in HEX patch because no A2L CHARACTERISTIC matched."},
    {"sheet": "Summary", "header": "hex_skipped_no_char_axis_pts", "meaning": "Of skipped_no_char, labels that exist as AXIS_PTS in A2L."},
    {"sheet": "Summary", "header": "hex_skipped_no_char_unknown", "meaning": "Of skipped_no_char, labels not found as relevant entries in A2L."},
    {"sheet": "Summary", "header": "hex_skipped_no_layout", "meaning": "Labels skipped because required RECORD_LAYOUT was missing."},
    {"sheet": "Summary", "header": "hex_errors_total", "meaning": "Total HEX patch errors."},
    {"sheet": "Summary", "header": "hex_warnings_total", "meaning": "Total HEX patch warnings."},
    {"sheet": "Summary", "header": "pipeline_events_total", "meaning": "Number of label-level events produced by CDFX update pipeline."},
    {"sheet": "Summary", "header": "patch_events_total", "meaning": "Number of label-level events produced by HEX patch stage."},
    {"sheet": "Summary", "header": "events_total", "meaning": "Total label-level events across pipeline and patch stages."},
    {"sheet": "Per Source Summary", "header": "source", "meaning": "Source CDFX filename processed in that pass."},
    {"sheet": "Per Source Summary", "header": "updated", "meaning": "Labels changed in destination using that source."},
    {"sheet": "Per Source Summary", "header": "found_but_unchanged", "meaning": "Labels found in source but values already matched destination."},
    {"sheet": "Per Source Summary", "header": "still_need_attention", "meaning": "Labels still unresolved after this source pass."},
    {"sheet": "Per Source Summary", "header": "remaining", "meaning": "Count of unresolved labels after this source pass."},
    {"sheet": "Label Events", "header": "timestamp_utc", "meaning": "UTC time when this event was recorded."},
    {"sheet": "Label Events", "header": "source", "meaning": "Source name or stage owner (for example source file, pipeline, hex_patch)."},
    {"sheet": "Label Events", "header": "label", "meaning": "Label name related to the event."},
    {"sheet": "Label Events", "header": "action", "meaning": "Event type such as updated, copied, skipped, padded, attention, patched."},
    {"sheet": "Label Events", "header": "stage", "meaning": "Processing stage where event occurred (cdfx_update, pipeline_result, hex_patch)."},
    {"sheet": "Label Events", "header": "reason", "meaning": "Human-readable reason for the action."},
    {"sheet": "Label Events", "header": "details", "meaning": "Extra structured context (for example row lengths or size details)."},
    {"sheet": "Remaining Labels", "header": "label", "meaning": "Label still needing manual attention after all processing."},
    {"sheet": "Patched Labels", "header": "label", "meaning": "Label successfully patched into HEX."},
    {"sheet": "Hex Patched Labels", "header": "label", "meaning": "Consolidated list of all labels successfully patched into HEX."},
    {"sheet": "Skipped Labels", "header": "label", "meaning": "Consolidated list of all labels skipped during HEX patch stage."},
    {"sheet": "Skipped Labels", "header": "skip_reason", "meaning": "Reason category for why the HEX patch skipped this label."},
    {"sheet": "Skipped No Char", "header": "label", "meaning": "Label skipped because no matching A2L CHARACTERISTIC was found."},
    {"sheet": "Skipped No Char AxisPts", "header": "label", "meaning": "Subset of skipped_no_char where label exists as AXIS_PTS in A2L."},
    {"sheet": "Skipped No Char Unknown", "header": "label", "meaning": "Subset of skipped_no_char where label was not found as relevant entry in A2L."},
    {"sheet": "Skipped No Layout", "header": "label", "meaning": "Label skipped because required RECORD_LAYOUT was missing."},
    {"sheet": "Patch Warnings", "header": "warning", "meaning": "Non-fatal warning message from HEX patch stage."},
    {"sheet": "Patch Errors", "header": "error", "meaning": "Error message from HEX patch stage."},
]


def _as_df(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def _build_patch_events(patch_info: dict[str, Any] | None) -> list[dict[str, Any]]:
    patch_info = patch_info or {}
    events: list[dict[str, Any]] = []

    for label in patch_info.get("patched_labels", []) or []:
        events.append({
            "source": "hex_patch",
            "label": label,
            "action": "patched",
            "stage": "hex_patch",
            "reason": "patched from CDFX",
        })

    axis_pts_set = set(patch_info.get("skipped_no_char_axis_pts_labels", []) or [])
    unknown_set = set(patch_info.get("skipped_no_char_unknown_labels", []) or [])

    for label in patch_info.get("skipped_no_char_labels", []) or []:
        if label in axis_pts_set:
            reason = "no A2L characteristic (label exists as AXIS_PTS)"
        elif label in unknown_set:
            reason = "no A2L characteristic (label not found in A2L)"
        else:
            reason = "no A2L characteristic"
        events.append({
            "source": "hex_patch",
            "label": label,
            "action": "skipped",
            "stage": "hex_patch",
            "reason": reason,
        })

    for label in patch_info.get("skipped_no_layout_labels", []) or []:
        events.append({
            "source": "hex_patch",
            "label": label,
            "action": "skipped",
            "stage": "hex_patch",
            "reason": "missing record layout",
        })

    return events


def write_detailed_logs(
    pipeline_result: dict[str, Any] | None,
    patch_info: dict[str, Any] | None,
    out_dir: str | Path,
    base_name: str = "calibration_detailed_log",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_result = pipeline_result or {}
    patch_info = patch_info or {}

    pipeline_events = list(pipeline_result.get("detailed_events", []) or [])
    patch_events = _build_patch_events(patch_info)
    events_all = pipeline_events + patch_events

    summary_rows = [{
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "destination_labels": pipeline_result.get("all_labels", 0),
        "cdfx_updated": pipeline_result.get("all_updated", 0),
        "cdfx_copied_unchanged": pipeline_result.get("all_copied", 0),
        "attention_labels": len(pipeline_result.get("remaining", []) or []),
        "hex_patched": patch_info.get("patched", 0),
        "hex_axes_patched": patch_info.get("axes_patched", 0),
        "hex_skipped_no_char": patch_info.get("skipped_no_char", 0),
        "hex_skipped_no_char_axis_pts": patch_info.get("skipped_no_char_axis_pts", 0),
        "hex_skipped_no_char_unknown": patch_info.get("skipped_no_char_unknown", 0),
        "hex_skipped_no_layout": patch_info.get("skipped_no_layout", 0),
        "hex_errors_total": patch_info.get("errors_total", 0),
        "hex_warnings_total": patch_info.get("warnings_total", 0),
        "pipeline_events_total": len(pipeline_events),
        "patch_events_total": len(patch_events),
        "events_total": len(events_all),
    }]

    remaining_rows = [{"label": x} for x in (pipeline_result.get("remaining", []) or [])]
    patched_rows = [{"label": x} for x in (patch_info.get("patched_labels", []) or [])]
    consolidated_patched_rows = [{"label": x} for x in (patch_info.get("patched_labels", []) or [])]
    skipped_rows = (
        [{"label": x, "skip_reason": "no_a2l_characteristic"} for x in (patch_info.get("skipped_no_char_labels", []) or [])]
        + [{"label": x, "skip_reason": "missing_record_layout"} for x in (patch_info.get("skipped_no_layout_labels", []) or [])]
    )
    skipped_no_char_rows = [{"label": x} for x in (patch_info.get("skipped_no_char_labels", []) or [])]
    skipped_no_char_axis_pts_rows = [{"label": x} for x in (patch_info.get("skipped_no_char_axis_pts_labels", []) or [])]
    skipped_no_char_unknown_rows = [{"label": x} for x in (patch_info.get("skipped_no_char_unknown_labels", []) or [])]
    skipped_no_layout_rows = [{"label": x} for x in (patch_info.get("skipped_no_layout_labels", []) or [])]

    errors_all = patch_info.get("errors_all", patch_info.get("errors", [])) or []
    warnings_all = patch_info.get("warnings_all", patch_info.get("warnings", [])) or []
    error_rows = [{"error": e} for e in errors_all]
    warning_rows = [{"warning": w} for w in warnings_all]

    payload = {
        "summary": summary_rows[0],
        "pipeline_summary": pipeline_result.get("summary", []),
        "event_counts": pipeline_result.get("detailed_event_counts", {}),
        "events": events_all,
        "remaining_labels": pipeline_result.get("remaining", []) or [],
        "hex_patched_labels": patch_info.get("patched_labels", []) or [],
        "hex_skipped_labels": skipped_rows,
        "patched_labels": patch_info.get("patched_labels", []) or [],
        "skipped_no_char_labels": patch_info.get("skipped_no_char_labels", []) or [],
        "skipped_no_char_axis_pts_labels": patch_info.get("skipped_no_char_axis_pts_labels", []) or [],
        "skipped_no_char_unknown_labels": patch_info.get("skipped_no_char_unknown_labels", []) or [],
        "skipped_no_layout_labels": patch_info.get("skipped_no_layout_labels", []) or [],
        "patch_errors": errors_all,
        "patch_warnings": warnings_all,
    }

    json_path = out_dir / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    xlsx_path = out_dir / f"{base_name}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        _as_df(summary_rows, list(summary_rows[0].keys())).to_excel(
            writer, index=False, sheet_name="Summary"
        )
        _as_df(pipeline_result.get("summary", []) or [], [
            "source", "updated", "found_but_unchanged", "still_need_attention", "remaining"
        ]).to_excel(writer, index=False, sheet_name="Per Source Summary")
        _as_df(events_all, ["timestamp_utc", "source", "label", "action", "stage", "reason", "details"]).to_excel(
            writer, index=False, sheet_name="Label Events"
        )
        _as_df(remaining_rows, ["label"]).to_excel(writer, index=False, sheet_name="Remaining Labels")
        _as_df(patched_rows, ["label"]).to_excel(writer, index=False, sheet_name="Patched Labels")
        _as_df(consolidated_patched_rows, ["label"]).to_excel(
            writer, index=False, sheet_name="Hex Patched Labels"
        )
        _as_df(skipped_rows, ["label", "skip_reason"]).to_excel(
            writer, index=False, sheet_name="Skipped Labels"
        )
        _as_df(skipped_no_char_rows, ["label"]).to_excel(
            writer, index=False, sheet_name="Skipped No Char"
        )
        _as_df(skipped_no_char_axis_pts_rows, ["label"]).to_excel(
            writer, index=False, sheet_name="Skipped No Char AxisPts"
        )
        _as_df(skipped_no_char_unknown_rows, ["label"]).to_excel(
            writer, index=False, sheet_name="Skipped No Char Unknown"
        )
        _as_df(skipped_no_layout_rows, ["label"]).to_excel(
            writer, index=False, sheet_name="Skipped No Layout"
        )
        _as_df(warning_rows, ["warning"]).to_excel(writer, index=False, sheet_name="Patch Warnings")
        _as_df(error_rows, ["error"]).to_excel(writer, index=False, sheet_name="Patch Errors")
        _as_df(_HEADER_DICTIONARY_ROWS, ["sheet", "header", "meaning"]).to_excel(
            writer, index=False, sheet_name="Header Dictionary"
        )

    return {
        "json_path": str(json_path),
        "xlsx_path": str(xlsx_path),
        "events_total": len(events_all),
    }