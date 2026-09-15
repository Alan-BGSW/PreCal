"""
pipeline.py
===========
CDFX -> CDFX update pipeline. Sources and destination are both CDFX files;
A2L+HEX inputs are converted to CDFX first (see a2l_hex_to_cdfx.py) so every
path goes through the same comparison logic and updated/unchanged ratios stay
consistent regardless of input origin.
"""

import re
import lxml.etree as ET
import pandas as pd
import shutil
import os
import logging
import time
from collections import Counter
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

# Tolerance for treating two numeric text values as semantically equal.
# CDFX written by INCA truncates display precision (e.g. "33.3008") while a
# value reconstructed from A2L+HEX carries full IEEE-754 precision
# (e.g. "33.30078125"). A relative tolerance of 1e-4 (~0.01 %) absorbs that
# display rounding without masking real calibration changes; the absolute
# floor handles values close to zero.
_NUM_REL_TOL = 1e-4
_NUM_ABS_TOL = 1e-6


logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s',
)
file_logger = logging.getLogger('file_logger')
file_logger.setLevel(logging.INFO)
if not file_logger.handlers:
    fh = logging.FileHandler('calibration_process.log', mode='w')
    fh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    file_logger.addHandler(fh)


def _append_detail(detail_logs, source, label, action, stage, reason="", details=None):
    if detail_logs is None:
        return
    event = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": source or "",
        "label": label or "",
        "action": action,
        "stage": stage,
        "reason": reason,
    }
    if details is not None:
        event["details"] = details
    detail_logs.append(event)


def _ensure_monotonic(src_x, src_v, label_for_log=""):
    n = len(src_x)
    if n <= 1:
        return list(src_x), list(src_v)
    strictly_increasing = all(src_x[i] < src_x[i + 1] for i in range(n - 1))
    if strictly_increasing:
        return list(src_x), list(src_v)
    file_logger.warning(
        f"  {label_for_log}: src x-axis not strictly increasing — sorting and de-duplicating before interpolation."
    )
    pairs = sorted(zip(src_x, src_v), key=lambda p: p[0])
    out_x, out_v = [], []
    for x, v in pairs:
        if out_x and x == out_x[-1]:
            continue
        out_x.append(x)
        out_v.append(v)
    return out_x, out_v


def interp_extrap_values(src_x, src_v, dest_x, label_for_log=""):
    n = len(src_x)
    m = len(dest_x)
    if n == 0:
        file_logger.warning(f"  {label_for_log}: empty source — returning zeros.")
        return [0.0] * m
    if n == 1:
        return [src_v[0]] * m
    src_x, src_v = _ensure_monotonic(src_x, src_v, label_for_log)
    n = len(src_x)
    if n == 1:
        return [src_v[0]] * m
    slope_left = (src_v[1] - src_v[0]) / (src_x[1] - src_x[0]) if src_x[1] != src_x[0] else 0.0
    slope_right = (src_v[-1] - src_v[-2]) / (src_x[-1] - src_x[-2]) if src_x[-1] != src_x[-2] else 0.0
    result = []
    for dx in dest_x:
        if dx <= src_x[0]:
            val = src_v[0] + slope_left * (dx - src_x[0])
        elif dx >= src_x[-1]:
            val = src_v[-1] + slope_right * (dx - src_x[-1])
        else:
            i = 0
            for k in range(n - 1):
                if src_x[k] <= dx < src_x[k + 1]:
                    i = k
                    break
            span = src_x[i + 1] - src_x[i]
            t = (dx - src_x[i]) / span if span != 0 else 0.0
            val = src_v[i] + t * (src_v[i + 1] - src_v[i])
        result.append(val)
    return result


def _parse_floats(elements):
    out = []
    for el in elements:
        if el.text and el.text.strip():
            try:
                out.append(float(el.text.strip()))
            except ValueError:
                pass
    return out


def _parse_strings(elements):
    return [el.text.strip() for el in elements if el.text and el.text.strip()]


def _safe_float_list(lst):
    out = []
    for v in lst:
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            pass
    return out


def _try_float_list(strings):
    out = []
    for v in strings:
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            return [], False
    return out, True


def _axis_values(axis_entry):
    if isinstance(axis_entry, list):
        return axis_entry
    if isinstance(axis_entry, dict):
        return axis_entry.get('values', [])
    return []


def _nearest_neighbour_index_map(src_n, dest_n):
    if src_n == 0 or dest_n == 0:
        return []
    if src_n == 1:
        return [0] * dest_n
    out = []
    for i in range(dest_n):
        src_idx = round(i * (src_n - 1) / max(dest_n - 1, 1))
        out.append(max(0, min(src_n - 1, src_idx)))
    return out


def _nearest_neighbour_label_map(src_labels, dest_labels):
    src_n = len(src_labels)
    dest_n = len(dest_labels)
    if src_n == 0 or dest_n == 0:
        return []
    src_index = {}
    for i, s in enumerate(src_labels):
        if s not in src_index:
            src_index[s] = i
    nn_fallback = _nearest_neighbour_index_map(src_n, dest_n)
    out = []
    for i, d in enumerate(dest_labels):
        if d in src_index:
            out.append(src_index[d])
        else:
            out.append(nn_fallback[i])
    return out


def _format_num(v):
    return f"{v}"


def _maybe_float(s):
    """Return float(s) or None — used to compare CDFX text values numerically."""
    if s is None:
        return None
    try:
        return float(s.strip()) if isinstance(s, str) else float(s)
    except (ValueError, TypeError):
        return None


def _texts_equal(a, b):
    """True if both texts represent the same value (numerically when possible)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    sa = a.strip() if isinstance(a, str) else str(a)
    sb = b.strip() if isinstance(b, str) else str(b)
    if sa == sb:
        return True
    fa = _maybe_float(sa)
    fb = _maybe_float(sb)
    if fa is None or fb is None:
        return False
    diff = abs(fa - fb)
    return diff <= _NUM_ABS_TOL or diff <= _NUM_REL_TOL * max(abs(fa), abs(fb))


def _text_lists_equal(a_list, b_list):
    if len(a_list) != len(b_list):
        return False
    return all(_texts_equal(x, y) for x, y in zip(a_list, b_list))


def _snapshot_map(vgs, x_elems, y_elems, namespace):
    """Capture all texts that an _update_map_2d_cdfx call may rewrite."""
    rows = []
    for vg in vgs:
        lbl_el = vg.find("LABEL", namespaces=namespace)
        lbl_text = lbl_el.text if (lbl_el is not None) else None
        v_texts = [(e.text or "") for e in vg.findall("V", namespaces=namespace)]
        rows.append((lbl_text, v_texts))
    return {
        "x": [(e.text or "") for e in (x_elems or [])],
        "y": [(e.text or "") for e in (y_elems or [])],
        "rows": rows,
    }


def _snapshots_equal_numeric(a, b):
    if not _text_lists_equal(a["x"], b["x"]):
        return False
    if not _text_lists_equal(a["y"], b["y"]):
        return False
    if len(a["rows"]) != len(b["rows"]):
        return False
    for (la, va), (lb, vb) in zip(a["rows"], b["rows"]):
        if not _texts_equal(la, lb):
            return False
        if not _text_lists_equal(va, vb):
            return False
    return True


def filter_destination_file(original_dest, filtered_dest, labels_to_keep, namespace):
    tree = ET.parse(original_dest)
    root = tree.getroot()
    for sw in root.findall('.//SW-INSTANCE', namespaces=namespace):
        sn = sw.find('SHORT-NAME', namespaces=namespace)
        if sn is None or sn.text.strip() not in labels_to_keep:
            parent = sw.getparent()
            if parent is not None:
                parent.remove(sw)
    tree.write(filtered_dest, encoding='utf-8', xml_declaration=True)


def merge_updated_instances(main_dest, updated_dest, labels_to_merge, namespace):
    main_tree = ET.parse(main_dest)
    main_root = main_tree.getroot()
    updated_tree = ET.parse(updated_dest)
    updated_root = updated_tree.getroot()
    main_sw_map = {}
    for sw in main_root.findall('.//SW-INSTANCE', namespaces=namespace):
        sn = sw.find('SHORT-NAME', namespaces=namespace)
        if sn is not None and sn.text.strip() in labels_to_merge:
            main_sw_map[sn.text.strip()] = sw
    for sw in updated_root.findall('.//SW-INSTANCE', namespaces=namespace):
        sn = sw.find('SHORT-NAME', namespaces=namespace)
        if sn is not None and sn.text.strip() in labels_to_merge:
            old_sw = main_sw_map.get(sn.text.strip())
            if old_sw is not None:
                parent = old_sw.getparent()
                if parent is not None:
                    parent.replace(old_sw, sw)
    main_tree.write(main_dest, encoding='utf-8', xml_declaration=True)


def extract_labels_with_c(cdfx_file, namespace):
    try:
        tree = ET.parse(cdfx_file)
        root = tree.getroot()
    except ET.XMLSyntaxError as e:
        logging.error(f"XML Syntax Error while parsing '{cdfx_file}': {e}")
        return pd.DataFrame()
    except FileNotFoundError:
        logging.error(f"File not found: '{cdfx_file}'")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error while parsing '{cdfx_file}': {e}")
        return pd.DataFrame()

    data = []
    for sw in root.findall(".//SW-INSTANCE", namespaces=namespace):
        sn = sw.find("SHORT-NAME", namespaces=namespace)
        if sn is None or not sn.text:
            continue
        label = sn.text.strip()

        if label.endswith(('_C', '_CW', '_c')):
            v = sw.find("SW-VALUE-CONT/SW-VALUES-PHYS/V", namespaces=namespace)
            if v is not None and v.text and v.text.strip():
                data.append({'label': label, 'values': v.text.strip()})
            else:
                vt = sw.find("SW-VALUE-CONT/SW-VALUES-PHYS/VT", namespaces=namespace)
                if vt is not None and vt.text and vt.text.strip():
                    data.append({'label': label, 'values': vt.text.strip()})

        elif label.endswith('_CA'):
            dim = sw.find("SW-VALUE-CONT/SW-ARRAYSIZE/V", namespaces=namespace)
            if dim is not None and dim.text and dim.text.strip():
                vt_elements = sw.findall(".//SW-VALUE-CONT/SW-VALUES-PHYS/VT", namespaces=namespace)
                value = [vt.text.strip() for vt in vt_elements if vt.text and vt.text.strip()]
                if not value:
                    v_elements = sw.findall(".//SW-VALUE-CONT/SW-VALUES-PHYS/V", namespaces=namespace)
                    value = [v.text.strip() for v in v_elements if v.text and v.text.strip()]
                data.append({'label': label, 'dimen': dim.text, 'values': value})

        elif label.endswith(('_MAP', '_M')):
            sw_values = sw.find("SW-VALUE-CONT/SW-VALUES-PHYS", namespaces=namespace)
            if sw_values is not None:
                value_dict = {}
                for vg in sw_values.findall("VG", namespaces=namespace):
                    label_elem = vg.find("LABEL", namespaces=namespace)
                    if label_elem is not None and label_elem.text and label_elem.text.strip():
                        key = label_elem.text.strip()
                        try:
                            key = float(key)
                        except ValueError:
                            pass
                        values = _parse_floats(vg.findall("V", namespaces=namespace))
                        value_dict[key] = values
                axis = sw.findall(".//SW-AXIS-CONT", namespaces=namespace)
                x_vals, y_vals = [], []
                if axis:
                    sw_vp = axis[0].find("SW-VALUES-PHYS", namespaces=namespace)
                    if sw_vp is not None:
                        x_vals = _parse_floats(sw_vp.findall("V", namespaces=namespace))
                    try:
                        y_vals = sorted(value_dict.keys(),
                                        key=lambda x: (float(x) if isinstance(x, float) else x))
                    except TypeError:
                        y_vals = list(value_dict.keys())
                data.append({'label': label, 'values': value_dict,
                             'x_dim': len(x_vals), 'y_dim': len(y_vals),
                             'x_dim_val': x_vals, 'y_dim_val': y_vals})

        elif label.endswith(('_T', '_CUR', '_Cur')):
            axis = sw.findall(".//SW-VALUES-PHYS", namespaces=namespace)
            if axis:
                z_vals = _parse_strings(axis[0].findall("V", namespaces=namespace))
                if not z_vals:
                    z_vals = _parse_strings(axis[0].findall("VT", namespaces=namespace))
                x_vals = []
                if len(axis) > 1:
                    x_vals = _parse_strings(axis[1].findall("V", namespaces=namespace))
                    if not x_vals:
                        x_vals = _parse_strings(axis[1].findall("VT", namespaces=namespace))
                data.append({'label': label, 'x_values': x_vals,
                             'z_values': z_vals, 'x_dim': len(x_vals)})

    return pd.DataFrame(data)


def build_label_index(root, namespace):
    label_index = {}
    for sw in root.findall(".//SW-INSTANCE", namespaces=namespace):
        sn = sw.find("SHORT-NAME", namespaces=namespace)
        if sn is not None and sn.text and sn.text.strip():
            label_index[sn.text.strip()] = sw
    return label_index


def _read_axis_strings_and_numbers(svp_elem, namespace):
    if svp_elem is None:
        return [], None, False, []
    v_elems = svp_elem.findall("V", namespaces=namespace)
    vt_elems = svp_elem.findall("VT", namespaces=namespace)
    if v_elems:
        elems = v_elems
        strings = [e.text.strip() if e.text else "" for e in elems]
        nums, ok = _try_float_list(strings)
        return strings, (nums if ok else None), ok, elems
    elif vt_elems:
        elems = vt_elems
        strings = [e.text.strip() if e.text else "" for e in elems]
        return strings, None, False, elems
    return [], None, False, []


def _resample_axis(src_axis_vals, dest_elems, label, axis_name):
    dest_n = len(dest_elems)
    src_n = len(src_axis_vals)
    if dest_n == 0:
        return []
    if src_n == 0:
        return _parse_floats(dest_elems)
    if src_n == 1:
        for elem in dest_elems:
            elem.text = _format_num(src_axis_vals[0])
        return [src_axis_vals[0]] * dest_n
    src_x_norm = [i * (dest_n - 1) / (src_n - 1) for i in range(src_n)]
    dest_pos = list(range(dest_n))
    new_vals = interp_extrap_values(src_x_norm, src_axis_vals, dest_pos, f"{label} ({axis_name})")
    for i, elem in enumerate(dest_elems):
        elem.text = _format_num(new_vals[i])
    return new_vals


def _update_scalar_cdfx(label, correct_sw, incorrect_sw, row, namespace,
                        detail_logs=None, source_name=None):
    correct_value = row['values_correct']
    incorrect_value = row.get('values_incorrect', None)

    sw_unit_incorrect = incorrect_sw.find(".//SW-VALUE-CONT/UNIT-DISPLAY-NAME", namespaces=namespace)
    sw_unit_correct = correct_sw.find(".//SW-VALUE-CONT/UNIT-DISPLAY-NAME", namespaces=namespace)
    if sw_unit_incorrect is not None and sw_unit_correct is not None:
        sw_unit_incorrect.text = sw_unit_correct.text

    if _texts_equal(correct_value, incorrect_value):
        _append_detail(detail_logs, source_name, label, "copied", "cdfx_update", "no value change")
        return 'no_change', None

    sw_val_incorrect = incorrect_sw.find(".//SW-VALUE-CONT/SW-VALUES-PHYS/V", namespaces=namespace)
    sw_val_correct = correct_sw.find(".//SW-VALUE-CONT/SW-VALUES-PHYS/V", namespaces=namespace)
    if sw_val_incorrect is None:
        sw_val_incorrect = incorrect_sw.find(".//SW-VALUE-CONT/SW-VALUES-PHYS/VT", namespaces=namespace)
    if sw_val_correct is None:
        sw_val_correct = correct_sw.find(".//SW-VALUE-CONT/SW-VALUES-PHYS/VT", namespaces=namespace)
    if sw_val_incorrect is None or sw_val_correct is None:
        _append_detail(detail_logs, source_name, label, "skipped", "cdfx_update", "missing V/VT nodes")
        return 'skip', None

    sw_val_incorrect.text = f"{correct_value}"
    _append_detail(detail_logs, source_name, label, "updated", "cdfx_update", "scalar value replaced")
    return 'updated', {'label': label, 'values': correct_value, 'status': 'Updated'}


def _update_array_1d_cdfx(label, correct_sw, incorrect_sw, namespace,
                          detail_logs=None, source_name=None):
    correct_values_phys = correct_sw.find("SW-VALUE-CONT/SW-VALUES-PHYS", namespaces=namespace)
    incorrect_values_phys = incorrect_sw.find("SW-VALUE-CONT/SW-VALUES-PHYS", namespaces=namespace)
    if correct_values_phys is None or incorrect_values_phys is None:
        file_logger.warning(f"  {label}: Missing SW-VALUES-PHYS — skipping")
        _append_detail(detail_logs, source_name, label, "skipped", "cdfx_update", "missing SW-VALUES-PHYS")
        return 'skip', None

    src_str = _parse_strings(correct_values_phys.findall("VT", namespaces=namespace))
    if not src_str:
        src_str = _parse_strings(correct_values_phys.findall("V", namespaces=namespace))

    dest_elems = incorrect_values_phys.findall("VT", namespaces=namespace)
    if not dest_elems:
        dest_elems = incorrect_values_phys.findall("V", namespaces=namespace)

    src_n = len(src_str)
    dest_n = len(dest_elems)
    if src_n == 0 or dest_n == 0:
        _append_detail(detail_logs, source_name, label, "skipped", "cdfx_update", "empty source or destination array")
        return 'skip', None

    old_texts = [e.text or "" for e in dest_elems]
    src_v_float, numeric = _try_float_list(src_str)

    if numeric:
        src_x = ([i * (dest_n - 1) / (src_n - 1) for i in range(src_n)] if src_n > 1 else [0.0])
        new_vals = interp_extrap_values(src_x, src_v_float, list(range(dest_n)), label)
        new_texts = [_format_num(v) for v in new_vals]
    else:
        idx_map = _nearest_neighbour_index_map(src_n, dest_n)
        new_texts = [src_str[i] for i in idx_map]

    if _text_lists_equal(old_texts, new_texts):
        _append_detail(detail_logs, source_name, label, "copied", "cdfx_update", "no value change")
        return 'no_change', None

    for elem, text in zip(dest_elems, new_texts):
        elem.text = text

    _append_detail(
        detail_logs,
        source_name,
        label,
        "updated",
        "cdfx_update",
        "array values updated",
        details={"dest_count": dest_n, "src_count": src_n},
    )

    return 'updated', {'label': label, 'dimen': dest_n, 'values': new_texts, 'status': 'Updated'}


def _update_map_2d_cdfx(label, correct_sw, incorrect_sw, namespace,
                        detail_logs=None, source_name=None):
    sw_val_correct = correct_sw.find("SW-VALUE-CONT/SW-VALUES-PHYS", namespaces=namespace)
    sw_val_incorrect = incorrect_sw.find("SW-VALUE-CONT/SW-VALUES-PHYS", namespaces=namespace)
    if sw_val_correct is None or sw_val_incorrect is None:
        _append_detail(detail_logs, source_name, label, "skipped", "cdfx_update", "missing SW-VALUES-PHYS")
        return 'skip', None

    axis_correct = correct_sw.findall(".//SW-AXIS-CONT", namespaces=namespace)
    axis_incorrect = incorrect_sw.findall(".//SW-AXIS-CONT", namespaces=namespace)

    src_x_strs, src_x_nums, src_x_is_num = [], None, False
    dest_x_strs, dest_x_nums, dest_x_is_num = [], None, False
    dest_x_elems = []
    if axis_correct and axis_incorrect:
        svp_c = axis_correct[0].find("SW-VALUES-PHYS", namespaces=namespace)
        svp_d = axis_incorrect[0].find("SW-VALUES-PHYS", namespaces=namespace)
        src_x_strs, src_x_nums, src_x_is_num, _ = _read_axis_strings_and_numbers(svp_c, namespace)
        dest_x_strs, dest_x_nums, dest_x_is_num, dest_x_elems = _read_axis_strings_and_numbers(svp_d, namespace)

    src_y_strs, src_y_nums, src_y_is_num = [], None, False
    dest_y_strs, dest_y_nums, dest_y_is_num = [], None, False
    dest_y_elems = []
    if len(axis_correct) > 1 and len(axis_incorrect) > 1:
        svp_c1 = axis_correct[1].find("SW-VALUES-PHYS", namespaces=namespace)
        svp_d1 = axis_incorrect[1].find("SW-VALUES-PHYS", namespaces=namespace)
        src_y_strs, src_y_nums, src_y_is_num, _ = _read_axis_strings_and_numbers(svp_c1, namespace)
        dest_y_strs, dest_y_nums, dest_y_is_num, dest_y_elems = _read_axis_strings_and_numbers(svp_d1, namespace)

    x_axis_numeric = src_x_is_num and dest_x_is_num
    y_axis_numeric = src_y_is_num and dest_y_is_num

    correct_vgs = sw_val_correct.findall("VG", namespaces=namespace)
    incorrect_vgs = sw_val_incorrect.findall("VG", namespaces=namespace)
    src_row_count = len(correct_vgs)
    dest_row_count = len(incorrect_vgs)
    if src_row_count == 0 or dest_row_count == 0:
        _append_detail(detail_logs, source_name, label, "skipped", "cdfx_update", "empty source or destination map rows")
        return 'skip', None

    snapshot_old = _snapshot_map(incorrect_vgs, dest_x_elems, dest_y_elems, namespace)

    src_row_labels = []
    src_row_data = []
    for vg in correct_vgs:
        lbl_el = vg.find("LABEL", namespaces=namespace)
        lbl_text = lbl_el.text.strip() if (lbl_el is not None and lbl_el.text) else ""
        src_row_labels.append(lbl_text)
        src_row_data.append(_parse_floats(vg.findall("V", namespaces=namespace)))

    dest_row_labels = []
    for vg in incorrect_vgs:
        lbl_el = vg.find("LABEL", namespaces=namespace)
        lbl_text = lbl_el.text.strip() if (lbl_el is not None and lbl_el.text) else ""
        dest_row_labels.append(lbl_text)

    row_lengths = [len(r) for r in src_row_data]
    max_src_cols = max(row_lengths, default=0)
    if max_src_cols == 0:
        _append_detail(detail_logs, source_name, label, "skipped", "cdfx_update", "source map has no columns")
        return 'skip', None
    if any(rl != max_src_cols for rl in row_lengths):
        file_logger.warning(
            f"  {label}: source rows are ragged (lengths={row_lengths}) — padding short rows with their last value."
        )
        _append_detail(
            detail_logs,
            source_name,
            label,
            "padded",
            "cdfx_update",
            "source map rows are ragged; padded with last value",
            details={"row_lengths": row_lengths},
        )

    x_vals_new_display = []
    if x_axis_numeric and len(src_x_nums) >= 2 and dest_x_nums:
        x_vals_new_display = _resample_axis(src_x_nums, dest_x_elems, label, "x-axis")
    elif x_axis_numeric and len(src_x_nums) == 1 and dest_x_elems:
        for elem in dest_x_elems:
            elem.text = _format_num(src_x_nums[0])
        x_vals_new_display = [src_x_nums[0]] * len(dest_x_elems)
    else:
        x_vals_new_display = dest_x_strs or _parse_floats(dest_x_elems)

    y_vals_new_display = []
    if y_axis_numeric and len(src_y_nums) >= 2 and dest_y_nums:
        y_vals_new_display = _resample_axis(src_y_nums, dest_y_elems, label, "y-axis")
    elif y_axis_numeric and len(src_y_nums) == 1 and dest_y_elems:
        for elem in dest_y_elems:
            elem.text = _format_num(src_y_nums[0])
        y_vals_new_display = [src_y_nums[0]] * len(dest_y_elems)
    else:
        y_vals_new_display = dest_y_strs or _parse_floats(dest_y_elems)

    if y_axis_numeric and src_y_nums and dest_y_nums \
            and len(src_y_nums) == src_row_count and len(dest_y_nums) == dest_row_count:
        pairs = sorted(zip(src_y_nums, src_row_data), key=lambda p: p[0])
        src_y_sorted = [p[0] for p in pairs]
        src_rows_sorted = [p[1] for p in pairs]
        row_strategy = ('numeric', src_y_sorted, dest_y_nums, src_rows_sorted)
    elif src_y_strs and dest_row_labels and len(src_y_strs) == src_row_count:
        idx_map = _nearest_neighbour_label_map(src_y_strs, dest_row_labels)
        row_strategy = ('enum_label', idx_map, src_row_data)
    else:
        src_row_norm = ([i * (dest_row_count - 1) / (src_row_count - 1) for i in range(src_row_count)]
                        if src_row_count >= 2 else [0.0])
        row_strategy = ('index', src_row_norm, src_row_data)

    if x_axis_numeric and src_x_nums and dest_x_nums and len(src_x_nums) == max_src_cols:
        src_x_sorted_pairs = sorted(enumerate(src_x_nums), key=lambda p: p[1])
        src_x_sorted = [p[1] for p in src_x_sorted_pairs]
        col_perm = [p[0] for p in src_x_sorted_pairs]
        col_strategy = ('numeric', src_x_sorted, dest_x_nums, col_perm)
    elif src_x_strs and len(src_x_strs) == max_src_cols:
        col_strategy = ('enum_label', src_x_strs)
    else:
        src_col_norm = ([j * (1) for j in range(max_src_cols)] if max_src_cols >= 2 else [0.0])
        col_strategy = ('index', src_col_norm)

    for row_i, dest_vg in enumerate(incorrect_vgs):
        dest_v_elems = dest_vg.findall("V", namespaces=namespace)
        dest_col_count = len(dest_v_elems)
        if dest_col_count == 0:
            continue

        intermediate = []
        if row_strategy[0] == 'numeric':
            _, src_y_sorted, dest_y_nums_local, src_rows_sorted = row_strategy
            dest_y_pos = dest_y_nums_local[row_i]
            for col_j in range(max_src_cols):
                col_signal = [r[col_j] if col_j < len(r) else (r[-1] if r else 0.0) for r in src_rows_sorted]
                v = interp_extrap_values(src_y_sorted, col_signal, [dest_y_pos],
                                         f"{label} (y-resample col={col_j})")[0]
                intermediate.append(v)
        elif row_strategy[0] == 'enum_label':
            _, idx_map, src_rows = row_strategy
            chosen_row = src_rows[idx_map[row_i]] if idx_map else []
            for col_j in range(max_src_cols):
                intermediate.append(
                    chosen_row[col_j] if col_j < len(chosen_row) else (chosen_row[-1] if chosen_row else 0.0)
                )
        else:
            _, src_row_norm, src_rows = row_strategy
            for col_j in range(max_src_cols):
                col_signal = [r[col_j] if col_j < len(r) else (r[-1] if r else 0.0) for r in src_rows]
                v = interp_extrap_values(src_row_norm, col_signal, [row_i],
                                         f"{label} (row-index col={col_j})")[0]
                intermediate.append(v)

        if col_strategy[0] == 'numeric':
            _, src_x_sorted, dest_x_nums_local, col_perm = col_strategy
            intermediate_sorted = [intermediate[j] for j in col_perm]
            final_col_vals = interp_extrap_values(src_x_sorted, intermediate_sorted, dest_x_nums_local,
                                                  f"{label} (x-resample row={row_i})")
        elif col_strategy[0] == 'enum_label':
            _, src_x_labels = col_strategy
            idx_map = _nearest_neighbour_label_map(src_x_labels, dest_x_strs)
            if idx_map:
                final_col_vals = [intermediate[i] for i in idx_map]
            else:
                idx_map = _nearest_neighbour_index_map(max_src_cols, dest_col_count)
                final_col_vals = [intermediate[i] for i in idx_map]
        else:
            if max_src_cols >= 2:
                src_col_norm_local = [j * (dest_col_count - 1) / (max_src_cols - 1) for j in range(max_src_cols)]
                final_col_vals = interp_extrap_values(src_col_norm_local, intermediate, list(range(dest_col_count)),
                                                      f"{label} (col-index row={row_i})")
            else:
                final_col_vals = [intermediate[0]] * dest_col_count

        for col_i, v_elem in enumerate(dest_v_elems):
            v_elem.text = _format_num(final_col_vals[col_i])

        if y_axis_numeric:
            lbl_el = dest_vg.find("LABEL", namespaces=namespace)
            if lbl_el is not None and y_vals_new_display and row_i < len(y_vals_new_display):
                new_lbl = y_vals_new_display[row_i]
                lbl_el.text = (_format_num(new_lbl) if isinstance(new_lbl, (int, float)) else str(new_lbl))

    value_dict = {}
    for vg in incorrect_vgs:
        lbl_el = vg.find("LABEL", namespaces=namespace)
        key = lbl_el.text.strip() if (lbl_el is not None and lbl_el.text) else str(id(vg))
        value_dict[key] = _parse_floats(vg.findall("V", namespaces=namespace))

    snapshot_new = _snapshot_map(incorrect_vgs, dest_x_elems, dest_y_elems, namespace)
    if _snapshots_equal_numeric(snapshot_old, snapshot_new):
        _append_detail(detail_logs, source_name, label, "copied", "cdfx_update", "no value change")
        return 'no_change', None

    _append_detail(
        detail_logs,
        source_name,
        label,
        "updated",
        "cdfx_update",
        "2D map values updated",
        details={"src_rows": src_row_count, "dest_rows": dest_row_count},
    )

    return 'updated', {'label': label, 'values': value_dict,
                       'x_dim': len(x_vals_new_display), 'y_dim': len(y_vals_new_display),
                       'x_dim_val': x_vals_new_display, 'y_dim_val': y_vals_new_display,
                       'status': 'Updated'}


def _update_curve_1d_cdfx(label, correct_sw, incorrect_sw, namespace,
                          detail_logs=None, source_name=None):
    axis_correct = correct_sw.findall(".//SW-VALUES-PHYS", namespaces=namespace)
    axis_incorrect = incorrect_sw.findall(".//SW-VALUES-PHYS", namespaces=namespace)
    if not axis_correct or not axis_incorrect:
        _append_detail(detail_logs, source_name, label, "skipped", "cdfx_update", "missing curve axis blocks")
        return 'skip', None

    src_z_str = _parse_strings(axis_correct[0].findall("V", namespaces=namespace))
    if not src_z_str:
        src_z_str = _parse_strings(axis_correct[0].findall("VT", namespaces=namespace))

    dest_z_elems = axis_incorrect[0].findall("V", namespaces=namespace)
    if not dest_z_elems:
        dest_z_elems = axis_incorrect[0].findall("VT", namespaces=namespace)

    src_z_n = len(src_z_str)
    dest_z_n = len(dest_z_elems)
    if src_z_n == 0 or dest_z_n == 0:
        _append_detail(detail_logs, source_name, label, "skipped", "cdfx_update", "empty source or destination curve")
        return 'skip', None

    src_x_str = []
    dest_x_elems = []
    if len(axis_correct) > 1 and len(axis_incorrect) > 1:
        src_x_str = _parse_strings(axis_correct[1].findall("V", namespaces=namespace))
        if not src_x_str:
            src_x_str = _parse_strings(axis_correct[1].findall("VT", namespaces=namespace))
        dest_x_elems = axis_incorrect[1].findall("V", namespaces=namespace)
        if not dest_x_elems:
            dest_x_elems = axis_incorrect[1].findall("VT", namespaces=namespace)

    src_z_float, numeric_z = _try_float_list(src_z_str)
    src_x_float, src_x_ok = _try_float_list(src_x_str) if src_x_str else ([], False)
    dest_x_float = _parse_floats(dest_x_elems) if dest_x_elems else []
    numeric_x = src_x_ok and bool(dest_x_float)

    old_z_texts = [e.text or "" for e in dest_z_elems]

    if numeric_z and numeric_x and len(src_x_float) == src_z_n:
        new_z = interp_extrap_values(src_x_float, src_z_float, dest_x_float, f"{label} (z@dest_x)")
        new_z_texts = [_format_num(v) for v in new_z]
    elif numeric_z and numeric_x and len(src_x_float) != src_z_n:
        file_logger.warning(
            f"  {label}: src z length ({src_z_n}) != src x length ({len(src_x_float)}) — falling back to index-based interpolation."
        )
        src_x_norm = ([i * (dest_z_n - 1) / (src_z_n - 1) for i in range(src_z_n)] if src_z_n >= 2 else [0.0])
        new_z = interp_extrap_values(src_x_norm, src_z_float, list(range(dest_z_n)), f"{label} (index fallback)")
        new_z_texts = [_format_num(v) for v in new_z]
    elif numeric_z:
        src_x_norm = ([i * (dest_z_n - 1) / (src_z_n - 1) for i in range(src_z_n)] if src_z_n >= 2 else [0.0])
        new_z = interp_extrap_values(src_x_norm, src_z_float, list(range(dest_z_n)), f"{label} (index)")
        new_z_texts = [_format_num(v) for v in new_z]
    else:
        idx_map = _nearest_neighbour_index_map(src_z_n, dest_z_n)
        new_z_texts = [src_z_str[i] for i in idx_map]

    if _text_lists_equal(old_z_texts, new_z_texts):
        _append_detail(detail_logs, source_name, label, "copied", "cdfx_update", "no value change")
        return 'no_change', None

    for elem, text in zip(dest_z_elems, new_z_texts):
        elem.text = text

    _append_detail(
        detail_logs,
        source_name,
        label,
        "updated",
        "cdfx_update",
        "curve values updated",
        details={"dest_points": dest_z_n, "src_points": src_z_n},
    )

    x_vals = [e.text for e in dest_x_elems] if dest_x_elems else []
    z_vals = [e.text for e in dest_z_elems]
    return 'updated', {'label': label, 'x_values': x_vals, 'z_values': z_vals,
                       'x_dim': len(x_vals), 'status': 'Updated'}



def override_incorrect_values(correct_cdfx_file, incorrect_cdfx_file, namespace,
                              progress_callback=None,
                              source_name=None,
                              source_index=None,
                              total_sources=None,
                              detail_logs=None):
    if not os.path.exists(incorrect_cdfx_file):
        try:
            shutil.copyfile(correct_cdfx_file, incorrect_cdfx_file)
            print(f"Created: {incorrect_cdfx_file}")
            return pd.DataFrame({'label': [], 'value': [], 'status': ['Destination Created by Copying Source']})
        except Exception as e:
            logging.error(f"Failed to copy source to destination: {e}")
            return pd.DataFrame()

    correct_df = extract_labels_with_c(correct_cdfx_file, namespace)
    incorrect_df = extract_labels_with_c(incorrect_cdfx_file, namespace)

    if correct_df.empty:
        logging.error("No data extracted from the source file. Aborting update.")
        return pd.DataFrame()

    total_correct_labels = len(correct_df)
    total_destination_labels = len(incorrect_df)
    source_labels = set(correct_df['label'])
    dest_labels = set(incorrect_df['label'])
    dest_only_count = len(dest_labels - source_labels)

    print(f"Source: {total_correct_labels} labels | Destination: {total_destination_labels} labels | "
          f"Dest-only: {dest_only_count} | Log: calibration_process.log")

    merged_df = pd.merge(correct_df, incorrect_df, on='label',
                         suffixes=('_correct', '_incorrect'), how='left', indicator=True)

    try:
        tree_incorrect = ET.parse(incorrect_cdfx_file)
        root_incorrect = tree_incorrect.getroot()
    except Exception as e:
        logging.error(f"Failed to parse destination CDFX file: {e}")
        return pd.DataFrame()

    try:
        tree_correct = ET.parse(correct_cdfx_file)
        root_correct = tree_correct.getroot()
    except Exception as e:
        logging.error(f"Failed to parse source CDFX file: {e}")
        return pd.DataFrame()

    label_index_correct = build_label_index(root_correct, namespace)
    label_index_incorrect = build_label_index(root_incorrect, namespace)

    updated_data = []
    start_time = time.time()
    last_progress_time = start_time
    progress_interval = 120
    processed_count = 0
    updated_count = 0
    no_update_needed_count = 0
    unhandled_suffix_count = 0
    unhandled_suffixes_seen = {}

    for idx, row in merged_df.iterrows():
        label = row['label']
        correct_sw = label_index_correct.get(label)
        incorrect_sw = label_index_incorrect.get(label)
        if row['_merge'] == 'left_only':
            continue
        if correct_sw is None or incorrect_sw is None:
            continue

        processed_count += 1
        if progress_callback and (processed_count == 1 or processed_count % 250 == 0
                                  or processed_count == total_destination_labels):
            progress_callback({
                "type": "row_progress",
                "source": source_name,
                "source_index": source_index,
                "total_sources": total_sources,
                "processed": processed_count,
                "total": total_destination_labels,
                "updated_so_far": updated_count,
            })
        current_time = time.time()
        if current_time - last_progress_time >= progress_interval:
            elapsed_min = (current_time - start_time) / 60
            pct = 100 * processed_count / total_destination_labels
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {elapsed_min:.0f}min | "
                  f"{processed_count}/{total_destination_labels} ({pct:.1f}%) | "
                  f"Updated: {updated_count} | No change: {no_update_needed_count}")
            last_progress_time = current_time

        if label.endswith(('_C', '_CW', '_c')):
            status, record = _update_scalar_cdfx(
                label, correct_sw, incorrect_sw, row, namespace,
                detail_logs=detail_logs, source_name=source_name,
            )
        elif label.endswith('_CA'):
            status, record = _update_array_1d_cdfx(
                label, correct_sw, incorrect_sw, namespace,
                detail_logs=detail_logs, source_name=source_name,
            )
        elif label.endswith(('_MAP', '_M')):
            status, record = _update_map_2d_cdfx(
                label, correct_sw, incorrect_sw, namespace,
                detail_logs=detail_logs, source_name=source_name,
            )
        elif label.endswith(('_T', '_CUR', '_Cur')):
            status, record = _update_curve_1d_cdfx(
                label, correct_sw, incorrect_sw, namespace,
                detail_logs=detail_logs, source_name=source_name,
            )
        else:
            unhandled_suffix_count += 1
            suffix = label.rsplit('_', 1)[-1] if '_' in label else label
            unhandled_suffixes_seen[suffix] = unhandled_suffixes_seen.get(suffix, 0) + 1
            file_logger.info(f"  Unhandled suffix: {label} (suffix=_{suffix}) — skipped")
            _append_detail(
                detail_logs,
                source_name,
                label,
                "skipped",
                "cdfx_update",
                f"unhandled suffix _{suffix}",
            )
            continue

        if status == 'updated':
            updated_data.append(record)
            updated_count += 1
        elif status == 'no_change':
            no_update_needed_count += 1

    try:
        tree_incorrect.write(incorrect_cdfx_file, encoding='utf-8', xml_declaration=True)
        print(f"\nDestination file updated and saved to '{incorrect_cdfx_file}'.")
    except Exception as e:
        logging.error(f"Failed to write updates: {e}")
        return pd.DataFrame()

    _print_summary(updated_count, no_update_needed_count, total_destination_labels,
                   dest_labels - source_labels, time.time() - start_time,
                   unhandled_suffix_count, unhandled_suffixes_seen)
    return pd.DataFrame(updated_data)



def _print_summary(updated_count, no_update_needed_count, total_dest,
                   dest_only_set, elapsed_seconds,
                   unhandled_count=0, unhandled_seen=None):
    dest_only_count = len(dest_only_set)
    pct_updated = 100 * updated_count / total_dest if total_dest else 0
    pct_no_change = 100 * no_update_needed_count / total_dest if total_dest else 0
    attention = total_dest - updated_count - no_update_needed_count
    pct_dest_only = 100 * dest_only_count / total_dest if total_dest else 0
    total_min = elapsed_seconds / 60

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Updated:        {updated_count:5d} ({pct_updated:.1f}%)")
    print(f"  No change:      {no_update_needed_count:5d} ({pct_no_change:.1f}%)")
    print(f"  Need attention: {attention:5d} "
          f"({100*attention/total_dest if total_dest else 0:.1f}%)")
    print(f"  Dest-only:      {dest_only_count:5d} ({pct_dest_only:.1f}%)")
    if unhandled_count:
        print(f"  Unhandled:      {unhandled_count:5d} (suffixes: {unhandled_seen})")
    print(f"  Total:          {total_dest:5d}")
    print(f"  Time:           {total_min:.1f} min")
    print(f"{'='*60}")


# =====================================================================
# Orchestrator — multi-source priority merge into the destination CDFX
# =====================================================================
NAMESPACE = {'autosar': 'http://autosar.org/schema/r4.0'}


def run_pipeline(source_files, destination_file, work_dir=".",
                 attention_xlsx="needs_attention_labels.xlsx",
                 progress_callback=None):
    """
    Apply each CDFX source to destination_file in priority order.
    A label is taken from the first source where it both exists and changes
    the destination value. Writes destination_file in place. Returns a dict
    summary plus the list of leftover labels. A2L+HEX inputs must be
    pre-converted to CDFX (see a2l_hex_to_cdfx.write_cdfx_from_a2l_hex).
    """
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    namespace = NAMESPACE

    dest_df = extract_labels_with_c(destination_file, namespace)
    if dest_df.empty:
        return {"error": "No labels found in destination file.",
                "summary": [], "remaining": [], "all_labels": 0,
                "all_updated": 0}

    all_labels = set(dest_df['label'])
    labels_left = set(all_labels)
    all_labels_updated = set()
    all_labels_copied = set()
    summary = []
    detail_logs = []

    if progress_callback:
        progress_callback({
            "type": "pipeline_start",
            "total_sources": len(source_files),
            "all_labels": len(all_labels),
        })

    for idx, src in enumerate(source_files):
        if progress_callback:
            progress_callback({
                "type": "source_start",
                "source": os.path.basename(src),
                "source_index": idx + 1,
                "total_sources": len(source_files),
                "remaining_before": len(labels_left),
            })
        temp_dest = os.path.join(work_dir, f"_temp_dest_{idx}.CDFX")
        filter_destination_file(destination_file, temp_dest, labels_left, namespace)

        src_df = extract_labels_with_c(src, namespace)
        src_labels = set(src_df['label']) if not src_df.empty else set()
        updated_df = override_incorrect_values(
            src,
            temp_dest,
            namespace,
            progress_callback=progress_callback,
            source_name=os.path.basename(src),
            source_index=idx + 1,
            total_sources=len(source_files),
            detail_logs=detail_logs,
        )

        updated_labels = set(updated_df['label']) if not updated_df.empty else set()
        all_labels_updated.update(updated_labels)

        found_in_src = labels_left & src_labels
        found_but_unchanged = found_in_src - updated_labels
        not_found_in_src = labels_left - found_in_src
        all_labels_copied.update(found_but_unchanged)

        merge_updated_instances(destination_file, temp_dest, updated_labels, namespace)
        try:
            os.remove(temp_dest)
        except OSError:
            pass

        labels_left = not_found_in_src
        summary.append({
            'source': os.path.basename(src),
            'updated': len(updated_labels),
            'found_but_unchanged': len(found_but_unchanged),
            'still_need_attention': len(not_found_in_src),
            'remaining': len(labels_left),
        })
        if progress_callback:
            progress_callback({
                "type": "source_done",
                "source": os.path.basename(src),
                "source_index": idx + 1,
                "total_sources": len(source_files),
                "updated": len(updated_labels),
                "remaining": len(labels_left),
            })

    attention_path = None
    if labels_left:
        attention_path = os.path.join(work_dir, attention_xlsx)
        pd.DataFrame({'label': sorted(labels_left)}).to_excel(attention_path, index=False)
        for label in sorted(labels_left):
            _append_detail(
                detail_logs,
                "pipeline",
                label,
                "attention",
                "pipeline_result",
                "label still needs attention after all sources",
            )

    event_counts = dict(Counter(event.get("action", "unknown") for event in detail_logs))

    result = {
        "all_labels": len(all_labels),
        "all_updated": len(all_labels_updated),
        "all_copied": len(all_labels_copied),
        "updated_labels": sorted(all_labels_updated),
        "copied_labels": sorted(all_labels_copied),
        "remaining": sorted(labels_left),
        "summary": summary,
        "attention_xlsx": attention_path,
        "destination_file": destination_file,
        "detailed_events": detail_logs,
        "detailed_event_counts": event_counts,
    }
    if progress_callback:
        progress_callback({
            "type": "pipeline_done",
            "all_updated": result["all_updated"],
            "remaining": len(result["remaining"]),
        })
    return result


# =====================================================================
# PHASE 2 — group "needs attention" labels by FUNCTION / FUNCTION_VERSION
# Lifted from the first code cell of preCal_v1.ipynb.
# =====================================================================
_FUNC_BLOCK_RE = re.compile(r'/begin\s+FUNCTION',           re.IGNORECASE)
_END_FUNC_RE   = re.compile(r'/end\s+FUNCTION',             re.IGNORECASE)
_DEF_CHAR_RE   = re.compile(r'/begin\s+DEF_CHARACTERISTIC', re.IGNORECASE)
_END_DEF_RE    = re.compile(r'/end\s+DEF_CHARACTERISTIC',   re.IGNORECASE)
_FUNC_VER_RE   = re.compile(r'FUNCTION_VERSION\s+"([^"]*)"', re.IGNORECASE)


def _build_label_lookup_from_a2l(a2l_path):
    """Returns (label->func_name, label->func_version) by single-pass parse."""
    label_to_func = {}
    label_to_ver  = {}

    with open(a2l_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    func_starts = [m.start() for m in _FUNC_BLOCK_RE.finditer(content)]
    func_ends   = [m.end()   for m in _END_FUNC_RE.finditer(content)]

    pairs, ei = [], 0
    for si in func_starts:
        while ei < len(func_ends) and func_ends[ei] <= si:
            ei += 1
        if ei < len(func_ends):
            pairs.append((si, func_ends[ei]))
            ei += 1

    for start, end in pairs:
        block = content[start:end]
        lines = block.splitlines()
        func_name = None
        for i, line in enumerate(lines):
            if _FUNC_BLOCK_RE.search(line.strip()):
                for j in range(i + 1, len(lines)):
                    c = lines[j].strip()
                    if c and not c.startswith('"') and not c.startswith('/'):
                        func_name = c.split()[0]
                        break
                break
        if func_name is None:
            continue

        ver_match = _FUNC_VER_RE.search(block)
        func_ver  = ver_match.group(1) if ver_match else ""

        def_match = _DEF_CHAR_RE.search(block)
        end_match = _END_DEF_RE.search(block)
        if def_match and end_match and def_match.start() < end_match.start():
            for lbl in block[def_match.end(): end_match.start()].split():
                label_to_func[lbl] = func_name
                label_to_ver[lbl]  = func_ver

    return label_to_func, label_to_ver, len(pairs)


def _build_grouped_rows(df, label_col, label_to_func, label_to_ver):
    df = df.copy()
    df['_func']  = df[label_col].map(label_to_func).fillna("NOT FOUND")
    df['_ver']   = df[label_col].map(label_to_ver).fillna("")
    df['_full']  = df['_func'] + '  ' + df['_ver']

    rows = []
    not_found_labels = []

    seen_groups = {}
    for _, row in df.iterrows():
        key = (row['_func'], row['_ver'], row['_full'])
        if key not in seen_groups:
            seen_groups[key] = []
        seen_groups[key].append(row[label_col])

    for (func_name, func_ver, func_full), labels in seen_groups.items():
        if func_name == "NOT FOUND":
            not_found_labels.extend(labels)
            continue
        rows.append({
            'function_full_name': func_full,
            'function_name':      func_name,
            'function_version':   func_ver,
            'label':              f'*** {func_full} ***',
            '_is_header': True,
        })
        for lbl in labels:
            rows.append({
                'function_full_name': '',
                'function_name':      '',
                'function_version':   '',
                'label':              lbl,
                '_is_header': False,
            })
        rows.append({'function_full_name': '', 'function_name': '', 'function_version': '', 'label': '', '_is_header': False})
        rows.append({'function_full_name': '', 'function_name': '', 'function_version': '', 'label': '', '_is_header': False})

    if not_found_labels:
        rows.append({
            'function_full_name': 'NOT FOUND',
            'function_name':      'NOT FOUND',
            'function_version':   '',
            'label':              '*** NOT FOUND IN A2L ***',
            '_is_header': True,
        })
        for lbl in not_found_labels:
            rows.append({
                'function_full_name': 'NOT FOUND',
                'function_name':      'NOT FOUND',
                'function_version':   '',
                'label':              lbl,
                '_is_header': False,
            })

    return rows, len(not_found_labels)


def _write_grouped_excel(rows, output_path, updated_labels=None, copied_labels=None):
    updated_labels = sorted(set(updated_labels or []))
    copied_labels = sorted(set(copied_labels or []))

    cols = ['function_full_name', 'function_name', 'function_version', 'label']
    data = [{c: r[c] for c in cols} for r in rows]
    df_out = pd.DataFrame(data, columns=cols)

    df_updated = pd.DataFrame({'label': updated_labels})
    df_copied = pd.DataFrame({'label': copied_labels})

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_out.to_excel(writer, index=False, sheet_name='Needs Attention Grouped')
        df_updated.to_excel(writer, index=False, sheet_name='Updated Labels')
        df_copied.to_excel(writer, index=False, sheet_name='Copied Labels')

    wb = load_workbook(output_path)
    ws = wb['Needs Attention Grouped']

    hdr_font   = Font(name='Arial', bold=True, color='FFFFFF')
    hdr_fill   = PatternFill('solid', start_color='1F4E79')
    grp_font   = Font(name='Arial', bold=True, color='FFFFFF')
    grp_fill   = PatternFill('solid', start_color='2E75B6')
    lbl_font   = Font(name='Arial', size=10)
    center     = Alignment(horizontal='center', vertical='center')
    left       = Alignment(horizontal='left',   vertical='center')

    for cell in ws[1]:
        cell.font      = hdr_font
        cell.fill      = hdr_fill
        cell.alignment = center

    is_header_map = {i + 2: r['_is_header'] for i, r in enumerate(rows)}
    for row_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):
        is_grp_hdr = is_header_map.get(row_idx, False)
        for cell in row:
            if is_grp_hdr:
                cell.font      = grp_font
                cell.fill      = grp_fill
                cell.alignment = center
            else:
                cell.font      = lbl_font
                cell.alignment = left

    for col_letter, width in {'A': 45, 'B': 25, 'C': 25, 'D': 40}.items():
        ws.column_dimensions[col_letter].width = width
    ws.freeze_panes = 'A2'

    for sheet_name in ('Updated Labels', 'Copied Labels'):
        ws_simple = wb[sheet_name]
        for cell in ws_simple[1]:
            cell.font = hdr_font
            cell.fill = hdr_fill
            cell.alignment = center
        ws_simple.column_dimensions['A'].width = 50
        ws_simple.freeze_panes = 'A2'

    wb.save(output_path)


def group_attention_labels_by_function(xlsx_path, a2l_path, output_path=None,
                                       updated_labels=None, copied_labels=None):
    """
    Read the "needs attention" xlsx (one label per row, column named 'label'
    or the first column), look each label up in the A2L FUNCTION blocks, and
    write a grouped/styled xlsx where labels are segregated by FUNCTION and
    FUNCTION_VERSION.

    Returns dict: {output_path, functions_parsed, labels_indexed,
                   not_found_count, total_labels, updated_total, copied_total}.
    """
    if output_path is None:
        output_path = xlsx_path  # overwrite in place, matching notebook

    label_to_func, label_to_ver, func_count = _build_label_lookup_from_a2l(a2l_path)

    df = pd.read_excel(xlsx_path)
    rows = []
    not_found = 0
    total_labels = 0

    if not df.empty:
        label_col = next(
            (c for c in df.columns if isinstance(c, str) and c.strip().lower() == 'label'),
            df.columns[0],
        )
        rows, not_found = _build_grouped_rows(df, label_col, label_to_func, label_to_ver)
        total_labels = len(df)

    _write_grouped_excel(
        rows,
        output_path,
        updated_labels=updated_labels,
        copied_labels=copied_labels,
    )

    return {
        "output_path": output_path,
        "functions_parsed": func_count,
        "labels_indexed": len(label_to_func),
        "not_found_count": not_found,
        "total_labels": total_labels,
        "updated_total": len(set(updated_labels or [])),
        "copied_total": len(set(copied_labels or [])),
    }
