"""
Extract _map labels (case-insensitive) from a CDFX file whose SW-VALUE-CONT
contains string/enum values (<VT>) or a mix of numeric (<V>) and string (<VT>).
Pure-numeric _map labels are excluded.
"""
import sys
from xml.etree import ElementTree as ET
from pathlib import Path

CDFX = Path(r"C:/Users/SHUE1KOR/Desktop/CalAi/A2L_HEX_parser/ADM_28.CDFX")

enum_only = []   # only <VT>
mixed     = []   # both <V> and <VT>

# iterparse end-events on SW-INSTANCE, then clear to keep memory low.
context = ET.iterparse(str(CDFX), events=("end",))
for _, elem in context:
    tag = elem.tag.split('}')[-1]  # strip namespace if any
    if tag != "SW-INSTANCE":
        continue

    short_name_el = next(
        (c for c in elem if c.tag.split('}')[-1] == "SHORT-NAME"), None
    )
    if short_name_el is None or not short_name_el.text:
        elem.clear()
        continue
    name = short_name_el.text

    if "_map" not in name.lower():
        elem.clear()
        continue

    # Look only inside SW-VALUE-CONT (skip axis content)
    has_v = False
    has_vt = False
    for value_cont in elem.iter():
        if value_cont.tag.split('}')[-1] != "SW-VALUE-CONT":
            continue
        for phys in value_cont.iter():
            t = phys.tag.split('}')[-1]
            if t == "V":
                has_v = True
            elif t == "VT":
                has_vt = True
            if has_v and has_vt:
                break
        if has_v and has_vt:
            break

    if has_vt and has_v:
        mixed.append(name)
    elif has_vt:
        enum_only.append(name)
    # pure numeric (has_v only) -> skip

    elem.clear()

out = Path(CDFX.parent, "map_enum_labels.txt")
with out.open("w", encoding="utf-8") as f:
    f.write(f"# Enum-only _map labels ({len(enum_only)})\n")
    for n in enum_only:
        f.write(n + "\n")
    f.write(f"\n# Mixed (numeric + string) _map labels ({len(mixed)})\n")
    for n in mixed:
        f.write(n + "\n")

print(f"Enum-only _map labels: {len(enum_only)}")
print(f"Mixed   _map labels: {len(mixed)}")
print(f"Output written to: {out}")
