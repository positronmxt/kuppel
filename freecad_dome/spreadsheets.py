"""FreeCAD Spreadsheet export helpers.

Goal: generate human-friendly manufacturing/reference tables inside the FreeCAD
Document without affecting geometry generation.

This module degrades gracefully when FreeCAD isn't available.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math

from .parameters import DomeParameters
from .tessellation import TessellatedDome
from .struts import StrutInstance

__all__ = ["export_dome_spreadsheets"]


def export_dome_spreadsheets(
    doc: Any,
    *,
    params: DomeParameters,
    dome: TessellatedDome,
    struts: Sequence[StrutInstance] = (),
) -> Dict[str, Any]:
    """Create/update spreadsheets in the FreeCAD document.

    Creates 3-4 sheets:
    - Dome_Params: configuration values
    - Dome_Summary: derived counts + key computed values
    - Dome_Struts: one row per generated strut instance (incl. split halves)
    - Dome_Porch: entry porch cut list + glazing panels (when enabled)

    Returns a dict with created/updated object names.
    """
    try:
        import FreeCAD  # type: ignore
    except Exception:  # pragma: no cover
        return {"created": [], "updated": []}

    if doc is None:
        doc = getattr(FreeCAD, "ActiveDocument", None)
    if doc is None:
        return {"created": [], "updated": []}

    created: List[str] = []
    updated: List[str] = []

    sh_params, is_new = _ensure_sheet(doc, "Dome_Params")
    created += [sh_params.Name] if is_new else []
    updated += [sh_params.Name]
    _fill_params_sheet(sh_params, params)

    sh_summary, is_new = _ensure_sheet(doc, "Dome_Summary")
    created += [sh_summary.Name] if is_new else []
    updated += [sh_summary.Name]
    _fill_summary_sheet(sh_summary, params, dome)

    sh_struts, is_new = _ensure_sheet(doc, "Dome_Struts")
    created += [sh_struts.Name] if is_new else []
    updated += [sh_struts.Name]
    _fill_struts_sheet(sh_struts, doc, params, dome, list(struts))

    if params.generate_entry_porch:
        sh_porch, is_new = _ensure_sheet(doc, "Dome_Porch")
        created += [sh_porch.Name] if is_new else []
        updated += [sh_porch.Name]
        _fill_porch_sheet(sh_porch, params)

    try:
        doc.recompute()
    except Exception:
        pass

    return {"created": created, "updated": updated}


def _ensure_sheet(doc: Any, name: str) -> Tuple[Any, bool]:
    """Return (sheet, created)."""
    for o in list(getattr(doc, "Objects", []) or []):
        if str(getattr(o, "Name", "")) == name and str(getattr(o, "TypeId", "")) == "Spreadsheet::Sheet":
            return o, False
    sheet = doc.addObject("Spreadsheet::Sheet", name)
    try:
        sheet.Label = name
    except Exception:
        pass
    return sheet, True


def _clear_sheet(sheet: Any, *, rows: int = 3000, cols: int = 26) -> None:
    # Spreadsheet API differences across FreeCAD versions; best-effort clear.
    try:
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                sheet.set(_cell_name(c, r), "")
    except Exception:
        # If bulk clear fails, do nothing (we'll overwrite used cells anyway).
        pass


def _cell_name(col: int, row: int) -> str:
    # 1 -> A, 26 -> Z, 27 -> AA ... (we only use <= Z)
    col = max(1, int(col))
    row = max(1, int(row))
    if col <= 26:
        return f"{chr(ord('A') + col - 1)}{row}"
    col -= 1
    return f"{chr(ord('A') + (col // 26) - 1)}{chr(ord('A') + (col % 26))}{row}"


def _write_table(sheet: Any, start_row: int, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> int:
    r = int(start_row)
    for c, h in enumerate(headers, start=1):
        sheet.set(_cell_name(c, r), str(h))
    r += 1
    for row in rows:
        for c, val in enumerate(row, start=1):
            sheet.set(_cell_name(c, r), _format_cell(val))
        r += 1
    return r


def _format_cell(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        # Keep a stable precision without being noisy.
        if isinstance(val, float):
            if math.isfinite(val):
                return f"{val:.6g}"
            return ""
        return str(val)
    return str(val)


def _fill_params_sheet(sheet: Any, params: DomeParameters) -> None:
    _clear_sheet(sheet, rows=500, cols=8)

    data = params.to_dict()
    # Materials are a nested dict; not useful in a flat sheet.
    data.pop("materials", None)

    def _unit_for_key(k: str) -> str:
        if k.endswith("_m"):
            return "m"
        if k.endswith("_mm"):
            return "mm"
        if k.endswith("_ratio"):
            return "(0..1)"
        if k in {"frequency"}:
            return ""  # unitless
        return ""

    rows = []
    for key in sorted(data.keys()):
        rows.append((key, data[key], _unit_for_key(key)))

    _write_table(sheet, 1, ["Key", "Value", "Unit"], rows)


def _fill_summary_sheet(sheet: Any, params: DomeParameters, dome: TessellatedDome) -> None:
    _clear_sheet(sheet, rows=200, cols=8)

    hemi = float(params.hemisphere_ratio)
    belt_height = float(params.radius_m) * (1.0 - 2.0 * hemi) if hemi < 1.0 else None

    rows = [
        ("nodes", len(dome.nodes), ""),
        ("panels", len(dome.panels), ""),
        ("struts", len(dome.struts), ""),
        ("radius_m", float(params.radius_m), "m"),
        ("frequency", int(params.frequency), ""),
        ("hemisphere_ratio", float(hemi), "(0..1)"),
        ("belt_height_m", belt_height, "m"),
        ("split_struts_per_panel", params.split_struts_per_panel, ""),
        ("generate_entry_porch", params.generate_entry_porch, ""),
    ]

    _write_table(sheet, 1, ["Metric", "Value", "Unit"], rows)


def _fill_struts_sheet(
    sheet: Any,
    doc: Any,
    params: DomeParameters,
    dome: TessellatedDome,
    instances: List[StrutInstance],
) -> None:
    # Keep this sheet large but bounded.
    _clear_sheet(sheet, rows=3000, cols=26)

    by_name: Dict[str, Any] = {str(getattr(o, "Name", "")): o for o in list(getattr(doc, "Objects", []) or [])}

    hemi = float(params.hemisphere_ratio)
    belt_height = float(params.radius_m) * (1.0 - 2.0 * hemi) if hemi < 1.0 else None
    eps = max(1e-6, float(params.radius_m) * 1e-5)

    def _is_belt_strut_def(s) -> bool:
        if belt_height is None:
            return False
        try:
            return abs(float(s.start[2]) - belt_height) <= eps and abs(float(s.end[2]) - belt_height) <= eps
        except Exception:
            return False

    # Map (group, seq) -> base strut def index (approx) to pull topology info.
    # In FreeCAD mode, names are Strut_Lxx_### or Strut_Lxx_###_P####.
    def _parse_key(n: str) -> Tuple[str, int]:
        parts = n.split("_")
        if len(parts) < 3:
            return ("", -1)
        group = parts[1]
        try:
            seq = int(parts[2])
        except Exception:
            seq = -1
        return (group, seq)

    # Build a stable ordering using instance list; fall back to name ordering.
    ordered = list(instances)
    if not ordered and dome.struts:
        # If caller didn't pass instances, we can still list dome.struts.
        ordered = [
            StrutInstance(
                name=f"Strut_Unknown_{i+1:03d}",
                length=s.length,
                material=params.material,
                group="",
                sequence=i + 1,
                ifc_guid="",
            )
            for i, s in enumerate(dome.struts)
        ]

    headers = [
        "Name",
        "Family",
        "Seq",
        "Length_m",
        "Length_mm",
        "Belt",
        "Split",
        "PanelA",
        "PanelB",
        "NodeA",
        "NodeB",
        "Start_x",
        "Start_y",
        "Start_z",
        "End_x",
        "End_y",
        "End_z",
        "BevelUsed",
        "StartCutAxisAngleDeg",
        "EndCutAxisAngleDeg",
    ]

    rows: List[List[Any]] = []

    # For lookup by approximate group/seq into dome.struts, we keep the grouped order used
    # by StrutBuilder._group_by_length.
    grouped = _group_struts_by_length_like_builder(dome.struts, params)
    group_to_list: Dict[str, List[Any]] = {g: lst for g, lst in grouped}

    for inst in ordered:
        name = inst.name
        group, seq = _parse_key(name)
        sdef = None
        if group in group_to_list and 1 <= seq <= len(group_to_list[group]):
            sdef = group_to_list[group][seq - 1]

        belt = _is_belt_strut_def(sdef) if sdef is not None else False
        split = "_P" in name

        panel_a = None
        panel_b = None
        node_a = None
        node_b = None
        start = end = None
        if sdef is not None:
            node_a = getattr(sdef, "start_index", None)
            node_b = getattr(sdef, "end_index", None)
            panels = list(getattr(sdef, "panel_indices", ()) or [])
            if len(panels) >= 1:
                panel_a = panels[0]
            if len(panels) >= 2:
                panel_b = panels[1]
            start = getattr(sdef, "start", None)
            end = getattr(sdef, "end", None)

        fc_obj = by_name.get(name)
        bevel_used = None
        start_axis_angle = None
        end_axis_angle = None
        if fc_obj is not None:
            bevel_used = getattr(fc_obj, "BevelUsed", None)
            start_axis_angle = _axis_angle_deg(
                getattr(fc_obj, "StrutDirection", None),
                getattr(fc_obj, "StartCutNormal", None),
            )
            end_axis_angle = _axis_angle_deg(
                getattr(fc_obj, "StrutDirection", None),
                getattr(fc_obj, "EndCutNormal", None),
            )

        rows.append(
            [
                name,
                inst.group,
                inst.sequence,
                float(inst.length),
                float(inst.length) * 1000.0,
                belt,
                split,
                panel_a,
                panel_b,
                node_a,
                node_b,
                (start[0] if start is not None else None),
                (start[1] if start is not None else None),
                (start[2] if start is not None else None),
                (end[0] if end is not None else None),
                (end[1] if end is not None else None),
                (end[2] if end is not None else None),
                bevel_used,
                start_axis_angle,
                end_axis_angle,
            ]
        )

    _write_table(sheet, 1, headers, rows)


def _fill_porch_sheet(sheet: Any, params: DomeParameters) -> None:
    _clear_sheet(sheet, rows=400, cols=16)

    members, glass = _compute_entry_porch_bom(params)

    r = 1
    sheet.set(_cell_name(1, r), "Entry porch")
    r += 2

    r = _write_table(
        sheet,
        r,
        ["Part", "Role", "Count", "Stock_mm", "CutLength_mm"],
        members,
    )
    r += 2
    r = _write_table(
        sheet,
        r,
        ["Part", "Location", "Count", "Width_mm", "Height_mm", "Thickness_mm"],
        glass,
    )


def _compute_entry_porch_bom(params: DomeParameters) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Return (member_rows, glass_rows) for the entry porch.

    This is intentionally FreeCAD-independent so it can be unit-tested.

    Member rows: [Part, Role, Count, Stock_mm, CutLength_mm]
    Glass rows:  [Part, Location, Count, Width_mm, Height_mm, Thickness_mm]
    """

    depth_m = float(params.porch_depth_m)
    depth_m = min(depth_m, 0.5)
    width_m = float(params.porch_width_m)
    height_m = float(params.porch_height_m)
    member_m = float(params.porch_member_size_m)
    glass_m = float(params.porch_glass_thickness_m)

    door_w_m = float(params.door_width_m)
    door_h_m = float(params.door_height_m)
    door_h_eff_m = min(door_h_m, height_m)

    stock_mm = member_m * 1000.0

    # Mirrors the solid inventory in EntryPorchBuilder.
    member_specs: List[Tuple[str, int, float]] = [
        ("Front post", 2, height_m),
        ("Front rail", 2, width_m),
        ("Side return", 4, depth_m),
        ("Side post", 2, height_m),
        ("Top return", 1, width_m),
        ("Door stile", 2, door_h_eff_m),
        ("Door rail", 2, door_w_m),
    ]

    members: List[List[Any]] = []
    for role, count, length_m in member_specs:
        members.append([
            "Member",
            role,
            int(count),
            float(stock_mm),
            float(length_m) * 1000.0,
        ])

    glass: List[List[Any]] = []
    if glass_m > 0:
        inset_m = member_m * 0.5
        front_w = max(0.0, width_m - 2.0 * inset_m)
        front_h = max(0.0, height_m - 2.0 * inset_m)
        if front_w > 0 and front_h > 0:
            glass.append([
                "Glass",
                "Front panel",
                1,
                front_w * 1000.0,
                front_h * 1000.0,
                glass_m * 1000.0,
            ])

        d_inset_m = member_m * 1.2
        door_w = max(0.0, door_w_m - 2.0 * d_inset_m)
        door_h = max(0.0, door_h_eff_m - 2.0 * d_inset_m)
        if door_w > 0 and door_h > 0:
            glass.append([
                "Glass",
                "Door leaf",
                1,
                door_w * 1000.0,
                door_h * 1000.0,
                glass_m * 1000.0,
            ])

    return members, glass


def _axis_angle_deg(direction: Any, plane_normal: Any) -> Optional[float]:
    """Angle between strut axis and cut-plane normal.

    0° means a square end-cut (normal aligned with axis).
    """
    d = _vec3(direction)
    n = _vec3(plane_normal)
    if d is None or n is None:
        return None
    dl = _len3(d)
    nl = _len3(n)
    if dl <= 1e-12 or nl <= 1e-12:
        return None
    dot = abs((d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) / (dl * nl))
    dot = 1.0 if dot > 1.0 else dot
    return float(math.degrees(math.acos(dot)))


def _vec3(v: Any) -> Optional[Tuple[float, float, float]]:
    if v is None:
        return None
    for attrs in (("x", "y", "z"), ("X", "Y", "Z")):
        try:
            return (float(getattr(v, attrs[0])), float(getattr(v, attrs[1])), float(getattr(v, attrs[2])))
        except Exception:
            pass
    if isinstance(v, (tuple, list)) and len(v) >= 3:
        try:
            return (float(v[0]), float(v[1]), float(v[2]))
        except Exception:
            return None
    return None


def _len3(v: Tuple[float, float, float]) -> float:
    return float((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5)


def _group_struts_by_length_like_builder(struts: List[Any], params: DomeParameters) -> List[Tuple[str, List[Any]]]:
    """Mirror StrutBuilder._group_by_length so spreadsheet can map group+seq back to strut defs."""
    tolerance = max(float(params.clearance_m), 1e-4)
    buckets: Dict[int, List[Any]] = {}
    for s in struts:
        try:
            key = round(float(getattr(s, "length", 0.0)) / tolerance)
        except Exception:
            key = 0
        buckets.setdefault(key, []).append(s)

    grouped: List[Tuple[str, List[Any]]] = []
    for idx, key in enumerate(sorted(buckets), start=1):
        grouped.append((f"L{idx:02d}", buckets[key]))
    return grouped
