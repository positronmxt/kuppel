"""TechDraw helpers for generating drawings of representative parts.

This module is intended to be imported from within FreeCAD (GUI or FreeCADCmd).
When FreeCAD isn't available, functions are safe no-ops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(slots=True)
class TechDrawOptions:
    template_path: str = ""
    page_prefix: str = "TD_"
    view_scale: float = 0.5
    auto_unit_scale: bool = True
    unit_scale: float = 1000.0
    include_struts: bool = True
    include_panel_frames: bool = True
    include_glass_panels: bool = True
    include_panel_faces: bool = False
    export_pdf: bool = False
    export_dxf: bool = False
    export_dir: str = ""


def default_template_path() -> str:
    """Best-effort default TechDraw template path."""
    # First, prefer a repo-provided template to avoid depending on FreeCAD's bundled templates.
    try:
        repo_root = Path(__file__).resolve().parents[1]
        repo_tpl = repo_root / "techdraw_templates" / "A2_Landscape.svg"
        if repo_tpl.exists():
            return str(repo_tpl)
    except Exception:
        pass

    try:
        import FreeCAD  # type: ignore

        base = Path(FreeCAD.getResourceDir())
        # Prefer A2 (requested for manufacturing drawings).
        for name in (
            "A2_LandscapeTD.svg",
            "A2_Landscape.svg",
            "A2_PortraitTD.svg",
            "A2_Portrait.svg",
            "A4_LandscapeTD.svg",
            "A4_Landscape.svg",
        ):
            cand = base / "Mod" / "TechDraw" / "Templates" / name
            if cand.exists():
                return str(cand)
    except Exception:
        return ""
    return ""


def generate_representative_techdraw(doc: Any, options: TechDrawOptions) -> Dict[str, Any]:
    """Create TechDraw pages for representative parts in the given FreeCAD document."""
    try:
        import FreeCAD  # type: ignore
        import TechDraw  # type: ignore
    except Exception as exc:  # pragma: no cover
        logging.warning("TechDraw generation requires FreeCAD: %s", exc)
        return {"created_pages": 0, "exported_pdf": 0, "exported_dxf": 0}

    if doc is None:
        doc = getattr(FreeCAD, "ActiveDocument", None)
    if doc is None:
        return {"created_pages": 0, "exported_pdf": 0, "exported_dxf": 0}

    template_path = (options.template_path or "").strip() or default_template_path()
    if template_path and not Path(template_path).exists():
        logging.warning("TechDraw template not found: %s", template_path)
        template_path = ""

    reps = _collect_representatives(doc, options)
    created_pages = 0
    exported_pdf = 0
    exported_dxf = 0

    for key, obj in reps.items():
        page_label = f"{options.page_prefix}{key}"
        page = _create_page(doc, page_label)
        if page is None:
            continue
        # TechDraw requires a template; always attach one.
        if not template_path:
            template_path = default_template_path()
        if template_path:
            _attach_template(doc, page, template_path)
        views = _create_part_views_4(
            doc,
            obj,
            base_label=key,
            options=options,
        )
        if not views:
            continue
        for v in views:
            try:
                page.addView(v)
            except Exception:
                try:
                    page.Views = list(getattr(page, "Views", []) or []) + [v]
                except Exception:
                    pass

            # Some FreeCAD/TechDraw versions only apply Direction/XDirection reliably
            # after the view is attached to a page. Re-apply if we stored a target.
            try:
                _reapply_view_orientation(v)
            except Exception:
                pass

        _layout_4views(page, views)

        # Force page/doc recompute so TechDraw regenerates projections.
        try:
            page.recompute()
        except Exception:
            pass
        try:
            doc.recompute()
        except Exception:
            pass

        created_pages += 1

        if options.export_pdf or options.export_dxf:
            out_dir = Path(options.export_dir or "").expanduser().resolve() if options.export_dir else None
            if out_dir:
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    out_dir = None

        if options.export_pdf:
            path = _export_page_pdf(page, options, key)
            if path:
                exported_pdf += 1
        if options.export_dxf:
            path = _export_page_dxf(page, options, key)
            if path:
                exported_dxf += 1

    try:
        doc.recompute()
    except Exception:
        pass

    return {"created_pages": created_pages, "exported_pdf": exported_pdf, "exported_dxf": exported_dxf}


def _collect_representatives(doc: Any, options: TechDrawOptions) -> Dict[str, Any]:
    """Return mapping key->FreeCAD object."""
    objs = list(getattr(doc, "Objects", []) or [])

    reps: Dict[str, Any] = {}

    if options.include_struts:
        # Prefer the geometry solids: Strut_<group>_<seq>_Geom
        for obj in objs:
            name = str(getattr(obj, "Name", ""))
            label = str(getattr(obj, "Label", ""))
            raw = name or label
            if not raw.startswith("Strut_"):
                continue
            if not raw.endswith("_Geom"):
                continue
            parts = raw.split("_")
            if len(parts) < 4:
                continue
            group = parts[1]
            seq = parts[2]
            if not seq.isdigit():
                continue
            key = f"Strut_{group}"
            # Pick the lowest sequence as representative.
            if key not in reps or int(seq) < int(str(getattr(reps[key], "Name", "999999")).split("_")[2]):
                reps[key] = obj

    def _sig_for_shape(o: Any) -> Optional[Tuple[int, int, int, float, float]]:
        shape = getattr(o, "Shape", None)
        if shape is None:
            return None
        try:
            faces = len(getattr(shape, "Faces", []) or [])
            edges = len(getattr(shape, "Edges", []) or [])
            verts = len(getattr(shape, "Vertexes", []) or [])
            area = float(getattr(shape, "Area", 0.0))
            vol = float(getattr(shape, "Volume", 0.0))
        except Exception:
            return None
        # Quantize to reduce tiny floating noise.
        return (faces, edges, verts, round(area, 6), round(vol, 6))

    def _collect_by_prefix(prefix: str, enabled: bool) -> None:
        if not enabled:
            return
        buckets: Dict[Tuple[int, int, int, float, float], Any] = {}
        for obj in objs:
            name = str(getattr(obj, "Name", ""))
            label = str(getattr(obj, "Label", ""))
            raw = name or label
            if not raw.startswith(prefix):
                continue
            if prefix == "Panel_" and raw.startswith("PanelFrame_"):
                continue
            if prefix == "Panel_" and raw.startswith("GlassPanel_"):
                continue
            sig = _sig_for_shape(obj)
            if sig is None:
                continue
            if sig not in buckets:
                buckets[sig] = obj
        for idx, o in enumerate(buckets.values(), start=1):
            reps[f"{prefix.rstrip('_')}_{idx:03d}"] = o

    _collect_by_prefix("PanelFrame_", options.include_panel_frames)
    _collect_by_prefix("GlassPanel_", options.include_glass_panels)
    _collect_by_prefix("Panel_", options.include_panel_faces)

    return reps


def _create_page(doc: Any, label: str):
    try:
        page = doc.addObject("TechDraw::DrawPage", label)
        # Keep label readable.
        try:
            page.Label = label
        except Exception:
            pass
        return page
    except Exception as exc:
        logging.warning("Failed to create TechDraw page %s: %s", label, exc)
        return None


def _attach_template(doc: Any, page: Any, template_path: str) -> None:
    # Prefer DrawSVGTemplate object (most compatible), fallback to direct path.
    templ_obj = None
    try:
        templ_obj = doc.addObject("TechDraw::DrawSVGTemplate", f"{page.Name}_Template")
    except Exception:
        templ_obj = None

    if templ_obj is not None:
        try:
            templ_obj.Template = template_path
        except Exception:
            try:
                templ_obj.Template = str(Path(template_path).resolve())
            except Exception:
                pass
        try:
            page.Template = templ_obj
            return
        except Exception:
            pass

    try:
        page.Template = template_path
    except Exception:
        pass


def _create_part_view(doc: Any, obj: Any, view_label: str):
    try:
        view = doc.addObject("TechDraw::DrawViewPart", view_label)
    except Exception as exc:
        logging.warning("Failed to create TechDraw view: %s", exc)
        return None

    try:
        view.Source = [obj]
    except Exception:
        try:
            view.Source = obj
        except Exception:
            pass
    return view


def _create_part_views_4(doc: Any, obj: Any, base_label: str, options: TechDrawOptions) -> List[Any]:
    """Create 4 views: top, side, end, isometric."""
    try:
        import FreeCAD  # type: ignore
    except Exception:
        return []

    # Determine local coordinate frame for the part (x, y, z).
    x_axis, y_axis, z_axis = _local_axes_for_object(obj)

    # Apply scale: requested drawing scale (e.g. 1:2 => 0.5) times optional unit scale.
    unit_scale = float(options.unit_scale)
    if bool(options.auto_unit_scale):
        unit_scale = float(_auto_unit_scale_for_object(obj, default=float(options.unit_scale)))
    scale = float(options.view_scale) * unit_scale

    views: List[Any] = []

    def _mk(suffix: str) -> Any | None:
        return _create_part_view(doc, obj, view_label=f"View_{base_label}_{suffix}")

    # Side view: look along +X, horizontal is +Y.
    v_front = _mk("FRONT")
    if v_front is not None:
        _set_view_direction(v_front, FreeCAD.Vector(*x_axis), FreeCAD.Vector(*y_axis))
        _set_view_scale(v_front, scale)
        views.append(v_front)

    # Top view: look along +Z, horizontal is +Y.
    v_top = _mk("TOP")
    if v_top is not None:
        _set_view_direction(v_top, FreeCAD.Vector(*z_axis), FreeCAD.Vector(*y_axis))
        _set_view_scale(v_top, scale)
        views.append(v_top)

    # End view: look along +Y (along length), horizontal is +X.
    v_end = _mk("END")
    if v_end is not None:
        _set_view_direction(v_end, FreeCAD.Vector(*y_axis), FreeCAD.Vector(*x_axis))
        _set_view_scale(v_end, scale)
        views.append(v_end)

    # Isometric.
    v_iso = _mk("ISO")
    if v_iso is not None:
        iso_dir = (
            x_axis[0] + y_axis[0] + z_axis[0],
            x_axis[1] + y_axis[1] + z_axis[1],
            x_axis[2] + y_axis[2] + z_axis[2],
        )
        iso_dir = _normalize3_tuple(iso_dir) or (1.0, 1.0, 1.0)
        _set_view_direction(v_iso, FreeCAD.Vector(*iso_dir), FreeCAD.Vector(*y_axis))
        _set_view_scale(v_iso, scale)
        views.append(v_iso)

    return views


def _set_obj_prop(obj: Any, name: str, value: Any) -> bool:
    """Best-effort property setter across FreeCAD versions."""
    if not name:
        return False
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        pass
    try:
        obj.setPropertyByName(name, value)
        return True
    except Exception:
        return False


def _pick_prop_name(obj: Any, candidates: Iterable[str]) -> str:
    props = set(getattr(obj, "PropertiesList", []) or [])
    for n in candidates:
        if n in props:
            return n
    for n in candidates:
        if hasattr(obj, n):
            return n
    return ""


def _mark_view_orientation(view: Any, direction: Any, x_direction: Any) -> None:
    """Store target orientation for later re-apply after attaching to a page."""
    try:
        import FreeCAD  # type: ignore

        view._geodesic_td_direction = FreeCAD.Vector(direction)
        view._geodesic_td_xdirection = FreeCAD.Vector(x_direction)
    except Exception:
        try:
            view._geodesic_td_direction = direction
            view._geodesic_td_xdirection = x_direction
        except Exception:
            pass


def _reapply_view_orientation(view: Any) -> None:
    d = getattr(view, "_geodesic_td_direction", None)
    xd = getattr(view, "_geodesic_td_xdirection", None)
    if d is None or xd is None:
        return
    _set_view_direction(view, d, xd, _store_target=False)


def _set_view_direction(view: Any, direction, x_direction, *, _store_target: bool = True) -> None:
    # TechDraw expects XDirection to lie in the view plane (orthogonal to Direction).
    try:
        import FreeCAD  # type: ignore

        d = FreeCAD.Vector(direction)
        if d.Length > 1e-12:
            d.normalize()
        xd = FreeCAD.Vector(x_direction)
        # Project onto view plane.
        xd = xd - d * float(xd.dot(d))
        if xd.Length <= 1e-12:
            # Fallback: pick any perpendicular.
            xd = d.cross(FreeCAD.Vector(0, 0, 1))
            if xd.Length <= 1e-12:
                xd = d.cross(FreeCAD.Vector(0, 1, 0))
        if xd.Length > 1e-12:
            xd.normalize()

        if _store_target:
            _mark_view_orientation(view, d, xd)

        dir_name = _pick_prop_name(view, ("Direction", "ViewDirection"))
        xdir_name = _pick_prop_name(view, ("XDirection", "XDir"))

        _set_obj_prop(view, dir_name, d)
        _set_obj_prop(view, xdir_name, xd)

        # Hint TechDraw to refresh.
        try:
            view.touch()
        except Exception:
            pass

        return
    except Exception:
        pass

    # Non-FreeCAD fallback (shouldn't happen in practice).
    if _store_target:
        _mark_view_orientation(view, direction, x_direction)
    _set_obj_prop(view, _pick_prop_name(view, ("Direction", "ViewDirection")), direction)
    _set_obj_prop(view, _pick_prop_name(view, ("XDirection", "XDir")), x_direction)


def _set_view_scale(view: Any, scale: float) -> None:
    try:
        s = float(scale)
    except Exception:
        return
    if s <= 0:
        return

    # Try to disable Auto scale / set Custom scale.
    # Different FreeCAD versions expose ScaleType differently.
    try:
        if hasattr(view, "ScaleType"):
            try:
                view.ScaleType = "Custom"
            except Exception:
                try:
                    view.ScaleType = "Manual"
                except Exception:
                    try:
                        view.ScaleType = 0
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        if hasattr(view, "Scale"):
            view.Scale = s
    except Exception:
        pass


def _layout_4views(page: Any, views: List[Any]) -> None:
    """Place 4 views on an A2 landscape page.

    Coordinates are in page units (mm). If the FreeCAD version doesn't support
    positioning properties, this is a no-op.
    """
    # Map by suffix.
    by = {}
    for v in views:
        name = str(getattr(v, "Name", ""))
        label = str(getattr(v, "Label", ""))
        key = name or label
        for suf in ("FRONT", "TOP", "END", "ISO"):
            if key.endswith(suf):
                by[suf] = v

    # A2 landscape 594x420 with 10mm border in our template.
    pos = {
        "TOP": (170.0, 330.0),
        "FRONT": (170.0, 200.0),
        "END": (410.0, 200.0),
        "ISO": (470.0, 320.0),
    }

    for suf, (x, y) in pos.items():
        v = by.get(suf)
        if v is None:
            continue
        _set_view_position(v, x, y)


def _set_view_position(view: Any, x_mm: float, y_mm: float) -> None:
    # Try common TechDraw properties across versions.
    try:
        if hasattr(view, "X") and hasattr(view, "Y"):
            view.X = float(x_mm)
            view.Y = float(y_mm)
            return
    except Exception:
        pass
    try:
        if hasattr(view, "Position"):
            view.Position = (float(x_mm), float(y_mm))
            return
    except Exception:
        pass
    try:
        import FreeCAD  # type: ignore

        if hasattr(view, "Position"):
            view.Position = FreeCAD.Vector(float(x_mm), float(y_mm), 0.0)
            return
    except Exception:
        pass


def _auto_unit_scale_for_object(obj: Any, default: float = 1000.0) -> float:
    """Return 1000 when the object looks like it's modeled in meters.

    Heuristic: if the bounding-box diagonal is under ~10, treat values as meters.
    """
    shape = getattr(obj, "Shape", None)
    bb = getattr(shape, "BoundBox", None) if shape is not None else None
    diag = None
    if bb is not None:
        try:
            diag = float(bb.DiagonalLength)
        except Exception:
            diag = None
    if diag is None:
        return default
    return default if diag < 10.0 else 1.0


def _local_axes_for_object(obj: Any) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """Return orthonormal (x, y, z) axes for drawing views.

    Rules:
    - Glass panels: Z = panel plane normal (obj.GlassNormal)
    - Struts: Y = strut length axis (obj.StrutDirection), Z = radial from origin through strut centroid
    - Fallback: choose stable axes from bounding box.
    """
    name = str(getattr(obj, "Name", ""))
    label = str(getattr(obj, "Label", ""))
    raw = name or label

    # Glass: Z from stored normal.
    gn = getattr(obj, "GlassNormal", None)
    if gn is not None:
        try:
            z = _normalize3_tuple((float(gn.x), float(gn.y), float(gn.z)))
        except Exception:
            z = None
        if z is not None:
            y = _project_axis_onto_plane((0.0, 1.0, 0.0), z)
            if y is None:
                y = _project_axis_onto_plane((1.0, 0.0, 0.0), z)
            if y is None:
                y = (0.0, 1.0, 0.0)
            x = _normalize3_tuple(_cross_tuple(y, z))
            if x is None:
                x = (1.0, 0.0, 0.0)
            y = _normalize3_tuple(_cross_tuple(z, x)) or y
            return x, y, z

    # Strut: Y from StrutDirection, Z from inner-side normal pointing toward sphere center.
    sd = getattr(obj, "StrutDirection", None)
    if sd is not None or raw.startswith("Strut_"):
        y = None
        if sd is not None:
            try:
                y = _normalize3_tuple((float(sd.x), float(sd.y), float(sd.z)))
            except Exception:
                y = None
        if y is None:
            y = (0.0, 1.0, 0.0)

        # Inward direction: from centroid to origin, projected to be perpendicular to Y.
        c = _object_centroid(obj)
        inward = None
        if c is not None:
            inward = (-c[0], -c[1], -c[2])
        if inward is None:
            inward = (0.0, 0.0, 1.0)
        z = _project_axis_onto_plane(inward, y)
        if z is None:
            # Fallback: any perpendicular to Y.
            z = _project_axis_onto_plane((0.0, 0.0, 1.0), y) or _project_axis_onto_plane((0.0, 1.0, 0.0), y)
        if z is None:
            z = (0.0, 0.0, 1.0)

        # Enforce the user's frame:
        # - y = strut length axis
        # - z = inner-side plane normal (points toward sphere center)
        # - x = normal of the (y,z) plane
        x = _normalize3_tuple(_cross_tuple(y, z))
        if x is None:
            # Degenerate (y || z): choose a z perpendicular to y, then recompute x.
            z2 = _project_axis_onto_plane((0.0, 0.0, 1.0), y)
            if z2 is None:
                z2 = _project_axis_onto_plane((0.0, 1.0, 0.0), y)
            if z2 is None:
                z2 = (0.0, 0.0, 1.0)
            z = z2
            x = _normalize3_tuple(_cross_tuple(y, z))
        if x is None:
            x = (1.0, 0.0, 0.0)

        # Re-orthonormalize z to be perpendicular to y while staying in the (x,z) plane.
        # This keeps y exact, and makes x exact normal.
        z2 = _normalize3_tuple(_cross_tuple(x, y))
        if z2 is not None:
            z = z2
        return x, y, z

    # Fallback: longest bound-box axis as Y, global Z as Z.
    shape = getattr(obj, "Shape", None)
    bb = getattr(shape, "BoundBox", None) if shape is not None else None
    y = None
    if bb is not None:
        try:
            lens = {
                (1.0, 0.0, 0.0): float(bb.XLength),
                (0.0, 1.0, 0.0): float(bb.YLength),
                (0.0, 0.0, 1.0): float(bb.ZLength),
            }
            y = max(lens.items(), key=lambda kv: kv[1])[0]
        except Exception:
            y = None
    if y is None:
        y = (0.0, 1.0, 0.0)
    z = (0.0, 0.0, 1.0)
    x = _normalize3_tuple(_cross_tuple(y, z)) or (1.0, 0.0, 0.0)
    y = _normalize3_tuple(_cross_tuple(z, x)) or y
    return x, y, z


def _project_axis_onto_plane(axis: Tuple[float, float, float], plane_normal: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
    # axis_proj = axis - n * dot(axis,n)
    dn = axis[0] * plane_normal[0] + axis[1] * plane_normal[1] + axis[2] * plane_normal[2]
    proj = (
        axis[0] - plane_normal[0] * dn,
        axis[1] - plane_normal[1] * dn,
        axis[2] - plane_normal[2] * dn,
    )
    return _normalize3_tuple(proj)


def _object_centroid(obj: Any) -> Optional[Tuple[float, float, float]]:
    shape = getattr(obj, "Shape", None)
    if shape is None:
        return None
    try:
        com = getattr(shape, "CenterOfMass", None)
        if com is not None:
            return (float(com.x), float(com.y), float(com.z))
    except Exception:
        pass
    bb = getattr(shape, "BoundBox", None)
    if bb is None:
        return None
    try:
        c = bb.Center
        return (float(c.x), float(c.y), float(c.z))
    except Exception:
        return None


def _cross_tuple(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize3_tuple(v: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
    x, y, z = v
    ln = (x * x + y * y + z * z) ** 0.5
    if ln <= 1e-12:
        return None
    return (x / ln, y / ln, z / ln)


def _export_page_pdf(page: Any, options: TechDrawOptions, key: str) -> str | None:
    if not options.export_dir:
        return None
    out = Path(options.export_dir).expanduser().resolve() / f"{options.page_prefix}{key}.pdf"
    try:
        import TechDrawGui  # type: ignore

        TechDrawGui.exportPageAsPdf(page, str(out))
        return str(out)
    except Exception:
        try:
            # Some versions expose method on page.
            if hasattr(page, "exportPdf"):
                page.exportPdf(str(out))
                return str(out)
        except Exception:
            pass
    logging.warning("PDF export not available in this FreeCAD build")
    return None


def _export_page_dxf(page: Any, options: TechDrawOptions, key: str) -> str | None:
    if not options.export_dir:
        return None
    out = Path(options.export_dir).expanduser().resolve() / f"{options.page_prefix}{key}.dxf"
    try:
        import TechDrawGui  # type: ignore

        if hasattr(TechDrawGui, "exportPageAsDxf"):
            TechDrawGui.exportPageAsDxf(page, str(out))
            return str(out)
    except Exception:
        pass
    logging.warning("DXF export not available in this FreeCAD build")
    return None
