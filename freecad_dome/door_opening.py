"""Door opening cutting + reinforcement frame.

Applies when using the entry porch: we need a real opening in the dome shell.

Scope (minimal):
- Choose door azimuth via params.door_angle_deg.
- Cut a vertical rectangular prism through:
  - glass panels (GlassPanel_*)
  - panel faces/frames if present (Panel_*, PanelFrame_*)
  - strut base geometry (Strut_*_Geom)
- Add a simple rectangular reinforcement frame behind the cut edge.

All inputs are stored in meters; FreeCAD geometry is authored in mm (x1000).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

from .parameters import DomeParameters
from .tessellation import TessellatedDome

__all__ = ["DoorOpeningResult", "apply_door_opening"]


@dataclass(slots=True)
class DoorOpeningResult:
    cutter_name: str
    frame_name: str
    angle_deg: float


def apply_door_opening(
    doc: Any,
    *,
    params: DomeParameters,
    dome: TessellatedDome,
    include_glass: bool = True,
    include_panels: bool = True,
    include_struts: bool = True,
) -> DoorOpeningResult | None:
    """Cut a door opening through generated geometry and add reinforcement frame."""

    if doc is None:
        return None

    try:
        import FreeCAD  # type: ignore
        import Part  # type: ignore
        from FreeCAD import Vector  # type: ignore
    except Exception:  # pragma: no cover
        return None

    scale = 1000.0

    R = float(params.radius_m)
    hemi = float(params.hemisphere_ratio)
    if hemi >= 0.999999:
        return None

    belt_height_m = R * (1.0 - 2.0 * hemi)
    disc = R * R - belt_height_m * belt_height_m
    if disc <= 1e-12:
        return None
    belt_radius_m = float(math.sqrt(disc))

    door_w_m = float(params.door_width_m)
    door_h_m = float(params.door_height_m)
    door_clear_m = float(params.door_clearance_m)
    ang_deg = float(params.door_angle_deg)

    # Door starts at belt plane.
    z0_m = float(belt_height_m)

    # Cutter dimensions.
    cut_w_m = max(0.05, door_w_m + 2.0 * door_clear_m)
    cut_h_m = max(0.05, door_h_m + 2.0 * door_clear_m)

    # Cut depth: go inward far enough to remove intersecting struts/panels, and a bit outside.
    inside_depth_m = max(0.25, R)  # to (at least) the sphere center direction
    outside_depth_m = 0.6
    cut_x_m = inside_depth_m + outside_depth_m

    cut_x = cut_x_m * scale
    cut_y = cut_w_m * scale
    cut_z = cut_h_m * scale

    # Local frame (before rotation):
    # X = radial outward, Y = tangential, Z = vertical.
    # Place the box centered on the door axis in Y.
    x0 = (belt_radius_m - inside_depth_m) * scale
    y0 = -cut_y / 2.0
    z0 = (z0_m - door_clear_m) * scale

    cutter = Part.makeBox(cut_x, cut_y, cut_z)
    cutter.Placement = FreeCAD.Placement(Vector(float(x0), float(y0), float(z0)), FreeCAD.Rotation())

    rot = FreeCAD.Rotation(Vector(0, 0, 1), float(ang_deg))
    cutter.Placement = FreeCAD.Placement(Vector(0.0, 0.0, 0.0), rot).multiply(cutter.Placement)

    # Keep a visible reference (hidden by default) for debugging.
    cutter_obj = doc.addObject("Part::Feature", "DoorCutter")
    cutter_obj.Label = "DoorCutter"
    cutter_obj.Shape = cutter
    try:
        cutter_vo = getattr(cutter_obj, "ViewObject", None)
        if cutter_vo is not None:
            cutter_vo.Visibility = False
    except Exception:
        pass

    def _should_cut(o: Any) -> bool:
        n = str(getattr(o, "Name", ""))
        tid = str(getattr(o, "TypeId", ""))
        if "Group" in tid:
            return False
        if not hasattr(o, "Shape"):
            return False
        if include_glass and n.startswith("GlassPanel_"):
            return True
        if include_panels and (n.startswith("Panel_") or n.startswith("PanelFrame_")):
            return True
        if include_struts and n.startswith("Strut_") and n.endswith("_Geom"):
            return True
        return False

    cut_count = 0
    for o in list(getattr(doc, "Objects", []) or []):
        if not _should_cut(o):
            continue
        try:
            shape = getattr(o, "Shape", None)
            if shape is None:
                continue
            new_shape = shape.cut(cutter)
            o.Shape = new_shape
            cut_count += 1
        except Exception:
            continue

    # Reinforcement frame behind the cut edge (simple rectangle): jambs + header.
    member_m = float(params.porch_member_size_m)
    msz = member_m * scale

    frame_x = (belt_radius_m - (member_m * 1.2)) * scale  # slightly inside the surface
    door_w = door_w_m * scale
    door_h = door_h_m * scale

    def _box(len_x: float, len_y: float, len_z: float, px: float, py: float, pz: float):
        b = Part.makeBox(len_x, len_y, len_z)
        b.Placement = FreeCAD.Placement(Vector(float(px), float(py), float(pz)), FreeCAD.Rotation())
        return b

    jamb_left = _box(msz, msz, door_h, frame_x - msz, -door_w / 2.0, z0_m * scale)
    jamb_right = _box(msz, msz, door_h, frame_x - msz, door_w / 2.0 - msz, z0_m * scale)
    header = _box(msz, door_w, msz, frame_x - msz, -door_w / 2.0, z0_m * scale + door_h - msz)

    frame_shape = Part.makeCompound([jamb_left, jamb_right, header])
    frame_shape.Placement = FreeCAD.Placement(Vector(0.0, 0.0, 0.0), rot)

    frame_obj = doc.addObject("Part::Feature", "DoorOpeningFrame")
    frame_obj.Label = "DoorOpeningFrame"
    frame_obj.Shape = frame_shape

    # Group under Base for convenience.
    try:
        grp = None
        for oo in list(getattr(doc, "Objects", []) or []):
            if str(getattr(oo, "Name", "")) == "Base" and str(getattr(oo, "TypeId", "")) == "App::DocumentObjectGroup":
                grp = oo
                break
        if grp is None:
            grp = doc.addObject("App::DocumentObjectGroup", "Base")
            grp.Label = "Base"
        grp.addObject(frame_obj)
        grp.addObject(cutter_obj)
    except Exception:
        pass

    try:
        doc.recompute()
    except Exception:
        pass

    return DoorOpeningResult(
        cutter_name=str(getattr(cutter_obj, "Name", "DoorCutter")),
        frame_name=str(getattr(frame_obj, "Name", "DoorOpeningFrame")),
        angle_deg=float(ang_deg),
    )
