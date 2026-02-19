"""Base wall (knee wall) generator for a vertical door.

Variant 1 approach:
- Generate a cylindrical wall whose top matches the dome belt plane.
- Cut a vertical rectangular door opening through the wall.

All params are stored in meters; FreeCAD geometry is authored in mm (x1000).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .parameters import DomeParameters
from .tessellation import TessellatedDome

__all__ = ["BaseWallResult", "BaseWallBuilder", "suggest_door_angle_deg"]


@dataclass(slots=True)
class BaseWallResult:
    name: str
    wall_radius_m: float
    belt_height_m: float
    wall_bottom_z_m: float
    wall_top_z_m: float


class BaseWallBuilder:
    def __init__(self, params: DomeParameters):
        self.params = params
        self._doc: Optional[Any] = None
        self._fc_unit_scale: float = 1000.0

    @property
    def document(self) -> Optional[Any]:
        return self._doc

    def _fc_len(self, meters: float) -> float:
        return float(meters) * float(self._fc_unit_scale)

    def ensure_document(self) -> Optional[Any]:
        try:
            import FreeCAD  # type: ignore
        except ImportError:  # pragma: no cover
            return None
        doc = FreeCAD.ActiveDocument
        if doc is None:
            doc = FreeCAD.newDocument("GeodesicDome")
        return doc

    def create_base_wall(self) -> BaseWallResult | None:
        if not self.params.generate_base_wall:
            return None

        doc = self.ensure_document()
        self._doc = doc
        if doc is None:
            return None

        try:
            import FreeCAD  # type: ignore
            import Part  # type: ignore
            from FreeCAD import Vector  # type: ignore
        except Exception:  # pragma: no cover
            return None

        R = float(self.params.radius_m)
        hemi = float(self.params.hemisphere_ratio)
        belt_height_m = R * (1.0 - 2.0 * hemi) if hemi < 1.0 else -R

        # Dome horizontal radius at the belt plane.
        disc = R * R - belt_height_m * belt_height_m
        if disc <= 1e-12:
            return None
        wall_radius_m = float(math.sqrt(disc))

        wall_h_m = float(self.params.base_wall_height_m)
        wall_t_m = float(self.params.base_wall_thickness_m)
        if wall_h_m <= 1e-6 or wall_t_m <= 1e-6:
            return None

        # Clamp thickness to be <= radius.
        wall_t_m = min(wall_t_m, wall_radius_m * 0.45)

        wall_top_z_m = belt_height_m
        wall_bottom_z_m = wall_top_z_m - wall_h_m

        outer_r = self._fc_len(wall_radius_m)
        inner_r = self._fc_len(max(1e-6, wall_radius_m - wall_t_m))
        h = self._fc_len(wall_h_m)

        # FreeCAD cylinder starts at Z=0 along +Z; place it at wall bottom.
        outer = Part.makeCylinder(outer_r, h)
        inner = Part.makeCylinder(inner_r, h)
        shell = outer.cut(inner)

        # Door cutout box (vertical), oriented radially at door_angle.
        door_w_m = float(self.params.door_width_m)
        door_h_m = float(self.params.door_height_m)
        door_clear_m = float(self.params.door_clearance_m)
        ang_deg = float(self.params.door_angle_deg)

        # Ensure the door height fits within the wall.
        door_h_m = min(door_h_m + door_clear_m * 2.0, wall_h_m)
        door_w_m = max(0.05, door_w_m + door_clear_m * 2.0)

        radial_len_m = wall_t_m * 4.0
        radial_len = self._fc_len(radial_len_m)
        door_w = self._fc_len(door_w_m)
        door_h = self._fc_len(door_h_m)

        # Local frame (before rotation):
        # X = radial outward, Y = tangential, Z = vertical.
        # Place the box so it fully intersects the wall thickness.
        x0 = self._fc_len(wall_radius_m - wall_t_m - door_clear_m)
        y0 = -door_w / 2.0
        z0 = 0.0  # relative to wall bottom
        door_box = Part.makeBox(radial_len, door_w, door_h)
        door_box.Placement = FreeCAD.Placement(Vector(x0, y0, z0), FreeCAD.Rotation())

        rot = FreeCAD.Rotation(Vector(0, 0, 1), float(ang_deg))
        wall_place = FreeCAD.Placement(Vector(0.0, 0.0, self._fc_len(wall_bottom_z_m)), rot)
        shell.Placement = wall_place

        # Door needs the same global placement (rotation + translation).
        door_box.Placement = wall_place.multiply(door_box.Placement)

        wall_with_door = shell.cut(door_box)

        obj = doc.addObject("Part::Feature", "BaseWall")
        obj.Label = "BaseWall"
        obj.Shape = wall_with_door

        # Try to set IFC type if Arch/BIM is available.
        try:
            if hasattr(obj, "addProperty") and not hasattr(obj, "IfcType"):
                obj.addProperty("App::PropertyString", "IfcType", "BIM", "IFC type")
            if hasattr(obj, "IfcType"):
                obj.IfcType = "IfcWall"
        except Exception:
            pass

        # Grouping for convenience.
        try:
            grp = None
            for o in list(getattr(doc, "Objects", []) or []):
                if str(getattr(o, "Name", "")) == "Base" and str(getattr(o, "TypeId", "")) == "App::DocumentObjectGroup":
                    grp = o
                    break
            if grp is None:
                grp = doc.addObject("App::DocumentObjectGroup", "Base")
                grp.Label = "Base"
            grp.addObject(obj)
        except Exception:
            pass

        try:
            doc.recompute()
        except Exception:
            pass

        return BaseWallResult(
            name=str(getattr(obj, "Name", "BaseWall")),
            wall_radius_m=wall_radius_m,
            belt_height_m=belt_height_m,
            wall_bottom_z_m=wall_bottom_z_m,
            wall_top_z_m=wall_top_z_m,
        )


def suggest_door_angle_deg(params: DomeParameters, dome: TessellatedDome) -> float | None:
    """Suggest a door azimuth angle for a symmetric dome.

    Since the dome is rotationally symmetric for practical purposes, any angle is valid.
    This helper chooses a deterministic, aesthetically reasonable direction:
    - find panels touching the belt plane
    - pick the lowest-index such panel
    - point the door toward that panel's centroid in XY

    Returns degrees in [0, 360).
    """
    if dome is None or not getattr(dome, "panels", None) or not getattr(dome, "nodes", None):
        return 0.0

    R = float(params.radius_m)
    hemi = float(params.hemisphere_ratio)

    # Primary preference: align the door centerline (vertical axis in the door plane)
    # with the midpoint of a side of the top (apex) pentagon.
    #
    # Important detail: the apex pentagon has 5 sides, thus 5 valid midpoints.
    # To avoid a "random" choice (e.g. by longest edge jitter), we pick the side whose
    # midpoint direction is closest to the user's preferred azimuth (params.door_angle_deg).
    #
    # In this project's coordinate frame, the door centerline points along the local
    # radial axis (local +X) after rotation by ang_deg about +Z, i.e. (cos(a), sin(a)).
    # We choose a so that this direction points to the edge midpoint (projected onto XY).
    try:
        preferred_deg = float(params.door_angle_deg) % 360.0

        def _ang_dist_deg(a: float, b: float) -> float:
            # Smallest absolute difference on a circle.
            d = (a - b + 180.0) % 360.0 - 180.0
            return abs(d)

        top_pent = None
        top_z = None
        for p in dome.panels:
            node_ids = list(getattr(p, "node_indices", ()) or ())
            if len(node_ids) != 5:
                continue
            zs = 0.0
            cnt = 0
            for ni in node_ids:
                try:
                    _x, _y, z = dome.nodes[int(ni)]
                except Exception:
                    continue
                zs += float(z)
                cnt += 1
            if cnt <= 0:
                continue
            cz = zs / cnt
            if top_z is None or cz > top_z:
                top_z = cz
                top_pent = p

        if top_pent is not None:
            node_ids = list(getattr(top_pent, "node_indices", ()) or ())

            candidates: list[float] = []
            for i in range(len(node_ids)):
                a = int(node_ids[i])
                b = int(node_ids[(i + 1) % len(node_ids)])
                try:
                    ax, ay, _az = dome.nodes[a]
                    bx, by, _bz = dome.nodes[b]
                except Exception:
                    continue
                mx = (float(ax) + float(bx)) * 0.5
                my = (float(ay) + float(by)) * 0.5
                if abs(mx) <= 1e-12 and abs(my) <= 1e-12:
                    continue
                candidates.append(float(math.degrees(math.atan2(my, mx)) % 360.0))

            if candidates:
                # Deterministic: sort by distance to preferred, then by angle.
                candidates.sort(key=lambda a: (_ang_dist_deg(a, preferred_deg), a))
                return float(candidates[0])
    except Exception:
        pass

    if hemi >= 0.999999:
        # No belt plane; fall back to a stable direction.
        return 0.0

    belt_height = R * (1.0 - 2.0 * hemi)
    eps = max(1e-6, R * 1e-5)

    def _touches_belt(panel) -> bool:
        for ni in getattr(panel, "node_indices", ()) or ():
            try:
                z = float(dome.nodes[int(ni)][2])
            except Exception:
                continue
            if abs(z - belt_height) <= eps:
                return True
        return False

    chosen = None

    # Fallback: prefer a pentagon aligned near the belt plane.
    # Some configurations may have no pentagon that directly touches the belt plane,
    # so we choose the pentagon whose centroid Z is closest to the belt height
    # (and on/above the belt plane).
    best_score = None
    for p in dome.panels:
        try:
            node_ids = list(getattr(p, "node_indices", ()) or ())
        except Exception:
            node_ids = []
        if len(node_ids) != 5:
            continue
        xs = ys = zs = 0.0
        cnt = 0
        for ni in node_ids:
            try:
                x, y, z = dome.nodes[int(ni)]
            except Exception:
                continue
            xs += float(x)
            ys += float(y)
            zs += float(z)
            cnt += 1
        if cnt <= 0:
            continue
        cz = zs / cnt
        if cz < belt_height - eps:
            continue
        score = abs(cz - belt_height)
        if best_score is None or score < best_score:
            best_score = score
            chosen = p

    # Fallback: first belt-touching panel.
    if chosen is None:
        for p in dome.panels:
            if _touches_belt(p):
                chosen = p
                break
    if chosen is None:
        return 0.0

    xs = ys = 0.0
    cnt = 0
    for ni in getattr(chosen, "node_indices", ()) or ():
        try:
            x, y, _z = dome.nodes[int(ni)]
        except Exception:
            continue
        xs += float(x)
        ys += float(y)
        cnt += 1
    if cnt <= 0:
        return 0.0
    cx = xs / cnt
    cy = ys / cnt
    if abs(cx) <= 1e-12 and abs(cy) <= 1e-12:
        return 0.0

    ang = math.degrees(math.atan2(cy, cx))
    # Normalize to [0, 360)
    ang = ang % 360.0
    return float(ang)
