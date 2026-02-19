"""Riser wall (pikendusring) for the geodesic dome.

A cylindrical vertical extension between the dome's belt plane and the
foundation, providing additional headroom and wall area.  The riser sits
directly below the dome's lower rim and transfers loads straight down to
the foundation.

Features
--------
- Parametric height and thickness.
- Material selection (concrete, wood, steel).
- Connection detail at the dome–riser interface (flange, embed, bolted).
- Optional door cutout through the riser wall (reuses door parameters).
- Stud layout for wood-frame risers.
- JSON report with geometry, BOM, and connection metadata.

Usage::

    from freecad_dome.riser_wall import plan_riser_wall, write_riser_report

    plan = plan_riser_wall(dome, params)
    write_riser_report(plan, out_dir)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .parameters import DomeParameters
from .tessellation import TessellatedDome, Vector3

__all__ = [
    "RiserWallPlan",
    "RiserConnection",
    "RiserStud",
    "plan_riser_wall",
    "write_riser_report",
    "create_riser_wall_solids",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RiserConnection:
    """Describes the dome-to-riser connection at one belt node."""

    node_index: int
    x_m: float
    y_m: float
    z_top_m: float           # belt-plane Z (top of riser)
    azimuth_deg: float
    connection_type: str     # "flange" | "embed" | "bolted"


@dataclass(slots=True)
class RiserStud:
    """A single stud in a wood-frame riser wall."""

    azimuth_deg: float
    x_bottom_m: float
    y_bottom_m: float
    z_bottom_m: float
    x_top_m: float
    y_top_m: float
    z_top_m: float
    length_m: float


@dataclass(slots=True)
class RiserDoorCutout:
    """Metadata for the door cutout through the riser wall."""

    door_width_m: float
    door_height_m: float
    door_angle_deg: float
    z_bottom_m: float
    z_top_m: float
    fits_in_riser: bool  # True if door_height <= riser_height


@dataclass(slots=True)
class RiserWallPlan:
    """Complete riser wall layout for a dome."""

    belt_radius_m: float
    belt_height_m: float         # Z of dome belt plane = riser top
    riser_height_m: float
    riser_thickness_m: float
    riser_bottom_z_m: float      # belt_height - riser_height
    riser_top_z_m: float         # = belt_height
    material: str
    connection_type: str
    segments: int

    # Geometry
    outer_radius_m: float
    inner_radius_m: float
    wall_area_m2: float          # outer surface area
    volume_m3: float             # material volume

    connections: List[RiserConnection] = field(default_factory=list)
    studs: List[RiserStud] = field(default_factory=list)
    door_cutout: Optional[RiserDoorCutout] = None

    def summary(self) -> str:
        door_str = ""
        if self.door_cutout is not None:
            fit = "fits" if self.door_cutout.fits_in_riser else "extends above"
            door_str = f", door {self.door_cutout.door_width_m:.2f}×{self.door_cutout.door_height_m:.2f}m ({fit})"
        return (
            f"Riser wall: {self.material}, h={self.riser_height_m:.2f}m, "
            f"r={self.belt_radius_m:.2f}m, t={self.riser_thickness_m:.3f}m, "
            f"{len(self.connections)} connections, {len(self.studs)} studs"
            f"{door_str}"
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "belt_radius_m": round(self.belt_radius_m, 4),
            "belt_height_m": round(self.belt_height_m, 4),
            "riser_height_m": round(self.riser_height_m, 4),
            "riser_thickness_m": round(self.riser_thickness_m, 4),
            "riser_bottom_z_m": round(self.riser_bottom_z_m, 4),
            "riser_top_z_m": round(self.riser_top_z_m, 4),
            "material": self.material,
            "connection_type": self.connection_type,
            "segments": self.segments,
            "outer_radius_m": round(self.outer_radius_m, 4),
            "inner_radius_m": round(self.inner_radius_m, 4),
            "wall_area_m2": round(self.wall_area_m2, 3),
            "volume_m3": round(self.volume_m3, 4),
            "connection_count": len(self.connections),
            "stud_count": len(self.studs),
            "connections": [
                {
                    "node_index": c.node_index,
                    "x_m": round(c.x_m, 4),
                    "y_m": round(c.y_m, 4),
                    "z_top_m": round(c.z_top_m, 4),
                    "azimuth_deg": round(c.azimuth_deg, 2),
                    "connection_type": c.connection_type,
                }
                for c in self.connections
            ],
        }
        if self.studs:
            d["studs"] = [
                {
                    "azimuth_deg": round(s.azimuth_deg, 2),
                    "length_m": round(s.length_m, 4),
                }
                for s in self.studs
            ]
        if self.door_cutout is not None:
            dc = self.door_cutout
            d["door_cutout"] = {
                "door_width_m": round(dc.door_width_m, 4),
                "door_height_m": round(dc.door_height_m, 4),
                "door_angle_deg": round(dc.door_angle_deg, 2),
                "z_bottom_m": round(dc.z_bottom_m, 4),
                "z_top_m": round(dc.z_top_m, 4),
                "fits_in_riser": dc.fits_in_riser,
            }
        return d


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_riser_wall(
    dome: TessellatedDome,
    params: DomeParameters,
) -> RiserWallPlan:
    """Compute the complete riser wall layout.

    Parameters
    ----------
    dome : TessellatedDome
        The tessellated dome with nodes and panels.
    params : DomeParameters
        Must have riser_* fields.

    Returns
    -------
    RiserWallPlan
        Geometry, connections, studs, and optional door cutout.
    """
    R = float(params.radius_m)
    hemi = float(params.hemisphere_ratio)

    belt_z = R * (1.0 - 2.0 * hemi) if hemi < 1.0 else -R
    disc = R * R - belt_z * belt_z
    belt_radius = math.sqrt(max(disc, 0.0))

    riser_h = float(params.riser_height_m)
    riser_t = float(params.riser_thickness_m)
    riser_t = min(riser_t, belt_radius * 0.45)  # clamp

    riser_bottom_z = belt_z - riser_h
    riser_top_z = belt_z

    outer_r = belt_radius
    inner_r = max(0.0, belt_radius - riser_t)

    # Surface area (outer cylinder wall)
    wall_area = 2.0 * math.pi * outer_r * riser_h

    # Volume (hollow cylinder)
    volume = math.pi * (outer_r ** 2 - inner_r ** 2) * riser_h

    material = params.riser_material
    conn_type = params.riser_connection_type
    segments = params.riser_segments

    # --- Connections at belt nodes ---
    connections = _compute_connections(dome, params, belt_z, conn_type)

    # --- Studs for wood riser ---
    studs: List[RiserStud] = []
    if material == "wood":
        studs = _compute_studs(
            belt_radius, riser_t, riser_bottom_z, riser_top_z,
            params.riser_stud_spacing_m,
        )

    # --- Door cutout ---
    door_cutout: Optional[RiserDoorCutout] = None
    if params.riser_door_integration and (params.generate_base_wall or params.generate_entry_porch):
        door_cutout = _compute_door_cutout(params, riser_bottom_z, riser_top_z, riser_h)

    plan = RiserWallPlan(
        belt_radius_m=belt_radius,
        belt_height_m=belt_z,
        riser_height_m=riser_h,
        riser_thickness_m=riser_t,
        riser_bottom_z_m=riser_bottom_z,
        riser_top_z_m=riser_top_z,
        material=material,
        connection_type=conn_type,
        segments=segments,
        outer_radius_m=outer_r,
        inner_radius_m=inner_r,
        wall_area_m2=wall_area,
        volume_m3=volume,
        connections=connections,
        studs=studs,
        door_cutout=door_cutout,
    )

    log.info("Riser wall plan: %s", plan.summary())
    return plan


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_riser_report(plan: RiserWallPlan, out_dir: Path) -> Path:
    """Write riser wall plan as a JSON report file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "riser_wall_plan.json"
    path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    log.info("Wrote riser wall plan %s", path)
    return path


# ---------------------------------------------------------------------------
# FreeCAD 3-D geometry
# ---------------------------------------------------------------------------

_FC_SCALE: float = 1000.0  # metres → mm


def create_riser_wall_solids(
    plan: RiserWallPlan,
    params: DomeParameters,
    document: Any = None,
) -> Optional[Any]:
    """Create FreeCAD Part::Feature solids for the riser wall.

    Creates a hollow cylindrical wall with optional door cutout and optional
    stud geometry for wood-frame risers.  All objects are placed under an
    ``App::DocumentObjectGroup`` named ``"RiserWall"``.

    Parameters
    ----------
    plan : RiserWallPlan
        Output of :func:`plan_riser_wall`.
    params : DomeParameters
        Dome parameters (used for door dimensions).
    document : optional
        Existing FreeCAD document.  If *None*, uses ``FreeCAD.ActiveDocument``
        or creates a new one.

    Returns
    -------
    The FreeCAD group object, or *None* on failure / missing FreeCAD.
    """
    try:
        import FreeCAD  # type: ignore
        import Part  # type: ignore
        from FreeCAD import Vector  # type: ignore
    except ImportError:
        return None

    doc = document
    if doc is None:
        doc = FreeCAD.ActiveDocument
    if doc is None:
        doc = FreeCAD.newDocument("GeodesicDome")

    def _mm(m: float) -> float:
        return float(m) * _FC_SCALE

    outer_r = _mm(plan.outer_radius_m)
    inner_r = _mm(plan.inner_radius_m)
    h = _mm(plan.riser_height_m)

    if outer_r <= 0 or inner_r <= 0 or h <= 0:
        return None

    # Hollow cylindrical shell --------------------------------------------------
    outer = Part.makeCylinder(outer_r, h)
    inner = Part.makeCylinder(inner_r, h)
    shell = outer.cut(inner)

    # Door cutout ---------------------------------------------------------------
    if plan.door_cutout is not None:
        dc = plan.door_cutout
        door_w = _mm(dc.door_width_m)
        door_h = _mm(min(dc.door_height_m, plan.riser_height_m))
        door_clear = _mm(getattr(params, "door_clearance_m", 0.005))
        door_w = max(1.0, door_w + door_clear * 2.0)
        door_h = max(1.0, door_h + door_clear * 2.0)

        wall_t = _mm(plan.riser_thickness_m)
        radial_len = wall_t * 4.0

        x0 = inner_r - wall_t  # extend past inner face
        y0 = -door_w / 2.0
        z0 = 0.0

        door_box = Part.makeBox(radial_len, door_w, door_h)
        door_box.Placement = FreeCAD.Placement(
            Vector(x0, y0, z0), FreeCAD.Rotation()
        )

        ang_deg = float(dc.door_angle_deg)
        rot = FreeCAD.Rotation(Vector(0, 0, 1), ang_deg)
        door_box.Placement = FreeCAD.Placement(
            Vector(0, 0, 0), rot
        ).multiply(door_box.Placement)

        shell = shell.cut(door_box)

    # Translate shell to correct Z position ------------------------------------
    bottom_z = _mm(plan.riser_bottom_z_m)
    shell.Placement = FreeCAD.Placement(
        Vector(0.0, 0.0, bottom_z), FreeCAD.Rotation()
    )

    # Add to document ----------------------------------------------------------
    obj = doc.addObject("Part::Feature", "RiserWall")
    obj.Label = "RiserWall"
    obj.Shape = shell

    try:
        if hasattr(obj, "addProperty") and not hasattr(obj, "IfcType"):
            obj.addProperty("App::PropertyString", "IfcType", "BIM", "IFC type")
        if hasattr(obj, "IfcType"):
            obj.IfcType = "IfcWall"
    except Exception:
        pass

    # Visual appearance --------------------------------------------------------
    _MATERIAL_COLORS = {
        "concrete": (0.75, 0.75, 0.72, 0.0),
        "wood":     (0.82, 0.63, 0.35, 0.0),
        "steel":    (0.60, 0.62, 0.65, 0.0),
    }
    try:
        color = _MATERIAL_COLORS.get(plan.material, (0.75, 0.75, 0.72, 0.0))
        obj.ViewObject.ShapeColor = color
    except Exception:
        pass

    created_objects: List[Any] = [obj]

    # Studs for wood-frame riser -----------------------------------------------
    if plan.studs:
        stud_w = _mm(max(plan.riser_thickness_m * 0.3, 0.038))  # typical 38 mm
        stud_d = _mm(max(plan.riser_thickness_m * 0.6, 0.089))  # typical 89 mm
        for i, stud in enumerate(plan.studs):
            s_len = _mm(stud.length_m)
            if s_len <= 0:
                continue
            s_box = Part.makeBox(stud_d, stud_w, s_len)
            sx = _mm(stud.x_bottom_m) - stud_d / 2.0
            sy = _mm(stud.y_bottom_m) - stud_w / 2.0
            sz = _mm(stud.z_bottom_m)
            az_rad = math.radians(stud.azimuth_deg)
            s_box.Placement = FreeCAD.Placement(
                Vector(sx, sy, sz),
                FreeCAD.Rotation(Vector(0, 0, 1), stud.azimuth_deg),
            )
            s_obj = doc.addObject("Part::Feature", f"RiserStud_{i:03d}")
            s_obj.Label = f"RiserStud_{i:03d}"
            s_obj.Shape = s_box
            try:
                s_obj.ViewObject.ShapeColor = (0.82, 0.63, 0.35, 0.0)
            except Exception:
                pass
            created_objects.append(s_obj)

    # Group all objects --------------------------------------------------------
    grp = None
    try:
        for o in list(getattr(doc, "Objects", []) or []):
            if (
                str(getattr(o, "Name", "")) == "RiserWall_Group"
                and str(getattr(o, "TypeId", "")) == "App::DocumentObjectGroup"
            ):
                grp = o
                break
        if grp is None:
            grp = doc.addObject("App::DocumentObjectGroup", "RiserWall_Group")
            grp.Label = "RiserWall"
        for co in created_objects:
            grp.addObject(co)
    except Exception:
        pass

    try:
        doc.recompute()
    except Exception:
        pass

    log.info("Created riser wall solids: %d objects", len(created_objects))
    return grp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _belt_node_indices(
    dome: TessellatedDome,
    belt_z: float,
    tolerance: float = 1e-4,
) -> List[int]:
    """Return indices of nodes lying on the belt plane, sorted by azimuth."""
    indices = [
        i for i, n in enumerate(dome.nodes)
        if abs(n[2] - belt_z) <= tolerance
    ]
    indices.sort(key=lambda i: math.atan2(dome.nodes[i][1], dome.nodes[i][0]))
    return indices


def _compute_connections(
    dome: TessellatedDome,
    params: DomeParameters,
    belt_z: float,
    connection_type: str,
) -> List[RiserConnection]:
    """Create a RiserConnection for every belt-plane node."""
    belt_indices = _belt_node_indices(dome, belt_z)
    connections: List[RiserConnection] = []
    for idx in belt_indices:
        nx, ny, nz = dome.nodes[idx]
        azimuth = math.degrees(math.atan2(ny, nx)) % 360.0
        connections.append(RiserConnection(
            node_index=idx,
            x_m=nx,
            y_m=ny,
            z_top_m=nz,
            azimuth_deg=azimuth,
            connection_type=connection_type,
        ))
    return connections


def _compute_studs(
    belt_radius: float,
    thickness: float,
    z_bottom: float,
    z_top: float,
    spacing: float,
) -> List[RiserStud]:
    """Compute stud positions for a wood-frame riser wall.

    Studs are evenly spaced around the circumference at the midline
    of the wall thickness.
    """
    if spacing <= 0 or belt_radius <= 0:
        return []

    circumference = 2.0 * math.pi * belt_radius
    n_studs = max(1, round(circumference / spacing))
    actual_spacing_deg = 360.0 / n_studs

    mid_r = belt_radius - thickness * 0.5
    length = z_top - z_bottom

    studs: List[RiserStud] = []
    for i in range(n_studs):
        az_deg = i * actual_spacing_deg
        az_rad = math.radians(az_deg)
        x = mid_r * math.cos(az_rad)
        y = mid_r * math.sin(az_rad)
        studs.append(RiserStud(
            azimuth_deg=az_deg,
            x_bottom_m=x,
            y_bottom_m=y,
            z_bottom_m=z_bottom,
            x_top_m=x,
            y_top_m=y,
            z_top_m=z_top,
            length_m=length,
        ))
    return studs


def _compute_door_cutout(
    params: DomeParameters,
    riser_bottom_z: float,
    riser_top_z: float,
    riser_height: float,
) -> RiserDoorCutout:
    """Compute the door cutout geometry through the riser wall."""
    door_w = float(params.door_width_m)
    door_h = float(params.door_height_m)
    door_angle = float(params.door_angle_deg)

    # Door starts at the bottom of the riser and extends upward.
    door_bottom = riser_bottom_z
    door_top = riser_bottom_z + door_h
    fits = door_h <= riser_height

    return RiserDoorCutout(
        door_width_m=door_w,
        door_height_m=door_h,
        door_angle_deg=door_angle,
        z_bottom_m=door_bottom,
        z_top_m=door_top,
        fits_in_riser=fits,
    )
