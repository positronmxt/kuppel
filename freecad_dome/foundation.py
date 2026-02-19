"""Foundation layout system for the geodesic dome.

Generates anchor positions, foundation geometry metadata, and a concrete
pour-plan report for constructing the base on which the dome sits.

Supported foundation types
--------------------------

- **strip** — continuous ring footing that follows the belt-plane circle.
  The footing width & depth are parameters, and anchor bolts are placed
  at every belt node position projected onto the ring.
- **point** — individual pier/pad footings under each belt node.  Cheaper
  for smaller domes on firm soil.
- **screw_anchor** — helical screw piles under each belt node.  No
  excavation required; suited to uneven or soft ground.

All types share the same anchor-bolt interface: each belt-edge node gets
one anchor bolt whose 2-D position (X, Y) and azimuth angle are exported
so the builder can set bolts before the concrete cures (strip / point) or
attach brackets (screw_anchor).

Usage::

    from freecad_dome.foundation import (
        foundation_for_params,
        write_foundation_report,
    )

    plan = foundation_for_params(dome, params)
    write_foundation_report(plan, params, "exports/foundation_report.json")
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .parameters import DomeParameters

__all__ = [
    "AnchorBolt",
    "FoundationPlan",
    "foundation_for_params",
    "create_foundation_solids",
    "write_foundation_report",
]

log = logging.getLogger(__name__)

Vector3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AnchorBolt:
    """Single anchor bolt at a belt node position."""

    node_index: int
    x_m: float                # 2-D position on the foundation plane
    y_m: float
    z_m: float                # top-of-bolt elevation (belt height)
    azimuth_deg: float        # angle from +X axis (0–360)
    diameter_m: float = 0.016 # M16 default
    embed_depth_m: float = 0.20  # how deep the bolt sits in concrete
    protrusion_m: float = 0.10   # how far above concrete surface

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_index": self.node_index,
            "x_m": round(self.x_m, 4),
            "y_m": round(self.y_m, 4),
            "z_m": round(self.z_m, 4),
            "azimuth_deg": round(self.azimuth_deg, 2),
            "diameter_m": self.diameter_m,
            "embed_depth_m": self.embed_depth_m,
            "protrusion_m": self.protrusion_m,
        }


@dataclass(slots=True)
class FoundationPlan:
    """Complete foundation layout for the dome."""

    foundation_type: str           # "strip" | "point" | "screw_anchor"
    belt_radius_m: float           # horizontal radius of belt circle
    belt_height_m: float           # Z elevation of the belt plane
    anchors: List[AnchorBolt] = field(default_factory=list)

    # Strip-specific
    strip_width_m: float = 0.30   # footing width
    strip_depth_m: float = 0.40   # footing depth below grade
    strip_top_m: float = 0.10     # top of strip above grade (for moisture)

    # Point-specific
    pier_diameter_m: float = 0.30
    pier_depth_m: float = 0.60

    # Screw-anchor-specific
    screw_shaft_diameter_m: float = 0.076  # standard 76 mm shaft
    screw_helix_diameter_m: float = 0.200
    screw_length_m: float = 1.5

    # Drainage
    drainage_slope_pct: float = 2.0    # % slope away from dome
    drainage_apron_m: float = 0.60     # concrete/gravel apron width
    waterproof_layer: str = "bitumen"  # "bitumen" | "membrane" | "none"

    @property
    def anchor_count(self) -> int:
        return len(self.anchors)

    @property
    def circumference_m(self) -> float:
        """Belt circle circumference."""
        return 2.0 * math.pi * self.belt_radius_m

    @property
    def concrete_volume_m3(self) -> float:
        """Approximate concrete volume for this foundation type."""
        if self.foundation_type == "strip":
            total_depth = self.strip_depth_m + self.strip_top_m
            return self.circumference_m * self.strip_width_m * total_depth
        elif self.foundation_type == "point":
            # Cylindrical pier per anchor
            area = math.pi * (self.pier_diameter_m / 2.0) ** 2
            return area * self.pier_depth_m * self.anchor_count
        else:
            return 0.0  # screw anchors need no concrete

    def anchor_spacing_m(self) -> float:
        """Average distance between adjacent anchors around the belt."""
        if self.anchor_count < 2:
            return 0.0
        return self.circumference_m / self.anchor_count

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "foundation_type": self.foundation_type,
            "belt_radius_m": round(self.belt_radius_m, 4),
            "belt_height_m": round(self.belt_height_m, 4),
            "circumference_m": round(self.circumference_m, 3),
            "anchor_count": self.anchor_count,
            "anchor_spacing_m": round(self.anchor_spacing_m(), 3),
            "drainage_slope_pct": self.drainage_slope_pct,
            "drainage_apron_m": self.drainage_apron_m,
            "waterproof_layer": self.waterproof_layer,
        }
        if self.foundation_type == "strip":
            d["strip_width_m"] = self.strip_width_m
            d["strip_depth_m"] = self.strip_depth_m
            d["strip_top_m"] = self.strip_top_m
            d["concrete_volume_m3"] = round(self.concrete_volume_m3, 3)
        elif self.foundation_type == "point":
            d["pier_diameter_m"] = self.pier_diameter_m
            d["pier_depth_m"] = self.pier_depth_m
            d["concrete_volume_m3"] = round(self.concrete_volume_m3, 3)
        elif self.foundation_type == "screw_anchor":
            d["screw_shaft_diameter_m"] = self.screw_shaft_diameter_m
            d["screw_helix_diameter_m"] = self.screw_helix_diameter_m
            d["screw_length_m"] = self.screw_length_m
        return d


# ---------------------------------------------------------------------------
# Belt node detection
# ---------------------------------------------------------------------------

def _belt_node_indices(
    nodes: List[Vector3],
    radius_m: float,
    hemisphere_ratio: float,
    tolerance: float = 1e-4,
) -> List[int]:
    """Return indices of nodes lying on the belt plane, sorted by azimuth."""
    if hemisphere_ratio >= 1.0:
        # Full sphere — no belt.
        return []
    belt_z = radius_m * (1.0 - 2.0 * hemisphere_ratio)
    indices = [i for i, n in enumerate(nodes) if abs(n[2] - belt_z) <= tolerance]
    # Sort by azimuth angle for consistent ordering.
    indices.sort(key=lambda i: math.atan2(nodes[i][1], nodes[i][0]))
    return indices


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def foundation_for_params(dome: Any, params: DomeParameters) -> FoundationPlan:
    """Build a FoundationPlan from the tessellated dome and parameters.

    Parameters
    ----------
    dome : TessellatedDome
        The tessellated dome with nodes, struts, panels.
    params : DomeParameters
        Current configuration (must have foundation_* fields).
    """
    R = float(params.radius_m)
    hemi = float(params.hemisphere_ratio)

    belt_z = R * (1.0 - 2.0 * hemi) if hemi < 1.0 else -R
    disc = R * R - belt_z * belt_z
    belt_radius = math.sqrt(max(disc, 0.0))

    ftype = getattr(params, "foundation_type", "strip")
    bolt_dia = getattr(params, "foundation_bolt_diameter_m", 0.016)
    bolt_embed = getattr(params, "foundation_bolt_embed_m", 0.20)
    bolt_protrude = getattr(params, "foundation_bolt_protrusion_m", 0.10)
    strip_w = getattr(params, "foundation_strip_width_m", 0.30)
    strip_d = getattr(params, "foundation_strip_depth_m", 0.40)
    pier_dia = getattr(params, "foundation_pier_diameter_m", 0.30)
    pier_d = getattr(params, "foundation_pier_depth_m", 0.60)

    belt_indices = _belt_node_indices(dome.nodes, R, hemi)

    anchors: List[AnchorBolt] = []
    for idx in belt_indices:
        nx, ny, nz = dome.nodes[idx]
        azimuth = math.degrees(math.atan2(ny, nx)) % 360.0
        anchors.append(AnchorBolt(
            node_index=idx,
            x_m=nx,
            y_m=ny,
            z_m=nz,
            azimuth_deg=azimuth,
            diameter_m=bolt_dia,
            embed_depth_m=bolt_embed,
            protrusion_m=bolt_protrude,
        ))

    plan = FoundationPlan(
        foundation_type=ftype,
        belt_radius_m=belt_radius,
        belt_height_m=belt_z,
        anchors=anchors,
        strip_width_m=strip_w,
        strip_depth_m=strip_d,
        pier_diameter_m=pier_dia,
        pier_depth_m=pier_d,
    )

    log.info(
        "Foundation plan: %s, %d anchors, belt R=%.3f m at z=%.3f m",
        ftype, len(anchors), belt_radius, belt_z,
    )
    return plan


# ---------------------------------------------------------------------------
# FreeCAD 3-D geometry
# ---------------------------------------------------------------------------

def create_foundation_solids(
    plan: FoundationPlan,
    params: DomeParameters,
    document: Any = None,
) -> list[Any]:
    """Create FreeCAD 3-D solids representing the foundation.

    Returns the created Part::Feature objects (strip footing / piers / anchors).
    Falls back gracefully when FreeCAD is not available.

    Generated geometry
    ------------------
    - **strip** — a toroidal (annular) solid centred on the belt circle.
    - **point** — individual cylinder piers under each anchor.
    - **screw_anchor** — symbolic cylinder shafts at each anchor.
    - *All types* additionally get small cylinder stubs for anchor bolts.
    """
    try:
        import FreeCAD  # type: ignore
        import Part  # type: ignore
    except ImportError:
        log.warning("FreeCAD not available; skipping foundation solid creation")
        return []

    doc = document
    if doc is None:
        doc = FreeCAD.ActiveDocument or FreeCAD.newDocument("Foundation")

    scale = 1000.0  # metres → mm
    objects: list[Any] = []

    # Helper: add a Part::Feature and collect it. -------------------------
    def _add(name: str, shape: Any) -> Any:
        obj = doc.addObject("Part::Feature", name)
        obj.Shape = shape
        objects.append(obj)
        return obj

    belt_r_mm = plan.belt_radius_m * scale
    belt_z_mm = plan.belt_height_m * scale

    # ------------------------------------------------------------------
    # 1. Main foundation body
    # ------------------------------------------------------------------
    if plan.foundation_type == "strip":
        # Annular ring footing built by revolving a rectangular profile
        # around the Z axis.
        w_mm = plan.strip_width_m * scale
        total_depth_mm = (plan.strip_depth_m + plan.strip_top_m) * scale
        top_offset_mm = plan.strip_top_m * scale

        # Profile rectangle: centred on (belt_r, belt_z - depth + top_offset).
        half_w = w_mm / 2.0
        profile_z_bottom = belt_z_mm - total_depth_mm + top_offset_mm
        # The profile sits in the XZ plane for revolution around Z.
        p1 = FreeCAD.Vector(belt_r_mm - half_w, 0, profile_z_bottom)
        p2 = FreeCAD.Vector(belt_r_mm + half_w, 0, profile_z_bottom)
        p3 = FreeCAD.Vector(belt_r_mm + half_w, 0, profile_z_bottom + total_depth_mm)
        p4 = FreeCAD.Vector(belt_r_mm - half_w, 0, profile_z_bottom + total_depth_mm)

        wire = Part.makePolygon([p1, p2, p3, p4, p1])
        face = Part.Face(wire)
        strip = face.revolve(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 360)
        obj = _add("Foundation_Strip", strip)
        # Concrete colour.
        if hasattr(obj, "ViewObject") and obj.ViewObject:
            try:
                obj.ViewObject.ShapeColor = (0.75, 0.75, 0.72, 0.0)
            except Exception:
                pass

    elif plan.foundation_type == "point":
        for i, ab in enumerate(plan.anchors):
            cx = ab.x_m * scale
            cy = ab.y_m * scale
            cz = (ab.z_m - plan.pier_depth_m) * scale
            r_mm = (plan.pier_diameter_m / 2.0) * scale
            h_mm = plan.pier_depth_m * scale
            pier = Part.makeCylinder(
                r_mm, h_mm,
                FreeCAD.Vector(cx, cy, cz),
                FreeCAD.Vector(0, 0, 1),
            )
            obj = _add(f"Foundation_Pier_{i:03d}", pier)
            if hasattr(obj, "ViewObject") and obj.ViewObject:
                try:
                    obj.ViewObject.ShapeColor = (0.75, 0.75, 0.72, 0.0)
                except Exception:
                    pass

    elif plan.foundation_type == "screw_anchor":
        for i, ab in enumerate(plan.anchors):
            cx = ab.x_m * scale
            cy = ab.y_m * scale
            cz = (ab.z_m - plan.screw_length_m) * scale
            shaft_r_mm = (plan.screw_shaft_diameter_m / 2.0) * scale
            shaft_h_mm = plan.screw_length_m * scale
            shaft = Part.makeCylinder(
                shaft_r_mm, shaft_h_mm,
                FreeCAD.Vector(cx, cy, cz),
                FreeCAD.Vector(0, 0, 1),
            )
            # Add a symbolic helix disc at the bottom.
            helix_r_mm = (plan.screw_helix_diameter_m / 2.0) * scale
            helix_h_mm = 10.0  # 10 mm symbolic disc
            helix = Part.makeCylinder(
                helix_r_mm, helix_h_mm,
                FreeCAD.Vector(cx, cy, cz),
                FreeCAD.Vector(0, 0, 1),
            )
            combined = shaft.fuse(helix)
            obj = _add(f"Foundation_Screw_{i:03d}", combined)
            if hasattr(obj, "ViewObject") and obj.ViewObject:
                try:
                    obj.ViewObject.ShapeColor = (0.5, 0.5, 0.5, 0.0)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 2. Anchor bolts (all types)
    # ------------------------------------------------------------------
    for i, ab in enumerate(plan.anchors):
        cx = ab.x_m * scale
        cy = ab.y_m * scale
        bolt_r_mm = (ab.diameter_m / 2.0) * scale
        # Bolt extends from embed depth below belt to protrusion above.
        bolt_bottom_z = (ab.z_m - ab.embed_depth_m) * scale
        bolt_h_mm = (ab.embed_depth_m + ab.protrusion_m) * scale
        bolt = Part.makeCylinder(
            bolt_r_mm, bolt_h_mm,
            FreeCAD.Vector(cx, cy, bolt_bottom_z),
            FreeCAD.Vector(0, 0, 1),
        )
        obj = _add(f"AnchorBolt_{i:03d}", bolt)
        if hasattr(obj, "ViewObject") and obj.ViewObject:
            try:
                obj.ViewObject.ShapeColor = (0.3, 0.3, 0.3, 0.0)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------
    try:
        grp = None
        for o in list(getattr(doc, "Objects", []) or []):
            if (
                str(getattr(o, "Name", "")) == "Foundation"
                and str(getattr(o, "TypeId", "")) == "App::DocumentObjectGroup"
            ):
                grp = o
                break
        if grp is None:
            grp = doc.addObject("App::DocumentObjectGroup", "Foundation")
            grp.Label = "Foundation"
        for obj in objects:
            grp.addObject(obj)
    except Exception:
        pass

    try:
        doc.recompute()
    except Exception:
        pass

    log.info("Created %d foundation solids in FreeCAD", len(objects))
    return objects


# ---------------------------------------------------------------------------
# BOM helpers
# ---------------------------------------------------------------------------

def foundation_bom_rows(plan: FoundationPlan) -> List[Dict[str, Any]]:
    """Generate BOM rows for the foundation."""
    rows: List[Dict[str, Any]] = []

    if plan.foundation_type in ("strip", "point"):
        rows.append({
            "item": f"Concrete ({plan.foundation_type} foundation)",
            "type": "concrete",
            "volume_m3": round(plan.concrete_volume_m3, 3),
            "note": "C25/30 or equivalent",
        })

    rows.append({
        "item": f"Anchor bolts M{int(plan.anchors[0].diameter_m * 1000)}" if plan.anchors else "Anchor bolts",
        "type": "anchor_bolt",
        "quantity": plan.anchor_count,
        "diameter_m": plan.anchors[0].diameter_m if plan.anchors else 0.016,
        "embed_depth_m": plan.anchors[0].embed_depth_m if plan.anchors else 0.20,
    })

    if plan.waterproof_layer != "none":
        # Approximate waterproofing area: strip surface area or pier tops.
        if plan.foundation_type == "strip":
            wp_area = plan.circumference_m * plan.strip_width_m
        elif plan.foundation_type == "point":
            wp_area = plan.anchor_count * math.pi * (plan.pier_diameter_m / 2) ** 2
        else:
            wp_area = 0.0
        if wp_area > 0:
            rows.append({
                "item": f"Waterproofing ({plan.waterproof_layer})",
                "type": "waterproofing",
                "area_m2": round(wp_area, 2),
            })

    if plan.foundation_type == "screw_anchor":
        rows.append({
            "item": "Screw anchors (helical piles)",
            "type": "screw_anchor",
            "quantity": plan.anchor_count,
            "shaft_diameter_m": plan.screw_shaft_diameter_m,
            "helix_diameter_m": plan.screw_helix_diameter_m,
            "length_m": plan.screw_length_m,
        })

    return rows


def pour_plan_coordinates(plan: FoundationPlan) -> List[Dict[str, Any]]:
    """Return anchor bolt coordinates formatted for a pour plan / layout drawing.

    Each entry has polar (azimuth, radius) and cartesian (x, y) coordinates
    suitable for marking on the construction site.
    """
    coords: List[Dict[str, Any]] = []
    for i, ab in enumerate(plan.anchors):
        coords.append({
            "bolt_number": i + 1,
            "node_index": ab.node_index,
            "x_m": round(ab.x_m, 4),
            "y_m": round(ab.y_m, 4),
            "radius_m": round(math.sqrt(ab.x_m ** 2 + ab.y_m ** 2), 4),
            "azimuth_deg": round(ab.azimuth_deg, 2),
            "bolt_diameter_m": ab.diameter_m,
        })
    return coords


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_foundation_report(
    plan: FoundationPlan,
    params: DomeParameters,
    path: Any,
) -> None:
    """Write a JSON report with foundation layout, BOM, and pour plan."""
    bom = foundation_bom_rows(plan)
    pour = pour_plan_coordinates(plan)

    report = {
        "foundation": plan.to_dict(),
        "bom": bom,
        "pour_plan": {
            "bolt_count": plan.anchor_count,
            "belt_radius_m": round(plan.belt_radius_m, 4),
            "coordinates": pour,
        },
        "notes": [
            f"Drainage slope: {plan.drainage_slope_pct}% away from dome for min {plan.drainage_apron_m} m",
            f"Waterproofing: {plan.waterproof_layer} layer on top of foundation",
            f"Ground level reference: z = {round(plan.belt_height_m, 4)} m (belt plane)",
        ],
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote foundation report to %s", out)
