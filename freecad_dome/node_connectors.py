"""Node connector (hub) geometry for the geodesic dome.

Each node in a geodesic dome is a meeting point of 5–7 struts.  This module
generates the **connector hardware** required to join them:

- **plate** — a circular/polygonal gusset plate with bolt holes, suitable for
  wood or steel struts.  The plate sits at the node, oriented along the
  sphere's radial direction.  Each incident strut gets one bolt hole at a
  configurable offset from center.
- **ball**  — a spherical hub with threaded inserts (pipe-frame domes).
- **pipe**  — a star-shaped pipe connector stub (pipe-frame domes).

For the MVP only the *plate* type is fully implemented, matching the existing
wood-strut workflow.

Typical usage::

    from freecad_dome.node_connectors import NodeConnectorBuilder
    builder = NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .parameters import DomeParameters
from .tessellation import TessellatedDome, Vector3
from . import vec3 as v3

__all__ = [
    "BoltPosition",
    "NodeConnector",
    "NodeConnectorBuilder",
    "build_incident_map",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector algebra helpers — thin wrappers for back-compat within this module
# ---------------------------------------------------------------------------

_norm = v3.norm
_normalize = v3.normalize
_dot = v3.dot
_cross = v3.cross
_sub = v3.sub
_add = v3.add
_scale = v3.scale
_angle_between = v3.angle_between


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BoltPosition:
    """One bolt hole on a node connector plate."""
    center: Vector3          # 3D position of bolt hole center (meters)
    axis: Vector3            # Bolt axis direction (= plate normal, outward)
    strut_index: int         # Which strut this bolt connects
    strut_end: str           # "start" or "end"
    direction_in_plate: Vector3  # Unit vector from plate center toward bolt, in the tangent plane


@dataclass(slots=True)
class NodeConnector:
    """Describes a single node hub/connector."""
    node_index: int
    position: Vector3                # dome.nodes[node_index]
    normal: Vector3                  # outward radial direction (plate normal)
    connector_type: str              # "plate" | "ball" | "pipe"
    valence: int                     # number of incident struts
    plate_radius_m: float            # radius of the connector plate
    plate_thickness_m: float         # thickness of the plate
    bolt_positions: List[BoltPosition]
    is_belt_node: bool               # True for nodes on the hemisphere belt plane
    angular_order: List[int]         # strut indices in clockwise angular order around the node

    def bolt_circle_radius_m(self) -> float:
        """Radius of the circle on which bolt holes are placed."""
        if not self.bolt_positions:
            return 0.0
        return max(
            _norm(_sub(bp.center, self.position))
            for bp in self.bolt_positions
        )

    def min_strut_angle_deg(self) -> float:
        """Smallest angle between adjacent struts at this node."""
        if self.valence < 2:
            return 360.0
        angles = []
        for i in range(self.valence):
            a = self.bolt_positions[i].direction_in_plate
            b = self.bolt_positions[(i + 1) % self.valence].direction_in_plate
            angles.append(math.degrees(_angle_between(a, b)))
        return min(angles) if angles else 360.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "node_index": self.node_index,
            "position": [round(c, 6) for c in self.position],
            "normal": [round(c, 6) for c in self.normal],
            "connector_type": self.connector_type,
            "valence": self.valence,
            "plate_radius_m": round(self.plate_radius_m, 6),
            "plate_thickness_m": round(self.plate_thickness_m, 6),
            "is_belt_node": self.is_belt_node,
            "min_strut_angle_deg": round(self.min_strut_angle_deg(), 2),
            "bolt_count": len(self.bolt_positions),
            "bolts": [
                {
                    "strut_index": bp.strut_index,
                    "strut_end": bp.strut_end,
                    "center": [round(c, 6) for c in bp.center],
                }
                for bp in self.bolt_positions
            ],
        }


# ---------------------------------------------------------------------------
# Incident map — same pattern as struts.py
# ---------------------------------------------------------------------------

IncidentEntry = Tuple[Any, str, Vector3]  # (Strut, end_label, direction_from_node)


def build_incident_map(
    dome: TessellatedDome,
) -> Dict[int, List[IncidentEntry]]:
    """Build a mapping from node_index → list of (strut, end_label, dir_from_node).

    This reproduces the core pattern from ``StrutBuilder._compute_endpoint_cut_planes``
    so the node-connector module can operate independently without a StrutBuilder.
    """
    incident: Dict[int, List[IncidentEntry]] = {}
    for s in dome.struts:
        sx, sy, sz = s.start
        ex, ey, ez = s.end
        incident.setdefault(s.start_index, []).append(
            (s, "start", (ex - sx, ey - sy, ez - sz))
        )
        incident.setdefault(s.end_index, []).append(
            (s, "end", (sx - ex, sy - ey, sz - ez))
        )
    return incident


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class NodeConnectorBuilder:
    """Creates node connector geometry and metadata for every dome node.

    Parameters
    ----------
    params : DomeParameters
        Must include ``node_connector_*`` fields.
    """

    def __init__(self, params: DomeParameters) -> None:
        self.params = params
        self.document: Any = None  # Set if FreeCAD objects are created.

    # ---- public API -------------------------------------------------------

    def create_connectors(
        self,
        dome: TessellatedDome,
        node_fit_data: Any = None,
    ) -> List[NodeConnector]:
        """Compute connector metadata for every node in the dome.

        Parameters
        ----------
        dome : TessellatedDome
            Tessellated dome geometry.
        node_fit_data : NodeFitData, optional
            Pre-computed shared node-fit data. When provided, the connector
            builder reuses the incident map and angular ordering computed by
            the strut builder rather than re-deriving them.
        """
        incident = build_incident_map(dome)
        connectors: List[NodeConnector] = []

        connector_type = self.params.node_connector_type
        plate_thickness = self.params.node_connector_thickness_m
        bolt_diameter = self.params.node_connector_bolt_diameter_m
        bolt_offset = self.params.node_connector_bolt_offset_m

        # Belt height for detecting belt nodes.
        belt_height: float | None = None
        belt_eps = max(1e-6, self.params.radius_m * 1e-5)
        if self.params.hemisphere_ratio < 1.0:
            belt_height = self.params.radius_m * (
                1.0 - 2.0 * self.params.hemisphere_ratio
            )

        for node_idx in range(len(dome.nodes)):
            items = incident.get(node_idx)
            if not items:
                continue

            p = dome.nodes[node_idx]
            is_belt = False
            if belt_height is not None:
                is_belt = abs(p[2] - belt_height) <= belt_eps

            # Reuse shared node-fit data when available.
            nfi = node_fit_data.get(node_idx) if node_fit_data is not None else None

            if nfi is not None:
                # Use pre-computed radial, angular order, and tangent dirs.
                radial = nfi.radial
                is_belt = nfi.is_belt_node

                # Build bolt positions from shared data.
                bolt_positions: List[BoltPosition] = []
                angular_order: List[int] = list(nfi.angular_order)

                for s, end_label, d in items:
                    t = nfi.tangent_directions.get(s.index)
                    if t is None:
                        continue
                    bolt_center = _add(p, _scale(t, bolt_offset))
                    bolt_positions.append(BoltPosition(
                        center=bolt_center,
                        axis=radial,
                        strut_index=s.index,
                        strut_end=end_label,
                        direction_in_plate=t,
                    ))

                # Plate radius: bolt offset + bolt-hole clearance + margin.
                plate_radius = bolt_offset + bolt_diameter * 0.5 + 0.005  # 5 mm margin

                connectors.append(NodeConnector(
                    node_index=node_idx,
                    position=p,
                    normal=radial,
                    connector_type=connector_type,
                    valence=len(bolt_positions),
                    plate_radius_m=plate_radius,
                    plate_thickness_m=plate_thickness,
                    bolt_positions=bolt_positions,
                    is_belt_node=is_belt,
                    angular_order=angular_order,
                ))
                continue

            # Fallback: compute locally (when node_fit_data not supplied).
            # Radial normal (outward from dome center).
            radial = _normalize(p)
            if _norm(radial) <= 1e-12:
                continue
            if is_belt:
                radial = (0.0, 0.0, 1.0)

            # Tangent-plane basis.
            ref = (1.0, 0.0, 0.0) if abs(radial[0]) < 0.9 else (0.0, 1.0, 0.0)
            u = _normalize(_cross(radial, ref))
            v = _normalize(_cross(radial, u))

            # Compute angular position of each strut in the tangent plane.
            enriched: List[Tuple[float, Any, str, Vector3, Vector3]] = []
            for s, end_label, d in items:
                a = _normalize(d)
                # Project onto tangent plane.
                t = _sub(a, _scale(radial, _dot(a, radial)))
                t = _normalize(t)
                if _norm(t) <= 1e-12:
                    if is_belt:
                        t = _normalize((a[0], a[1], 0.0))
                        if _norm(t) <= 1e-12:
                            t = u
                    else:
                        t = u
                ang = math.atan2(_dot(t, v), _dot(t, u))
                enriched.append((ang, s, end_label, a, t))

            if not enriched:
                continue

            # Sort by angle (clockwise around normal).
            enriched.sort(key=lambda row: row[0])

            # Build bolt positions.
            bolt_positions: List[BoltPosition] = []
            angular_order: List[int] = []

            for _ang, s, end_label, _a, t in enriched:
                # Bolt center: offset from node along the tangent-projected strut direction.
                bolt_center = _add(p, _scale(t, bolt_offset))
                bolt_positions.append(BoltPosition(
                    center=bolt_center,
                    axis=radial,
                    strut_index=s.index,
                    strut_end=end_label,
                    direction_in_plate=t,
                ))
                angular_order.append(s.index)

            # Plate radius: bolt offset + bolt-hole clearance + margin.
            plate_radius = bolt_offset + bolt_diameter * 0.5 + 0.005  # 5 mm margin

            connectors.append(NodeConnector(
                node_index=node_idx,
                position=p,
                normal=radial,
                connector_type=connector_type,
                valence=len(enriched),
                plate_radius_m=plate_radius,
                plate_thickness_m=plate_thickness,
                bolt_positions=bolt_positions,
                is_belt_node=is_belt,
                angular_order=angular_order,
            ))

        log.info(
            "Created %d node connectors (type=%s, valences: %s)",
            len(connectors),
            connector_type,
            _valence_histogram(connectors),
        )
        return connectors

    def create_connector_solids(
        self,
        connectors: List[NodeConnector],
    ) -> List[Any]:
        """Create FreeCAD Part objects for connectors.

        Dispatches to the appropriate solid builder based on ``connector_type``.
        Requires FreeCAD to be available.  Returns the created Feature objects.
        """
        try:
            import FreeCAD  # type: ignore
            import Part  # type: ignore
        except ImportError:
            log.warning("FreeCAD not available; skipping connector solid creation")
            return []

        if self.document is None:
            self.document = FreeCAD.ActiveDocument or FreeCAD.newDocument("NodeConnectors")

        scale = 1000.0  # m → mm
        objects: List[Any] = []

        for nc in connectors:
            name = f"Hub_N{nc.node_index}"
            try:
                if nc.connector_type == "ball":
                    solid = self._make_ball_solid(nc, scale)
                elif nc.connector_type == "pipe":
                    solid = self._make_pipe_solid(nc, scale)
                elif nc.connector_type == "lapjoint":
                    solid = self._make_lapjoint_solid(nc, scale)
                else:
                    solid = self._make_plate_solid(nc, scale)
                if solid is None:
                    continue
                obj = self.document.addObject("Part::Feature", name)
                obj.Shape = solid
                objects.append(obj)
            except Exception as exc:
                log.warning("Failed to create solid for %s: %s", name, exc)

        self.document.recompute()
        log.info("Created %d connector solids in FreeCAD", len(objects))
        return objects

    # ---- private ----------------------------------------------------------

    def _make_plate_solid(self, nc: NodeConnector, scale: float) -> Any:
        """Build a plate connector as a FreeCAD Shape (cylinder with bolt holes)."""
        import FreeCAD  # type: ignore
        import Part  # type: ignore

        cx, cy, cz = [c * scale for c in nc.position]
        nx, ny, nz = nc.normal
        r_mm = nc.plate_radius_m * scale
        t_mm = nc.plate_thickness_m * scale
        bolt_r_mm = self.params.node_connector_bolt_diameter_m * 0.5 * scale

        center = FreeCAD.Vector(cx, cy, cz)
        axis = FreeCAD.Vector(nx, ny, nz)

        # Offset the plate so it is centered on the node (half-thickness each side).
        base = center - axis * (t_mm * 0.5)
        plate = Part.makeCylinder(r_mm, t_mm, base, axis)

        # Drill bolt holes.
        for bp in nc.bolt_positions:
            bx, by, bz = [c * scale for c in bp.center]
            hole_base = FreeCAD.Vector(bx, by, bz) - axis * (t_mm * 0.5 + 1.0)
            hole = Part.makeCylinder(bolt_r_mm, t_mm + 2.0, hole_base, axis)
            plate = plate.cut(hole)

        return plate

    def _make_ball_solid(self, nc: NodeConnector, scale: float) -> Any:
        """Build a ball (sphere) connector with threaded insert stubs.

        A sphere is centred on the node. For each incident strut, a short
        cylindrical stub is subtracted (to form a socket) or added (to form a
        tenon) — here we add a through-hole along each strut axis so that a
        threaded rod can pass through.
        """
        import FreeCAD  # type: ignore
        import Part  # type: ignore

        cx, cy, cz = [c * scale for c in nc.position]
        center = FreeCAD.Vector(cx, cy, cz)

        # Ball radius: large enough to enclose all bolt positions + margin.
        ball_r_mm = nc.plate_radius_m * scale * 1.1
        # Ensure minimum radius relative to connector thickness.
        min_r = nc.plate_thickness_m * scale * 2.0
        ball_r_mm = max(ball_r_mm, min_r)

        ball = Part.makeSphere(ball_r_mm, center)

        # Drill a hole along each strut direction for threaded rod inserts.
        bolt_r_mm = self.params.node_connector_bolt_diameter_m * 0.5 * scale
        hole_depth = ball_r_mm * 2.5  # extend well past the ball on both sides

        for bp in nc.bolt_positions:
            d = bp.direction_in_plate  # tangent direction toward strut
            d_len = _norm(d)
            if d_len < 1e-12:
                continue
            d_unit = _normalize(d)
            ax = FreeCAD.Vector(*d_unit)
            hole_base = center - ax * hole_depth * 0.5
            hole = Part.makeCylinder(bolt_r_mm, hole_depth, hole_base, ax)
            ball = ball.cut(hole)

        return ball

    def _make_pipe_solid(self, nc: NodeConnector, scale: float) -> Any:
        """Build a pipe/star connector — short pipe stubs radiating from the node.

        Each stub is a hollow cylinder aimed along the strut direction, meeting
        at the node center. This yields a welded-pipe-star connector style.
        """
        import FreeCAD  # type: ignore
        import Part  # type: ignore

        cx, cy, cz = [c * scale for c in nc.position]
        center = FreeCAD.Vector(cx, cy, cz)

        t_mm = nc.plate_thickness_m * scale  # wall thickness of the pipe
        # Outer radius matches half the stock width (strut fits inside).
        outer_r_mm = max(
            self.params.stock_width_m * 0.5 * scale,
            self.params.stock_height_m * 0.5 * scale,
        ) + t_mm
        inner_r_mm = outer_r_mm - t_mm
        stub_len_mm = nc.plate_radius_m * scale * 1.5  # extension length

        parts: list[Any] = []
        for bp in nc.bolt_positions:
            d = bp.direction_in_plate
            d_len = _norm(d)
            if d_len < 1e-12:
                continue
            d_unit = _normalize(d)
            ax = FreeCAD.Vector(*d_unit)
            # Stub starts at centre, extends outward.
            outer = Part.makeCylinder(outer_r_mm, stub_len_mm, center, ax)
            inner = Part.makeCylinder(inner_r_mm, stub_len_mm + 2.0, center - ax * 1.0, ax)
            stub = outer.cut(inner)
            parts.append(stub)

        if not parts:
            return None
        # Fuse all stubs into one shape.
        result = parts[0]
        for p in parts[1:]:
            result = result.fuse(p)
        return result

    def _make_lapjoint_solid(self, nc: NodeConnector, scale: float) -> Any:
        """Build a lap-joint connector visualization.

        In a lap joint, each strut end extends past the node and rests against
        the side face of its clockwise neighbour.  This method creates a small
        representative solid at the node showing:
        - A gusset plate (thin triangular web) between each adjacent pair of struts
        - A bolt hole through the overlap region

        The actual strut extension is handled by the strut builder (negative inset
        when ``node_connector_type == "lapjoint"``).  This solid is purely for
        visualization and assembly reference.
        """
        import FreeCAD  # type: ignore
        import Part  # type: ignore

        cx, cy, cz = [c * scale for c in nc.position]
        center = FreeCAD.Vector(cx, cy, cz)

        bolt_r_mm = self.params.node_connector_bolt_diameter_m * 0.5 * scale
        t_mm = nc.plate_thickness_m * scale
        lap_ext = getattr(self.params, "node_connector_lap_extension_m", 0.03) * scale

        # Collect tangent directions in angular order.
        bps = nc.bolt_positions
        if len(bps) < 2:
            return None

        parts: list[Any] = []
        for i, bp in enumerate(bps):
            bp_next = bps[(i + 1) % len(bps)]
            d1 = _normalize(bp.direction_in_plate)
            d2 = _normalize(bp_next.direction_in_plate)
            d1_len = _norm(d1)
            d2_len = _norm(d2)
            if d1_len < 1e-12 or d2_len < 1e-12:
                continue

            ax1 = FreeCAD.Vector(*d1)
            ax2 = FreeCAD.Vector(*d2)
            normal = FreeCAD.Vector(*nc.normal)

            # Gusset triangle: node center → extension along strut i → extension along strut i+1
            p0 = center
            p1 = center + ax1 * lap_ext
            p2 = center + ax2 * lap_ext

            try:
                wire = Part.makePolygon([p0, p1, p2, p0])
                face = Part.Face(wire)
                gusset = face.extrude(normal * t_mm)
                parts.append(gusset)
            except Exception:
                continue

            # Bolt hole at the midpoint of the overlap region.
            bolt_center = center + (ax1 + ax2) * (lap_ext * 0.5)
            hole_base = bolt_center - normal * (t_mm * 0.5 + 1.0)
            hole = Part.makeCylinder(bolt_r_mm, t_mm + 2.0, hole_base, normal)
            if parts:
                parts[-1] = parts[-1].cut(hole)

        if not parts:
            return None
        result = parts[0]
        for p in parts[1:]:
            result = result.fuse(p)
        return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_connector_report(
    connectors: List[NodeConnector],
    path: Any,  # Path-like
) -> None:
    """Write a JSON summary of all node connectors."""
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    valences: Dict[int, int] = {}
    for nc in connectors:
        valences[nc.valence] = valences.get(nc.valence, 0) + 1

    report = {
        "total_connectors": len(connectors),
        "valence_histogram": {str(k): v for k, v in sorted(valences.items())},
        "unique_types": len(set(nc.connector_type for nc in connectors)),
        "belt_nodes": sum(1 for nc in connectors if nc.is_belt_node),
        "min_strut_angle_deg": round(
            min((nc.min_strut_angle_deg() for nc in connectors), default=0.0), 2
        ),
        "connectors": [nc.to_dict() for nc in connectors],
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote connector report to %s", out)


def connector_bom_rows(
    connectors: List[NodeConnector],
    params: DomeParameters,
) -> List[Dict[str, Any]]:
    """Generate BOM (bill of materials) rows for connector hardware.

    Returns one row per connector plus aggregated bolt/washer rows.
    """
    rows: List[Dict[str, Any]] = []

    # Group connectors by valence (each valence = a unique plate template).
    by_valence: Dict[int, List[NodeConnector]] = {}
    for nc in connectors:
        by_valence.setdefault(nc.valence, []).append(nc)

    for valence, group in sorted(by_valence.items()):
        representative = group[0]
        rows.append({
            "item": f"Gusset plate V{valence}",
            "type": params.node_connector_type,
            "quantity": len(group),
            "diameter_m": round(representative.plate_radius_m * 2, 4),
            "thickness_m": round(representative.plate_thickness_m, 4),
            "bolt_holes_per_plate": valence,
            "material": "steel",
        })

    # Total bolts, nuts, washers.
    total_bolts = sum(nc.valence for nc in connectors)
    rows.append({
        "item": f"Bolt M{int(params.node_connector_bolt_diameter_m * 1000)}",
        "type": "fastener",
        "quantity": total_bolts,
        "diameter_m": round(params.node_connector_bolt_diameter_m, 4),
        "length_m": round(params.node_connector_bolt_length_m, 4),
        "material": "steel",
    })
    rows.append({
        "item": f"Nut M{int(params.node_connector_bolt_diameter_m * 1000)}",
        "type": "fastener",
        "quantity": total_bolts,
        "material": "steel",
    })
    rows.append({
        "item": f"Washer M{int(params.node_connector_bolt_diameter_m * 1000)}",
        "type": "fastener",
        "quantity": total_bolts * 2,  # one per side
        "diameter_m": round(params.node_connector_washer_diameter_m, 4),
        "material": "steel",
    })

    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valence_histogram(connectors: List[NodeConnector]) -> str:
    """Compact string like '5×6, 6×60, 7×4'."""
    hist: Dict[int, int] = {}
    for nc in connectors:
        hist[nc.valence] = hist.get(nc.valence, 0) + 1
    return ", ".join(f"{v}×{cnt}" for v, cnt in sorted(hist.items()))
