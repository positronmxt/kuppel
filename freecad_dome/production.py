"""Production drawings and manufacturing data for geodesic dome components.

Generates:
- Strut cut-list with bevel/miter angles at each end
- Saw bench settings table (grouped by strut type)
- Node connector plate outlines (DXF/SVG)
- Assembly sequence plan (construction stages)
- Detail metadata for node connections, covering attachment, base joints
- PDF-ready report (standalone, not FreeCAD TechDraw dependent)

All geometry is computed from the tessellated dome model; no FreeCAD
dependency is required.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .parameters import DomeParameters
from .tessellation import Strut, TessellatedDome, Vector3

__all__ = [
    "StrutCutSpec",
    "SawTableRow",
    "NodePlateOutline",
    "AssemblyStage",
    "ProductionPack",
    "compute_strut_cuts",
    "compute_saw_table",
    "compute_node_plates",
    "compute_assembly_stages",
    "production_for_params",
    "write_production_report",
    "write_saw_table_csv",
    "write_node_plate_dxf",
    "write_node_plate_svg",
]

# ---------------------------------------------------------------------------
# Vector helpers (no NumPy dependency)
# ---------------------------------------------------------------------------

def _norm(v: Vector3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: Vector3) -> Vector3:
    n = _norm(v)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _angle_between(a: Vector3, b: Vector3) -> float:
    """Angle in degrees between two vectors."""
    na, nb = _norm(a), _norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cos_a = max(-1.0, min(1.0, _dot(a, b) / (na * nb)))
    return math.degrees(math.acos(cos_a))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StrutCutSpec:
    """Manufacturing specification for a single strut."""
    strut_index: int
    group: str          # length-group label (e.g. "A", "B", "C")
    raw_length_mm: float
    net_length_mm: float   # after kerf allowance
    stock_width_mm: float
    stock_height_mm: float
    start_miter_deg: float    # miter angle at start end (0 = square cut)
    end_miter_deg: float      # miter angle at end
    start_bevel_deg: float    # bevel (tilt) angle at start end
    end_bevel_deg: float      # bevel (tilt) angle at end
    start_node_index: int
    end_node_index: int
    quantity: int = 1


@dataclass(slots=True)
class SawTableRow:
    """Summary row for the saw bench settings table — one per unique strut type."""
    group: str
    count: int
    raw_length_mm: float
    net_length_mm: float
    start_miter_deg: float
    end_miter_deg: float
    start_bevel_deg: float
    end_bevel_deg: float
    stock_width_mm: float
    stock_height_mm: float


@dataclass(slots=True)
class NodePlateOutline:
    """2D outline for a CNC-cut connector plate at a dome node."""
    node_index: int
    valence: int            # number of struts meeting at this node
    radius_mm: float        # plate radius (circular)
    bolt_hole_positions: List[Tuple[float, float]]  # (x, y) in plate-local coords, mm
    bolt_diameter_mm: float
    outline_points: List[Tuple[float, float]]  # polygon approximation of plate edge


@dataclass(slots=True)
class AssemblyStage:
    """One construction stage for the dome assembly sequence."""
    stage_number: int
    name: str
    description: str
    strut_indices: List[int]
    node_indices: List[int]


@dataclass
class ProductionPack:
    """Complete production data for the dome."""
    strut_cuts: List[StrutCutSpec]
    saw_table: List[SawTableRow]
    node_plates: List[NodePlateOutline]
    assembly_stages: List[AssemblyStage]
    total_raw_length_m: float
    total_struts: int
    unique_types: int


# ---------------------------------------------------------------------------
# Strut cut specification
# ---------------------------------------------------------------------------

def _compute_end_angles(
    strut: Strut,
    nodes: List[Vector3],
    radius_m: float,
) -> Tuple[float, float, float, float]:
    """Compute miter and bevel angles for both ends of a strut.

    Miter = rotation in the plane of the strut and the radial vector (saw
    blade angle).  Bevel = out-of-plane tilt (compound miter).

    Returns (start_miter, end_miter, start_bevel, end_bevel) in degrees.
    """
    direction = _normalize(strut.direction)

    def _end_angles(node_pos: Vector3) -> Tuple[float, float]:
        radial = _normalize(node_pos)
        if _norm(radial) < 1e-12:
            return (0.0, 0.0)
        # The ideal cut plane is tangent to the sphere at the node, i.e.
        # its normal is the radial vector.  The miter angle is the deviation
        # of the strut axis from perpendicular to that plane:
        #   miter = 90° - angle(direction, radial)
        axis_angle = _angle_between(direction, radial)
        miter = abs(90.0 - axis_angle)

        # Bevel: angular component out of the strut-radial plane.
        # Project the radial vector onto the plane perpendicular to the strut
        # to get the bevel contribution.
        radial_proj = _sub(radial, (_dot(radial, direction) * direction[0],
                                     _dot(radial, direction) * direction[1],
                                     _dot(radial, direction) * direction[2]))
        # Use primary_normal as the "up" reference for the strut cross-section
        if strut.primary_normal and _norm(strut.primary_normal) > 1e-12:
            pn = _normalize(strut.primary_normal)
            rp_norm = _norm(radial_proj)
            if rp_norm > 1e-12:
                radial_proj_n = _normalize(radial_proj)
                cos_bevel = max(-1.0, min(1.0, _dot(radial_proj_n, pn)))
                bevel = math.degrees(math.acos(cos_bevel))
                # Bevel is the deviation from the primary normal
                bevel = min(bevel, 180.0 - bevel)
            else:
                bevel = 0.0
        else:
            bevel = 0.0
        return (round(miter, 2), round(bevel, 2))

    start_miter, start_bevel = _end_angles(nodes[strut.start_index])
    end_miter, end_bevel = _end_angles(nodes[strut.end_index])
    return (start_miter, end_miter, start_bevel, end_bevel)


def _group_label_for_length(length_mm: float, bins: Dict[float, str]) -> str:
    """Find the group label for a given length, using tolerance binning."""
    tol = 0.5  # mm
    for ref_len, label in bins.items():
        if abs(length_mm - ref_len) < tol:
            return label
    return "?"


def compute_strut_cuts(
    dome: TessellatedDome,
    params: DomeParameters,
) -> List[StrutCutSpec]:
    """Generate cut specifications for every strut in the dome."""
    stock_w_mm = params.stock_width_m * 1000.0
    stock_h_mm = params.stock_height_m * 1000.0
    kerf_mm = params.kerf_m * 1000.0

    # Build length-group bins from unique strut lengths (tolerance 0.5 mm).
    length_groups: Dict[float, str] = {}
    sorted_struts = sorted(dome.struts, key=lambda s: s.length)
    group_idx = 0

    for s in sorted_struts:
        l_mm = s.length * 1000.0
        found = False
        for ref_l in length_groups:
            if abs(l_mm - ref_l) < 0.5:
                found = True
                break
        if not found:
            label = chr(ord("A") + group_idx) if group_idx < 26 else f"G{group_idx}"
            length_groups[l_mm] = label
            group_idx += 1

    cuts: List[StrutCutSpec] = []

    for strut in dome.struts:
        l_mm = strut.length * 1000.0
        start_miter, end_miter, start_bevel, end_bevel = _compute_end_angles(
            strut, dome.nodes, params.radius_m
        )
        group = _group_label_for_length(l_mm, length_groups)
        cuts.append(StrutCutSpec(
            strut_index=strut.index,
            group=group,
            raw_length_mm=round(l_mm, 2),
            net_length_mm=round(l_mm - 2 * kerf_mm, 2),
            stock_width_mm=stock_w_mm,
            stock_height_mm=stock_h_mm,
            start_miter_deg=start_miter,
            end_miter_deg=end_miter,
            start_bevel_deg=start_bevel,
            end_bevel_deg=end_bevel,
            start_node_index=strut.start_index,
            end_node_index=strut.end_index,
        ))

    return cuts


# ---------------------------------------------------------------------------
# Saw bench settings table
# ---------------------------------------------------------------------------

def compute_saw_table(cuts: List[StrutCutSpec]) -> List[SawTableRow]:
    """Aggregate strut cuts into a saw settings summary, one row per type."""
    groups: Dict[str, List[StrutCutSpec]] = {}
    for c in cuts:
        groups.setdefault(c.group, []).append(c)

    rows: List[SawTableRow] = []
    for group in sorted(groups.keys()):
        members = groups[group]
        rep = members[0]
        rows.append(SawTableRow(
            group=group,
            count=len(members),
            raw_length_mm=rep.raw_length_mm,
            net_length_mm=rep.net_length_mm,
            start_miter_deg=rep.start_miter_deg,
            end_miter_deg=rep.end_miter_deg,
            start_bevel_deg=rep.start_bevel_deg,
            end_bevel_deg=rep.end_bevel_deg,
            stock_width_mm=rep.stock_width_mm,
            stock_height_mm=rep.stock_height_mm,
        ))

    return rows


# ---------------------------------------------------------------------------
# Node connector plate outlines
# ---------------------------------------------------------------------------

def compute_node_plates(
    dome: TessellatedDome,
    params: DomeParameters,
) -> List[NodePlateOutline]:
    """Generate 2D connector plate outlines for each node.

    The plate is circular, sized to accommodate all incident struts.
    Bolt holes are placed at uniform angular spacing around the plate.
    """
    bolt_d_mm = params.node_connector_bolt_diameter_m * 1000.0
    bolt_offset_mm = params.node_connector_bolt_offset_m * 1000.0
    plate_thickness_mm = params.node_connector_thickness_m * 1000.0

    # Build node → incident strut map
    incident: Dict[int, List[Strut]] = {}
    for strut in dome.struts:
        incident.setdefault(strut.start_index, []).append(strut)
        incident.setdefault(strut.end_index, []).append(strut)

    plates: List[NodePlateOutline] = []

    for node_idx in range(len(dome.nodes)):
        struts_at_node = incident.get(node_idx, [])
        valence = len(struts_at_node)
        if valence == 0:
            continue

        # Plate radius: bolt offset + margin for bolt head
        plate_radius_mm = bolt_offset_mm + bolt_d_mm * 1.5

        # Bolt hole positions at equal angular spacing
        bolt_positions: List[Tuple[float, float]] = []
        for i in range(valence):
            angle_rad = 2.0 * math.pi * i / valence
            x = bolt_offset_mm * math.cos(angle_rad)
            y = bolt_offset_mm * math.sin(angle_rad)
            bolt_positions.append((round(x, 3), round(y, 3)))

        # Circular outline approximated as polygon (32 segments)
        n_segments = 32
        outline: List[Tuple[float, float]] = []
        for i in range(n_segments):
            angle_rad = 2.0 * math.pi * i / n_segments
            x = plate_radius_mm * math.cos(angle_rad)
            y = plate_radius_mm * math.sin(angle_rad)
            outline.append((round(x, 3), round(y, 3)))

        plates.append(NodePlateOutline(
            node_index=node_idx,
            valence=valence,
            radius_mm=round(plate_radius_mm, 2),
            bolt_hole_positions=bolt_positions,
            bolt_diameter_mm=bolt_d_mm,
            outline_points=outline,
        ))

    return plates


# ---------------------------------------------------------------------------
# Assembly sequence
# ---------------------------------------------------------------------------

def compute_assembly_stages(
    dome: TessellatedDome,
    params: DomeParameters,
) -> List[AssemblyStage]:
    """Plan construction stages by elevation bands.

    The dome is assembled from bottom (belt) up to apex in horizontal
    rings.  Each stage corresponds to one Z-band of nodes and the struts
    connecting them.
    """
    if not dome.nodes or not dome.struts:
        return []

    # Classify nodes by elevation bands
    z_values = sorted(set(round(n[2], 4) for n in dome.nodes))
    if not z_values:
        return []

    # Group into bands of similar Z (within 5% of radius tolerance)
    band_tol = params.radius_m * 0.05
    bands: List[List[int]] = []
    current_band: List[int] = []
    current_z = z_values[0]

    # Map nodes to their Z band
    node_z = [(i, round(dome.nodes[i][2], 4)) for i in range(len(dome.nodes))]
    node_z.sort(key=lambda x: x[1])

    band_z_start = node_z[0][1]
    for idx, z in node_z:
        if abs(z - band_z_start) > band_tol and current_band:
            bands.append(current_band)
            current_band = [idx]
            band_z_start = z
        else:
            current_band.append(idx)
    if current_band:
        bands.append(current_band)

    # Create stages bottom-up
    stages: List[AssemblyStage] = []
    assembled_nodes: set = set()

    for stage_num, band_nodes in enumerate(bands, start=1):
        band_node_set = set(band_nodes)
        new_nodes = sorted(band_node_set - assembled_nodes)
        assembled_nodes.update(band_node_set)

        # Find struts that connect within or to this band
        stage_struts: List[int] = []
        for strut in dome.struts:
            if strut.start_index in band_node_set or strut.end_index in band_node_set:
                # Only include if both endpoints are now assembled
                if strut.start_index in assembled_nodes and strut.end_index in assembled_nodes:
                    stage_struts.append(strut.index)

        if not new_nodes and not stage_struts:
            continue

        # Name the stage
        if stage_num == 1:
            name = "Aluskiht / Base ring"
        elif stage_num == len(bands):
            name = "Tipusõlm / Apex"
        else:
            name = f"Kiht {stage_num} / Ring {stage_num}"

        z_min = min(dome.nodes[n][2] for n in band_nodes)
        z_max = max(dome.nodes[n][2] for n in band_nodes)

        desc = (
            f"Kõrgus / Height: {z_min:.3f} – {z_max:.3f} m, "
            f"{len(new_nodes)} sõlme / nodes, "
            f"{len(stage_struts)} latti / struts"
        )

        stages.append(AssemblyStage(
            stage_number=stage_num,
            name=name,
            description=desc,
            strut_indices=sorted(stage_struts),
            node_indices=sorted(new_nodes),
        ))

    return stages


# ---------------------------------------------------------------------------
# High-level factory
# ---------------------------------------------------------------------------

def production_for_params(
    dome: TessellatedDome,
    params: DomeParameters,
) -> ProductionPack:
    """Compute the full production data pack."""
    cuts = compute_strut_cuts(dome, params)
    saw_table = compute_saw_table(cuts)
    plates = compute_node_plates(dome, params)
    stages = compute_assembly_stages(dome, params)

    total_raw_m = sum(c.raw_length_mm for c in cuts) / 1000.0
    unique = len(saw_table)

    return ProductionPack(
        strut_cuts=cuts,
        saw_table=saw_table,
        node_plates=plates,
        assembly_stages=stages,
        total_raw_length_m=round(total_raw_m, 3),
        total_struts=len(cuts),
        unique_types=unique,
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_production_report(pack: ProductionPack, params: DomeParameters, path: Path) -> None:
    """Write a comprehensive JSON production report."""
    report = {
        "summary": {
            "total_struts": pack.total_struts,
            "unique_types": pack.unique_types,
            "total_raw_length_m": pack.total_raw_length_m,
            "stock_profile_mm": f"{params.stock_width_m * 1000:.1f} x {params.stock_height_m * 1000:.1f}",
            "kerf_mm": params.kerf_m * 1000.0,
        },
        "saw_table": [
            {
                "group": r.group,
                "count": r.count,
                "raw_length_mm": r.raw_length_mm,
                "net_length_mm": r.net_length_mm,
                "start_miter_deg": r.start_miter_deg,
                "end_miter_deg": r.end_miter_deg,
                "start_bevel_deg": r.start_bevel_deg,
                "end_bevel_deg": r.end_bevel_deg,
                "stock_mm": f"{r.stock_width_mm:.1f} x {r.stock_height_mm:.1f}",
            }
            for r in pack.saw_table
        ],
        "strut_cuts": [
            {
                "index": c.strut_index,
                "group": c.group,
                "raw_length_mm": c.raw_length_mm,
                "net_length_mm": c.net_length_mm,
                "start_miter_deg": c.start_miter_deg,
                "end_miter_deg": c.end_miter_deg,
                "start_bevel_deg": c.start_bevel_deg,
                "end_bevel_deg": c.end_bevel_deg,
                "start_node": c.start_node_index,
                "end_node": c.end_node_index,
            }
            for c in pack.strut_cuts
        ],
        "assembly_stages": [
            {
                "stage": s.stage_number,
                "name": s.name,
                "description": s.description,
                "strut_count": len(s.strut_indices),
                "node_count": len(s.node_indices),
            }
            for s in pack.assembly_stages
        ],
        "node_plates": [
            {
                "node_index": p.node_index,
                "valence": p.valence,
                "radius_mm": p.radius_mm,
                "bolt_diameter_mm": p.bolt_diameter_mm,
                "bolt_holes": len(p.bolt_hole_positions),
            }
            for p in pack.node_plates
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Production report written to %s", path)


def write_saw_table_csv(saw_table: List[SawTableRow], path: Path) -> None:
    """Write saw bench settings as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Group", "Count", "Raw Length (mm)", "Net Length (mm)",
            "Start Miter (°)", "End Miter (°)",
            "Start Bevel (°)", "End Bevel (°)",
            "Stock W (mm)", "Stock H (mm)",
        ])
        for row in saw_table:
            writer.writerow([
                row.group, row.count,
                f"{row.raw_length_mm:.2f}", f"{row.net_length_mm:.2f}",
                f"{row.start_miter_deg:.2f}", f"{row.end_miter_deg:.2f}",
                f"{row.start_bevel_deg:.2f}", f"{row.end_bevel_deg:.2f}",
                f"{row.stock_width_mm:.1f}", f"{row.stock_height_mm:.1f}",
            ])
    logging.info("Saw table CSV written to %s", path)


# ---------------------------------------------------------------------------
# DXF / SVG export for node connector plates
# ---------------------------------------------------------------------------

def _circle_points(cx: float, cy: float, r: float, n: int = 32) -> List[Tuple[float, float]]:
    """Generate *n* points on a circle centred at (cx, cy)."""
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def write_node_plate_dxf(plate: NodePlateOutline, path: Path) -> None:
    """Write a minimal ASCII DXF with the plate outline and bolt holes.

    This writer emits DXF R12 entities directly — no external library needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    def _add(code: int, value: str) -> None:
        lines.append(f"  {code}\n{value}")

    # Header section (minimal)
    _add(0, "SECTION")
    _add(2, "HEADER")
    _add(0, "ENDSEC")

    # Entities section
    _add(0, "SECTION")
    _add(2, "ENTITIES")

    # Plate outline as POLYLINE
    pts = plate.outline_points
    if pts:
        _add(0, "POLYLINE")
        _add(8, "PLATE_OUTLINE")
        _add(66, "1")
        _add(70, "1")  # closed polyline
        for x, y in pts:
            _add(0, "VERTEX")
            _add(8, "PLATE_OUTLINE")
            _add(10, f"{x:.4f}")
            _add(20, f"{y:.4f}")
            _add(30, "0.0")
        _add(0, "SEQEND")
        _add(8, "PLATE_OUTLINE")

    # Bolt holes as circles
    bolt_r = plate.bolt_diameter_mm / 2.0
    for bx, by in plate.bolt_hole_positions:
        _add(0, "CIRCLE")
        _add(8, "BOLT_HOLES")
        _add(10, f"{bx:.4f}")
        _add(20, f"{by:.4f}")
        _add(30, "0.0")
        _add(40, f"{bolt_r:.4f}")

    # Centre mark
    _add(0, "CIRCLE")
    _add(8, "CENTER")
    _add(10, "0.0")
    _add(20, "0.0")
    _add(30, "0.0")
    _add(40, "1.0")

    _add(0, "ENDSEC")
    _add(0, "EOF")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logging.info("Node plate DXF written to %s", path)


def write_node_plate_svg(plate: NodePlateOutline, path: Path) -> None:
    """Write an SVG file with the plate outline and bolt holes."""
    path.parent.mkdir(parents=True, exist_ok=True)

    margin = 5.0
    r = plate.radius_mm
    size = 2 * r + 2 * margin
    cx = r + margin
    cy = r + margin

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size:.2f}mm" height="{size:.2f}mm" '
        f'viewBox="0 0 {size:.2f} {size:.2f}">'
    )

    # Plate outline
    if plate.outline_points:
        pts_str = " ".join(
            f"{cx + x:.3f},{cy + y:.3f}" for x, y in plate.outline_points
        )
        parts.append(
            f'  <polygon points="{pts_str}" '
            f'fill="none" stroke="#333" stroke-width="0.3"/>'
        )

    # Bolt holes
    bolt_r = plate.bolt_diameter_mm / 2.0
    for bx, by in plate.bolt_hole_positions:
        parts.append(
            f'  <circle cx="{cx + bx:.3f}" cy="{cy + by:.3f}" '
            f'r="{bolt_r:.3f}" fill="none" stroke="#666" stroke-width="0.2"/>'
        )

    # Centre mark
    parts.append(
        f'  <circle cx="{cx:.3f}" cy="{cy:.3f}" '
        f'r="0.5" fill="none" stroke="#999" stroke-width="0.15"/>'
    )
    parts.append(
        f'  <line x1="{cx - 2:.3f}" y1="{cy:.3f}" '
        f'x2="{cx + 2:.3f}" y2="{cy:.3f}" stroke="#999" stroke-width="0.15"/>'
    )
    parts.append(
        f'  <line x1="{cx:.3f}" y1="{cy - 2:.3f}" '
        f'x2="{cx:.3f}" y2="{cy + 2:.3f}" stroke="#999" stroke-width="0.15"/>'
    )

    # Title
    parts.append(
        f'  <text x="{margin:.1f}" y="{size - 1:.1f}" '
        f'font-size="2.5" font-family="monospace" fill="#333">'
        f'Node {plate.node_index} – {plate.valence} struts – '
        f'R{plate.radius_mm:.1f}mm</text>'
    )

    parts.append("</svg>")

    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")
    logging.info("Node plate SVG written to %s", path)
