"""Assembly guide generation for geodesic dome construction.

Produces ring-by-ring assembly instructions with:
- Numbered stages (bottom-up)
- BOM references per stage
- Time estimates per stage and total
- SVG assembly diagrams
- JSON report to ``exports/assembly/``

Works headlessly (no FreeCAD dependency).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .parameters import DomeParameters
from .production import AssemblyStage, compute_assembly_stages
from .tessellation import TessellatedDome, Strut, Vector3

__all__ = [
    "AssemblyStepDetail",
    "AssemblyGuide",
    "assembly_guide_for_dome",
    "write_assembly_report",
    "write_assembly_svg",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StageBomEntry:
    """One BOM line for a single assembly stage."""
    category: str       # "strut" | "node_connector" | "panel" | "bolt"
    description: str
    quantity: int
    unit: str           # "tk" (pieces)


@dataclass(slots=True)
class AssemblyStepDetail:
    """Enriched assembly stage with BOM and time estimate."""
    stage_number: int
    name: str
    description: str
    strut_indices: List[int]
    node_indices: List[int]
    panel_indices: List[int]
    bom: List[StageBomEntry]
    estimated_minutes: float
    cumulative_minutes: float
    z_min: float
    z_max: float


@dataclass
class AssemblyGuide:
    """Complete assembly instruction set."""
    stages: List[AssemblyStepDetail]
    total_stages: int
    total_estimated_minutes: float
    total_estimated_hours: float
    workers: int
    dome_radius_m: float
    dome_frequency: int
    report_path: str = ""
    svg_paths: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strut_length_group(strut: Strut) -> str:
    """Return a human-readable length group label for a strut."""
    l_mm = strut.length * 1000.0
    return f"{l_mm:.0f}mm"


def _panels_touching_nodes(dome: TessellatedDome, node_indices: List[int]) -> List[int]:
    """Return panel indices whose all nodes are in the given set."""
    node_set = set(node_indices)
    result: List[int] = []
    for panel in dome.panels:
        if all(ni in node_set for ni in panel.node_indices):
            result.append(panel.index)
    return sorted(set(result))


def _estimate_stage_minutes(
    n_struts: int,
    n_nodes: int,
    n_panels: int,
    params: DomeParameters,
) -> float:
    """Estimate installation time for a stage in minutes (per crew)."""
    t_strut = float(getattr(params, "assembly_time_per_strut_min", 15.0) or 15.0)
    t_node = float(getattr(params, "assembly_time_per_node_min", 10.0) or 10.0)
    t_panel = float(getattr(params, "assembly_time_per_panel_min", 20.0) or 20.0)
    workers = max(1, int(getattr(params, "assembly_workers", 2) or 2))

    total_min = (n_struts * t_strut + n_nodes * t_node + n_panels * t_panel)
    # More workers = proportionally faster, but not perfectly linear
    efficiency = 0.7 + 0.3 / workers  # diminishing returns
    return total_min * efficiency / max(1, workers / 2)


def _build_stage_bom(
    n_struts: int,
    n_nodes: int,
    n_panels: int,
    strut_indices: List[int],
    dome: TessellatedDome,
    params: DomeParameters,
) -> List[StageBomEntry]:
    """Build BOM entries for a single assembly stage."""
    bom: List[StageBomEntry] = []

    if n_struts > 0:
        # Group struts by length
        length_groups: Dict[str, int] = {}
        for si in strut_indices:
            if 0 <= si < len(dome.struts):
                grp = _strut_length_group(dome.struts[si])
                length_groups[grp] = length_groups.get(grp, 0) + 1
        for grp, count in sorted(length_groups.items()):
            bom.append(StageBomEntry(
                category="strut",
                description=f"Latt / Strut {grp}",
                quantity=count,
                unit="tk",
            ))

    if n_nodes > 0:
        bom.append(StageBomEntry(
            category="node_connector",
            description="Sõlmeplaat / Node connector",
            quantity=n_nodes,
            unit="tk",
        ))
        # Bolts: assume 2 bolts per strut-end at each node
        bolts_per_node = 2  # conservative minimum
        bom.append(StageBomEntry(
            category="bolt",
            description="Polt / Bolt M12",
            quantity=n_nodes * bolts_per_node,
            unit="tk",
        ))

    if n_panels > 0:
        bom.append(StageBomEntry(
            category="panel",
            description="Paneel / Panel",
            quantity=n_panels,
            unit="tk",
        ))

    return bom


# ---------------------------------------------------------------------------
# SVG generation
# ---------------------------------------------------------------------------

def _project_to_2d(point: Vector3, center: Vector3, scale: float) -> Tuple[float, float]:
    """Simple isometric projection of a 3D point to 2D SVG coordinates."""
    x, y, z = point
    cx, cy, cz = center
    dx, dy, dz = x - cx, y - cy, z - cz
    # Isometric-ish: x-axis goes right, z-axis goes up, y-axis recedes
    px = (dx - dy) * 0.866 * scale
    py = -(dz - (dx + dy) * 0.5) * scale
    return (px, py)


def _generate_stage_svg(
    dome: TessellatedDome,
    stage: AssemblyStepDetail,
    all_assembled_nodes: set,
    all_assembled_struts: set,
    svg_width: float = 600.0,
    svg_height: float = 500.0,
) -> str:
    """Generate a standalone SVG for one assembly stage.

    Shows previously assembled structure in light grey and the current
    stage's new elements highlighted in color.
    """
    if not dome.nodes:
        return ""

    # Compute center and scale
    xs = [n[0] for n in dome.nodes]
    ys = [n[1] for n in dome.nodes]
    zs = [n[2] for n in dome.nodes]
    center = (
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        (min(zs) + max(zs)) / 2,
    )
    max_range = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 0.001)
    scale = min(svg_width, svg_height) * 0.4 / max_range

    offset_x = svg_width / 2
    offset_y = svg_height / 2

    def proj(pt: Vector3) -> Tuple[float, float]:
        px, py = _project_to_2d(pt, center, scale)
        return (px + offset_x, py + offset_y)

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'width="{svg_width}" height="{svg_height}" '
                 f'viewBox="0 0 {svg_width} {svg_height}">')
    lines.append(f'<rect width="100%" height="100%" fill="white"/>')

    # Title
    lines.append(f'<text x="10" y="25" font-size="16" font-family="sans-serif" '
                 f'font-weight="bold" fill="#333">'
                 f'Etapp {stage.stage_number}: {stage.name}</text>')
    lines.append(f'<text x="10" y="45" font-size="11" font-family="sans-serif" '
                 f'fill="#666">{stage.description}</text>')

    # Time estimate
    hours = int(stage.estimated_minutes // 60)
    mins = int(stage.estimated_minutes % 60)
    time_str = f"{hours}h {mins}min" if hours > 0 else f"{mins} min"
    lines.append(f'<text x="10" y="65" font-size="11" font-family="sans-serif" '
                 f'fill="#666">Hinnanguline aeg / Est. time: {time_str}</text>')

    # Previously assembled struts (grey)
    prev_strut_set = all_assembled_struts - set(stage.strut_indices)
    for si in sorted(prev_strut_set):
        if 0 <= si < len(dome.struts):
            s = dome.struts[si]
            x1, y1 = proj(s.start)
            x2, y2 = proj(s.end)
            lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
                         f'x2="{x2:.1f}" y2="{y2:.1f}" '
                         f'stroke="#ccc" stroke-width="1.5"/>')

    # Current stage struts (blue)
    for si in stage.strut_indices:
        if 0 <= si < len(dome.struts):
            s = dome.struts[si]
            x1, y1 = proj(s.start)
            x2, y2 = proj(s.end)
            lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
                         f'x2="{x2:.1f}" y2="{y2:.1f}" '
                         f'stroke="#2563eb" stroke-width="2.5"/>')

    # Previously assembled nodes (light grey dots)
    prev_node_set = all_assembled_nodes - set(stage.node_indices)
    for ni in sorted(prev_node_set):
        if 0 <= ni < len(dome.nodes):
            cx, cy = proj(dome.nodes[ni])
            lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" '
                         f'fill="#ddd" stroke="#bbb" stroke-width="0.5"/>')

    # Current stage nodes (orange with number)
    for ni in stage.node_indices:
        if 0 <= ni < len(dome.nodes):
            cx, cy = proj(dome.nodes[ni])
            lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" '
                         f'fill="#f59e0b" stroke="#d97706" stroke-width="1"/>')
            lines.append(f'<text x="{cx:.1f}" y="{cy + 15:.1f}" '
                         f'font-size="8" font-family="sans-serif" '
                         f'text-anchor="middle" fill="#92400e">{ni}</text>')

    # BOM legend box
    bom_y = svg_height - 20 - len(stage.bom) * 16
    lines.append(f'<rect x="10" y="{bom_y - 20}" '
                 f'width="250" height="{len(stage.bom) * 16 + 30}" '
                 f'fill="white" fill-opacity="0.9" stroke="#ccc" rx="4"/>')
    lines.append(f'<text x="20" y="{bom_y - 4}" font-size="11" '
                 f'font-family="sans-serif" font-weight="bold" '
                 f'fill="#333">Osaloend / BOM:</text>')
    for i, entry in enumerate(stage.bom):
        by = bom_y + 14 + i * 16
        lines.append(f'<text x="20" y="{by}" font-size="10" '
                     f'font-family="sans-serif" fill="#444">'
                     f'• {entry.quantity}× {entry.description}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def assembly_guide_for_dome(
    dome: TessellatedDome,
    params: DomeParameters,
) -> AssemblyGuide:
    """Build a complete assembly guide from the tessellated dome.

    Uses ``compute_assembly_stages`` from :mod:`production` for the
    ring-by-ring sequence, then enriches each stage with BOM references,
    panel assignments and time estimates.
    """
    raw_stages = compute_assembly_stages(dome, params)

    workers = max(1, int(getattr(params, "assembly_workers", 2) or 2))
    all_assembled_nodes: set = set()
    cumulative_min = 0.0
    enriched: List[AssemblyStepDetail] = []

    for raw in raw_stages:
        # Track cumulative assembled nodes for panel assignment
        stage_all_nodes = set(raw.node_indices) | all_assembled_nodes

        # Panels whose every node is assembled by end of this stage
        panel_indices = _panels_touching_nodes(dome, sorted(stage_all_nodes))
        # Only new panels (not assigned to earlier stages)
        already_assigned = set()
        for prev in enriched:
            already_assigned.update(prev.panel_indices)
        new_panel_indices = [pi for pi in panel_indices if pi not in already_assigned]

        # Z range
        z_vals = [dome.nodes[ni][2] for ni in raw.node_indices if 0 <= ni < len(dome.nodes)]
        z_min = min(z_vals) if z_vals else 0.0
        z_max = max(z_vals) if z_vals else 0.0

        # Time estimate
        est_min = _estimate_stage_minutes(
            len(raw.strut_indices),
            len(raw.node_indices),
            len(new_panel_indices),
            params,
        )
        cumulative_min += est_min

        # BOM
        bom = _build_stage_bom(
            len(raw.strut_indices),
            len(raw.node_indices),
            len(new_panel_indices),
            raw.strut_indices,
            dome,
            params,
        )

        enriched.append(AssemblyStepDetail(
            stage_number=raw.stage_number,
            name=raw.name,
            description=raw.description,
            strut_indices=raw.strut_indices,
            node_indices=raw.node_indices,
            panel_indices=new_panel_indices,
            bom=bom,
            estimated_minutes=round(est_min, 1),
            cumulative_minutes=round(cumulative_min, 1),
            z_min=z_min,
            z_max=z_max,
        ))

        all_assembled_nodes.update(raw.node_indices)

    return AssemblyGuide(
        stages=enriched,
        total_stages=len(enriched),
        total_estimated_minutes=round(cumulative_min, 1),
        total_estimated_hours=round(cumulative_min / 60.0, 2),
        workers=workers,
        dome_radius_m=params.radius_m,
        dome_frequency=params.frequency,
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_assembly_report(guide: AssemblyGuide, params: DomeParameters, path: Path) -> None:
    """Write JSON assembly guide report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "summary": {
            "total_stages": guide.total_stages,
            "total_estimated_minutes": guide.total_estimated_minutes,
            "total_estimated_hours": guide.total_estimated_hours,
            "workers": guide.workers,
            "dome_radius_m": guide.dome_radius_m,
            "dome_frequency": guide.dome_frequency,
        },
        "stages": [
            {
                "stage_number": s.stage_number,
                "name": s.name,
                "description": s.description,
                "z_range_m": {"min": round(s.z_min, 4), "max": round(s.z_max, 4)},
                "strut_count": len(s.strut_indices),
                "node_count": len(s.node_indices),
                "panel_count": len(s.panel_indices),
                "strut_indices": s.strut_indices,
                "node_indices": s.node_indices,
                "panel_indices": s.panel_indices,
                "bom": [
                    {
                        "category": b.category,
                        "description": b.description,
                        "quantity": b.quantity,
                        "unit": b.unit,
                    }
                    for b in s.bom
                ],
                "estimated_minutes": s.estimated_minutes,
                "cumulative_minutes": s.cumulative_minutes,
            }
            for s in guide.stages
        ],
    }

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logging.info("Assembly report written: %s", path)


def write_assembly_svg(
    guide: AssemblyGuide,
    dome: TessellatedDome,
    out_dir: Path,
) -> List[str]:
    """Write per-stage SVG assembly diagrams.

    Returns list of written SVG file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    all_assembled_nodes: set = set()
    all_assembled_struts: set = set()

    for stage in guide.stages:
        # Include all struts/nodes up to and including this stage
        all_assembled_nodes.update(stage.node_indices)
        all_assembled_struts.update(stage.strut_indices)

        svg_content = _generate_stage_svg(
            dome,
            stage,
            set(all_assembled_nodes),
            set(all_assembled_struts),
        )
        if not svg_content:
            continue

        svg_path = out_dir / f"assembly_stage_{stage.stage_number:02d}.svg"
        svg_path.write_text(svg_content, encoding="utf-8")
        paths.append(str(svg_path))
        logging.debug("Assembly SVG: %s", svg_path)

    logging.info("Assembly SVGs written: %d files to %s", len(paths), out_dir)
    return paths
