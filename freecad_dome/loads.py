"""Structural load calculations for the geodesic dome.

Computes design loads per Eurocode (EN 1991) and distributes them to
individual nodes for downstream FEM analysis or manual verification.

Load cases
----------

1. **Dead load (G)** — self-weight of struts + covering material.
2. **Snow load (S)** — ground snow load × shape factor for dome curvature.
3. **Wind load (W)** — reference wind pressure × Cp coefficients on a
   spherical surface.

The module outputs per-node force vectors and a load-combination summary
in JSON (or CSV) that can be imported into FEM software.

Usage::

    from freecad_dome.loads import compute_loads, write_load_report

    result = compute_loads(dome, params)
    write_load_report(result, params, "exports/load_report.json")
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .parameters import DomeParameters

__all__ = [
    "LoadCase",
    "NodeLoad",
    "LoadResult",
    "compute_loads",
    "write_load_report",
    "write_load_csv",
]

log = logging.getLogger(__name__)

Vector3 = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# Constants / defaults (Eurocode-based)
# ---------------------------------------------------------------------------

# Wind terrain categories → reference height factor (simplified).
TERRAIN_CATEGORIES = {
    "0": 1.30,    # sea / coastal
    "I": 1.17,    # open field, lakes
    "II": 1.00,   # farmland, scattered buildings (default)
    "III": 0.85,  # suburban, forest
    "IV": 0.70,   # urban centres
}

# Snow zone ground loads (kN/m²) — simplified Estonia / Scandinavia typical values.
SNOW_ZONES = {
    "I": 1.0,
    "II": 1.5,
    "III": 2.0,     # Estonia typical
    "IV": 2.5,
    "V": 3.0,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class NodeLoad:
    """Force acting on a single node."""

    node_index: int
    fx_kn: float = 0.0   # force in X (horizontal)
    fy_kn: float = 0.0   # force in Y (horizontal)
    fz_kn: float = 0.0   # force in Z (vertical, + up)

    @property
    def magnitude_kn(self) -> float:
        return math.sqrt(self.fx_kn ** 2 + self.fy_kn ** 2 + self.fz_kn ** 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_index": self.node_index,
            "fx_kn": round(self.fx_kn, 4),
            "fy_kn": round(self.fy_kn, 4),
            "fz_kn": round(self.fz_kn, 4),
            "magnitude_kn": round(self.magnitude_kn, 4),
        }


@dataclass(slots=True)
class LoadCase:
    """A named load case with per-node forces."""

    name: str                                # "dead" | "snow" | "wind"
    label: str                               # Human-readable
    partial_factor: float = 1.0              # γ factor for ULS combinations
    node_loads: List[NodeLoad] = field(default_factory=list)

    @property
    def total_vertical_kn(self) -> float:
        return sum(nl.fz_kn for nl in self.node_loads)

    @property
    def total_horizontal_kn(self) -> float:
        return sum(
            math.sqrt(nl.fx_kn ** 2 + nl.fy_kn ** 2) for nl in self.node_loads
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "partial_factor": self.partial_factor,
            "total_vertical_kn": round(self.total_vertical_kn, 3),
            "total_horizontal_kn": round(self.total_horizontal_kn, 3),
            "node_count": len(self.node_loads),
        }


@dataclass(slots=True)
class LoadCombination:
    """A ULS or SLS load combination."""

    name: str             # e.g. "ULS_1" or "SLS_char"
    factors: Dict[str, float] = field(default_factory=dict)  # case_name → factor

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "factors": self.factors}


@dataclass(slots=True)
class LoadResult:
    """Complete load analysis results."""

    cases: List[LoadCase] = field(default_factory=list)
    combinations: List[LoadCombination] = field(default_factory=list)
    total_dead_weight_kn: float = 0.0
    total_snow_kn: float = 0.0
    total_wind_kn: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_dead_weight_kn": round(self.total_dead_weight_kn, 3),
            "total_snow_kn": round(self.total_snow_kn, 3),
            "total_wind_kn": round(self.total_wind_kn, 3),
            "cases": [c.to_dict() for c in self.cases],
            "combinations": [c.to_dict() for c in self.combinations],
        }


# ---------------------------------------------------------------------------
# Tributary area computation
# ---------------------------------------------------------------------------

def _polygon_area_3d(pts: List[Vector3]) -> float:
    """Area of a 3-D polygon via Newell's method."""
    n = len(pts)
    if n < 3:
        return 0.0
    nx = ny = nz = 0.0
    for i in range(n):
        j = (i + 1) % n
        nx += (pts[i][1] - pts[j][1]) * (pts[i][2] + pts[j][2])
        ny += (pts[i][2] - pts[j][2]) * (pts[i][0] + pts[j][0])
        nz += (pts[i][0] - pts[j][0]) * (pts[i][1] + pts[j][1])
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def _tributary_areas(dome: Any) -> Dict[int, float]:
    """Compute tributary area per node (m²).

    Each panel's area is distributed equally to its corner nodes.
    """
    areas: Dict[int, float] = {}
    for panel in dome.panels:
        pts = [dome.nodes[i] for i in panel.node_indices]
        panel_area = _polygon_area_3d(pts)
        share = panel_area / max(len(panel.node_indices), 1)
        for ni in panel.node_indices:
            areas[ni] = areas.get(ni, 0.0) + share
    return areas


# ---------------------------------------------------------------------------
# Individual load cases
# ---------------------------------------------------------------------------

def _dead_load(
    dome: Any,
    params: DomeParameters,
    tributary: Dict[int, float],
) -> LoadCase:
    """Self-weight of struts and covering → vertical downward force per node.

    Strut weight is distributed to the two end-nodes of each strut.
    Covering weight is distributed via tributary area.
    """
    # Strut linear density.
    stock_w = float(params.stock_width_m)
    stock_h = float(params.stock_height_m)
    material_name = params.material
    mat = params.materials.get(material_name)
    density_kg_m3 = float(mat.density) if mat and mat.density else 500.0  # default wood
    strut_linear_kg_m = density_kg_m3 * stock_w * stock_h  # kg per metre of strut

    # Covering surface density.
    covering_weight_kg_m2 = 0.0
    try:
        from .covering import covering_for_params
        spec = covering_for_params(params)
        covering_weight_kg_m2 = spec.weight_kg_m2
    except Exception:
        # Fall back: glass ~10 kg/m² for 4 mm.
        glass_t = float(params.glass_thickness_m)
        if glass_t > 0:
            covering_weight_kg_m2 = 2500.0 * glass_t

    g = 9.81  # m/s²
    node_fz: Dict[int, float] = {}

    # Strut contribution.
    for strut in dome.struts:
        p0 = dome.nodes[strut.start_index]
        p1 = dome.nodes[strut.end_index]
        length = math.sqrt(sum((a - b) ** 2 for a, b in zip(p0, p1)))
        weight_kn = strut_linear_kg_m * length * g / 1000.0
        half = weight_kn / 2.0
        for ni in (strut.start_index, strut.end_index):
            node_fz[ni] = node_fz.get(ni, 0.0) - half  # downward = negative Z

    # Covering contribution.
    for ni, area in tributary.items():
        w = covering_weight_kg_m2 * area * g / 1000.0
        node_fz[ni] = node_fz.get(ni, 0.0) - w

    node_loads = [
        NodeLoad(node_index=ni, fz_kn=fz) for ni, fz in sorted(node_fz.items())
    ]

    total_kn = sum(nl.fz_kn for nl in node_loads)
    return LoadCase(
        name="dead",
        label="Dead load (self-weight)",
        partial_factor=1.35,  # EN 1990 ULS permanent
        node_loads=node_loads,
    )


def _snow_dome_shape_factor(zenith_deg: float) -> float:
    """Snow shape coefficient μ₁ for a dome surface.

    EN 1991-1-3 §5.3.5 / Annex B: for spherical roofs μ₁ varies from
    0.8 at the apex to 0 at zenith ≥ 60°. Linear interpolation.
    """
    if zenith_deg <= 0:
        return 0.8
    if zenith_deg >= 60:
        return 0.0
    return 0.8 * (1.0 - zenith_deg / 60.0)


def _snow_load(
    dome: Any,
    params: DomeParameters,
    tributary: Dict[int, float],
) -> LoadCase:
    """Snow load on the dome surface → vertical force per node.

    Uses ground snow load × exposure × thermal × shape factor.
    """
    snow_zone = getattr(params, "load_snow_zone", "III")
    sk = SNOW_ZONES.get(snow_zone, 2.0)   # kN/m² ground snow
    Ce = getattr(params, "load_snow_exposure", 1.0)
    Ct = getattr(params, "load_snow_thermal", 1.0)

    R = float(params.radius_m)
    center = (0.0, 0.0, 0.0)

    node_loads: List[NodeLoad] = []
    for ni, area in sorted(tributary.items()):
        pos = dome.nodes[ni]
        # Zenith angle: angle from vertical (+Z) axis.
        dx, dy, dz = pos[0], pos[1], pos[2]
        r_horiz = math.sqrt(dx * dx + dy * dy)
        zenith_deg = math.degrees(math.atan2(r_horiz, dz))
        mu = _snow_dome_shape_factor(zenith_deg)
        s = mu * Ce * Ct * sk  # kN/m² on surface
        fz = -s * area  # downward
        node_loads.append(NodeLoad(node_index=ni, fz_kn=fz))

    return LoadCase(
        name="snow",
        label="Snow load",
        partial_factor=1.50,  # EN 1990 variable
        node_loads=node_loads,
    )


def _wind_cp_dome(zenith_deg: float, phi_deg: float = 0.0) -> float:
    """External pressure coefficient Cp,e for a spherical dome.

    Simplified model based on EN 1991-1-4 §7.2.8 (domes):
    - Apex (zenith ≈ 0°): Cp ≈ -1.0 (suction)
    - Equator/base (zenith ≈ 90°): Cp ≈ +0.8 (windward) / -0.5 (leeward)

    phi_deg is the azimuth relative to wind direction (0 = windward face).
    """
    # Radial variation (zenith).
    if zenith_deg <= 30:
        cp_radial = -1.0 + 0.6 * (zenith_deg / 30.0)
    elif zenith_deg <= 90:
        cp_radial = -0.4 + 1.2 * ((zenith_deg - 30) / 60.0)
    else:
        cp_radial = 0.0

    # Azimuth correction: windward vs leeward.
    cos_phi = math.cos(math.radians(phi_deg))
    if cos_phi > 0:
        # Windward side — more positive pressure at base.
        return cp_radial + 0.3 * cos_phi * (zenith_deg / 90.0)
    else:
        # Leeward side — more suction.
        return cp_radial + 0.2 * cos_phi * (zenith_deg / 90.0)


def _wind_load(
    dome: Any,
    params: DomeParameters,
    tributary: Dict[int, float],
) -> LoadCase:
    """Wind load on the dome surface → force per node.

    Pressure acts normal to the dome surface at each node.
    """
    v_ref = getattr(params, "load_wind_speed_ms", 25.0)  # m/s reference
    terrain = getattr(params, "load_wind_terrain", "II")
    wind_dir_deg = getattr(params, "load_wind_direction_deg", 0.0)

    terrain_factor = TERRAIN_CATEGORIES.get(terrain, 1.0)
    rho = 1.25  # kg/m³ air density
    q_ref = 0.5 * rho * v_ref ** 2 / 1000.0  # kN/m²
    q_peak = q_ref * terrain_factor  # simplified peak pressure

    R = float(params.radius_m)
    node_loads: List[NodeLoad] = []

    for ni, area in sorted(tributary.items()):
        pos = dome.nodes[ni]
        dx, dy, dz = pos[0], pos[1], pos[2]
        r_horiz = math.sqrt(dx * dx + dy * dy)
        zenith_deg = math.degrees(math.atan2(r_horiz, dz))

        # Azimuth relative to wind direction.
        node_azimuth = math.degrees(math.atan2(dy, dx))
        phi_deg = (node_azimuth - wind_dir_deg + 180) % 360 - 180

        cp = _wind_cp_dome(zenith_deg, phi_deg)
        pressure_kn_m2 = q_peak * cp

        # Force normal to dome surface.
        r_total = math.sqrt(dx * dx + dy * dy + dz * dz)
        if r_total < 1e-10:
            continue
        # Outward normal at node (radial direction on sphere).
        nx_n = dx / r_total
        ny_n = dy / r_total
        nz_n = dz / r_total

        force = pressure_kn_m2 * area
        node_loads.append(NodeLoad(
            node_index=ni,
            fx_kn=force * nx_n,
            fy_kn=force * ny_n,
            fz_kn=force * nz_n,
        ))

    return LoadCase(
        name="wind",
        label="Wind load",
        partial_factor=1.50,  # EN 1990 variable
        node_loads=node_loads,
    )


# ---------------------------------------------------------------------------
# Load combinations (Eurocode EN 1990)
# ---------------------------------------------------------------------------

def _standard_combinations() -> List[LoadCombination]:
    """Standard ULS and SLS combinations per EN 1990 §6.4.3.2 (STR/GEO)."""
    return [
        # ULS: 1.35G + 1.50S + 0.90W
        LoadCombination(
            name="ULS_1_snow_dominant",
            factors={"dead": 1.35, "snow": 1.50, "wind": 0.90},
        ),
        # ULS: 1.35G + 0.75S + 1.50W
        LoadCombination(
            name="ULS_2_wind_dominant",
            factors={"dead": 1.35, "snow": 0.75, "wind": 1.50},
        ),
        # ULS: 1.0G + 0.0S + 1.50W  (uplift check)
        LoadCombination(
            name="ULS_3_wind_uplift",
            factors={"dead": 1.00, "snow": 0.00, "wind": 1.50},
        ),
        # SLS: 1.0G + 1.0S + 0.6W
        LoadCombination(
            name="SLS_characteristic",
            factors={"dead": 1.00, "snow": 1.00, "wind": 0.60},
        ),
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_loads(dome: Any, params: DomeParameters) -> LoadResult:
    """Compute all load cases and combinations for the dome.

    Parameters
    ----------
    dome : TessellatedDome
        Dome geometry with nodes, struts, and panels.
    params : DomeParameters
        Current configuration.

    Returns
    -------
    LoadResult
        Complete load analysis with per-node forces and combinations.
    """
    tributary = _tributary_areas(dome)

    dead = _dead_load(dome, params, tributary)
    snow = _snow_load(dome, params, tributary)
    wind = _wind_load(dome, params, tributary)

    result = LoadResult(
        cases=[dead, snow, wind],
        combinations=_standard_combinations(),
        total_dead_weight_kn=abs(dead.total_vertical_kn),
        total_snow_kn=abs(snow.total_vertical_kn),
        total_wind_kn=wind.total_horizontal_kn,
    )

    log.info(
        "Loads computed: dead=%.1f kN, snow=%.1f kN, wind=%.1f kN (horiz)",
        result.total_dead_weight_kn,
        result.total_snow_kn,
        result.total_wind_kn,
    )
    return result


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_load_report(
    result: LoadResult,
    params: DomeParameters,
    path: Any,
) -> None:
    """Write a JSON report with load analysis results."""
    report = {
        "summary": result.to_dict(),
        "parameters": {
            "radius_m": params.radius_m,
            "frequency": params.frequency,
            "hemisphere_ratio": params.hemisphere_ratio,
            "material": params.material,
            "wind_speed_ms": getattr(params, "load_wind_speed_ms", 25.0),
            "snow_zone": getattr(params, "load_snow_zone", "III"),
            "wind_terrain": getattr(params, "load_wind_terrain", "II"),
        },
        "node_loads": {},
    }

    for case in result.cases:
        report["node_loads"][case.name] = [nl.to_dict() for nl in case.node_loads]

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote load report to %s", out)


def write_load_csv(
    result: LoadResult,
    path: Any,
) -> None:
    """Write a CSV file with combined node forces for FEM import.

    Each row: node_index, x, y, z, Fx_dead, Fy_dead, Fz_dead, Fx_snow, ...
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build a lookup: case_name → node_index → NodeLoad.
    case_map: Dict[str, Dict[int, NodeLoad]] = {}
    all_nodes: set = set()
    for case in result.cases:
        lookup: Dict[int, NodeLoad] = {}
        for nl in case.node_loads:
            lookup[nl.node_index] = nl
            all_nodes.add(nl.node_index)
        case_map[case.name] = lookup

    header = ["node_index"]
    for case in result.cases:
        header.extend([
            f"Fx_{case.name}_kN",
            f"Fy_{case.name}_kN",
            f"Fz_{case.name}_kN",
        ])

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ni in sorted(all_nodes):
            row = [ni]
            for case in result.cases:
                nl = case_map[case.name].get(ni)
                if nl:
                    row.extend([
                        round(nl.fx_kn, 4),
                        round(nl.fy_kn, 4),
                        round(nl.fz_kn, 4),
                    ])
                else:
                    row.extend([0.0, 0.0, 0.0])
            writer.writerow(row)

    log.info("Wrote load CSV to %s", out)
