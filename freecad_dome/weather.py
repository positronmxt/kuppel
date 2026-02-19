"""Weather protection, sealing and drainage for geodesic domes.

Covers:
- Gasket profiles (EPDM, silicone) between struts and covering panels
- Gasket dimensions included in BOM
- Eave/edge drip-edge details for water-tightness
- Condensation drainage allowance for polycarbonate panels
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .parameters import DomeParameters
from .tessellation import Strut, TessellatedDome, Vector3

__all__ = [
    "GasketProfile",
    "GASKET_PROFILES",
    "DrainageSpec",
    "EaveDetail",
    "WeatherPack",
    "weather_for_params",
    "gasket_bom_rows",
    "write_weather_report",
]


# ---------------------------------------------------------------------------
# Gasket profile catalogue
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class GasketProfile:
    """Specification for a sealing gasket between strut/panel edges."""
    name: str                       # e.g. "EPDM D-profile 10x8"
    material: str                   # EPDM | silicone | neoprene | butyl
    width_mm: float                 # gasket width (contact face)
    height_mm: float                # gasket height (compressed)
    compression_ratio: float        # target compression (0.2 = 20%)
    temperature_min_c: float        # service temperature min °C
    temperature_max_c: float        # service temperature max °C
    price_eur_per_m: float          # indicative cost per running metre
    uv_resistant: bool = True
    colour: str = "black"


GASKET_PROFILES: Dict[str, GasketProfile] = {
    "epdm_d_10x8": GasketProfile(
        name="EPDM D-profile 10×8 mm",
        material="EPDM",
        width_mm=10.0,
        height_mm=8.0,
        compression_ratio=0.25,
        temperature_min_c=-40.0,
        temperature_max_c=120.0,
        price_eur_per_m=1.20,
    ),
    "epdm_p_9x5": GasketProfile(
        name="EPDM P-profile 9×5 mm",
        material="EPDM",
        width_mm=9.0,
        height_mm=5.5,
        compression_ratio=0.20,
        temperature_min_c=-40.0,
        temperature_max_c=120.0,
        price_eur_per_m=0.80,
    ),
    "silicone_12x8": GasketProfile(
        name="Silicone tube 12×8 mm",
        material="silicone",
        width_mm=12.0,
        height_mm=8.0,
        compression_ratio=0.30,
        temperature_min_c=-60.0,
        temperature_max_c=200.0,
        price_eur_per_m=2.50,
        colour="translucent",
    ),
    "neoprene_flat_15x3": GasketProfile(
        name="Neoprene flat strip 15×3 mm",
        material="neoprene",
        width_mm=15.0,
        height_mm=3.0,
        compression_ratio=0.15,
        temperature_min_c=-30.0,
        temperature_max_c=100.0,
        price_eur_per_m=1.50,
    ),
    "butyl_tape_20x1": GasketProfile(
        name="Butyl sealing tape 20×1 mm",
        material="butyl",
        width_mm=20.0,
        height_mm=1.0,
        compression_ratio=0.50,
        temperature_min_c=-30.0,
        temperature_max_c=80.0,
        price_eur_per_m=0.60,
        uv_resistant=False,
    ),
}


# ---------------------------------------------------------------------------
# Drainage spec for polycarbonate multi-wall panels
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DrainageSpec:
    """Condensation drainage parameters for multi-wall polycarbonate."""
    panel_index: int
    panel_area_m2: float
    slope_deg: float             # panel tilt from horizontal
    drain_hole_count: int        # number of drainage weep-holes
    drain_hole_diameter_mm: float
    anti_dust_tape_length_mm: float  # tape at bottom edge
    breather_tape_length_mm: float   # tape at top edge


@dataclass(slots=True)
class EaveDetail:
    """Drip-edge / eave detail at the base ring of the dome."""
    node_index: int
    position: Vector3
    azimuth_deg: float
    drip_edge_length_mm: float
    overhang_mm: float


@dataclass
class WeatherPack:
    """Complete weather protection data."""
    gasket_profile: GasketProfile
    total_gasket_length_m: float
    gasket_per_strut_m: Dict[int, float]     # strut index → gasket length
    gasket_per_panel_m: Dict[int, float]     # panel index → perimeter gasket
    drainage_specs: List[DrainageSpec]
    eave_details: List[EaveDetail]
    total_drain_holes: int
    estimated_gasket_cost_eur: float


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _dist(a: Vector3, b: Vector3) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _panel_perimeter(nodes: List[Vector3], node_indices: Tuple[int, ...]) -> float:
    """Sum of edge lengths around a panel."""
    total = 0.0
    n = len(node_indices)
    for i in range(n):
        a = nodes[node_indices[i]]
        b = nodes[node_indices[(i + 1) % n]]
        total += _dist(a, b)
    return total


def _panel_slope_deg(normal: Vector3) -> float:
    """Angle of panel from horizontal (0 = horizontal, 90 = vertical)."""
    nlen = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    if nlen < 1e-12:
        return 0.0
    cos_z = abs(normal[2]) / nlen
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_z))))


def _polygon_area_3d(points: List[Vector3]) -> float:
    """Area of a 3D polygon via Newell's method."""
    n = len(points)
    if n < 3:
        return 0.0
    sx = sy = sz = 0.0
    for i in range(n):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[(i + 1) % n]
        sx += (y1 - y2) * (z1 + z2)
        sy += (z1 - z2) * (x1 + x2)
        sz += (x1 - x2) * (y1 + y2)
    return 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)


# ---------------------------------------------------------------------------
# Gasket BOM
# ---------------------------------------------------------------------------

def _select_gasket(params: DomeParameters) -> GasketProfile:
    """Select appropriate gasket profile based on parameters."""
    gasket_type = getattr(params, "gasket_type", "epdm_d_10x8")
    if gasket_type in GASKET_PROFILES:
        return GASKET_PROFILES[gasket_type]
    # Default to EPDM D-profile
    return GASKET_PROFILES["epdm_d_10x8"]


def gasket_bom_rows(
    dome: TessellatedDome,
    params: DomeParameters,
) -> List[Dict[str, Any]]:
    """Return BOM rows for gasket material."""
    profile = _select_gasket(params)
    rows: List[Dict[str, Any]] = []

    # Gasket runs along each strut (both sides of the strut web)
    for strut in dome.struts:
        length_m = strut.length * 2.0  # both sides
        rows.append({
            "item": f"Gasket – Strut {strut.index}",
            "material": profile.name,
            "length_m": round(length_m, 3),
            "width_mm": profile.width_mm,
        })

    return rows


# ---------------------------------------------------------------------------
# Drainage computation
# ---------------------------------------------------------------------------

def _compute_drainage(
    dome: TessellatedDome,
    params: DomeParameters,
) -> List[DrainageSpec]:
    """Compute drainage specs for panels that need condensation management."""
    covering_type = getattr(params, "covering_type", "glass")

    # Drainage only applies to multi-wall polycarbonate
    if not covering_type.startswith("pc_"):
        return []

    specs: List[DrainageSpec] = []
    drain_hole_d_mm = 5.0  # standard weep-hole size
    tape_width_mm = 25.0   # anti-dust / breather tape width

    for panel in dome.panels:
        panel_pts = [dome.nodes[ni] for ni in panel.node_indices]
        area = _polygon_area_3d(panel_pts)
        slope = _panel_slope_deg(panel.normal)
        perimeter = _panel_perimeter(dome.nodes, panel.node_indices)

        # Number of drain holes: 1 per 100mm of bottom edge
        n_edges = len(panel.node_indices)
        bottom_edge_m = perimeter / n_edges  # approximate
        n_holes = max(1, int(bottom_edge_m * 1000 / 100))

        # Tape lengths: top and bottom edges
        tape_len = bottom_edge_m * 1000  # mm

        specs.append(DrainageSpec(
            panel_index=panel.index,
            panel_area_m2=round(area, 4),
            slope_deg=round(slope, 1),
            drain_hole_count=n_holes,
            drain_hole_diameter_mm=drain_hole_d_mm,
            anti_dust_tape_length_mm=round(tape_len, 1),
            breather_tape_length_mm=round(tape_len, 1),
        ))

    return specs


# ---------------------------------------------------------------------------
# Eave / drip-edge details
# ---------------------------------------------------------------------------

def _compute_eave_details(
    dome: TessellatedDome,
    params: DomeParameters,
) -> List[EaveDetail]:
    """Compute drip-edge details at belt (base) nodes."""
    belt_height = params.radius_m * (1.0 - 2.0 * params.hemisphere_ratio)
    eps = max(1e-6, params.radius_m * 1e-5)
    overhang_mm = 15.0  # standard drip-edge overhang

    details: List[EaveDetail] = []

    for i, node in enumerate(dome.nodes):
        if abs(node[2] - belt_height) > eps:
            continue
        azimuth = math.degrees(math.atan2(node[1], node[0])) % 360.0

        # Find struts ending at this belt node to determine drip-edge length
        connected_lengths = []
        for strut in dome.struts:
            if strut.start_index == i or strut.end_index == i:
                connected_lengths.append(strut.length)

        # Drip-edge covers half the average adjacent strut length
        avg_len = sum(connected_lengths) / max(1, len(connected_lengths))
        drip_len_mm = avg_len * 1000.0 * 0.5

        details.append(EaveDetail(
            node_index=i,
            position=node,
            azimuth_deg=round(azimuth, 2),
            drip_edge_length_mm=round(drip_len_mm, 1),
            overhang_mm=overhang_mm,
        ))

    # Sort by azimuth
    details.sort(key=lambda d: d.azimuth_deg)
    return details


# ---------------------------------------------------------------------------
# High-level factory
# ---------------------------------------------------------------------------

def weather_for_params(
    dome: TessellatedDome,
    params: DomeParameters,
) -> WeatherPack:
    """Compute complete weather protection data."""
    profile = _select_gasket(params)

    # Gasket per strut: both sides
    gasket_per_strut: Dict[int, float] = {}
    for strut in dome.struts:
        gasket_per_strut[strut.index] = round(strut.length * 2.0, 4)

    # Gasket per panel: full perimeter
    gasket_per_panel: Dict[int, float] = {}
    for panel in dome.panels:
        perim = _panel_perimeter(dome.nodes, panel.node_indices)
        gasket_per_panel[panel.index] = round(perim, 4)

    total_gasket_strut = sum(gasket_per_strut.values())
    total_gasket_panel = sum(gasket_per_panel.values())
    # Use the larger of strut-based or panel-based estimate
    # (they measure the same edges from different perspectives)
    total_gasket_m = max(total_gasket_strut, total_gasket_panel)

    drainage = _compute_drainage(dome, params)
    eave = _compute_eave_details(dome, params)

    return WeatherPack(
        gasket_profile=profile,
        total_gasket_length_m=round(total_gasket_m, 3),
        gasket_per_strut_m=gasket_per_strut,
        gasket_per_panel_m=gasket_per_panel,
        drainage_specs=drainage,
        eave_details=eave,
        total_drain_holes=sum(d.drain_hole_count for d in drainage),
        estimated_gasket_cost_eur=round(total_gasket_m * profile.price_eur_per_m, 2),
    )


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_weather_report(pack: WeatherPack, params: DomeParameters, path: Path) -> None:
    """Write weather protection report as JSON."""
    report: Dict[str, Any] = {
        "gasket": {
            "profile": pack.gasket_profile.name,
            "material": pack.gasket_profile.material,
            "width_mm": pack.gasket_profile.width_mm,
            "height_mm": pack.gasket_profile.height_mm,
            "compression_ratio": pack.gasket_profile.compression_ratio,
            "total_length_m": pack.total_gasket_length_m,
            "estimated_cost_eur": pack.estimated_gasket_cost_eur,
            "uv_resistant": pack.gasket_profile.uv_resistant,
            "temperature_range_c": [
                pack.gasket_profile.temperature_min_c,
                pack.gasket_profile.temperature_max_c,
            ],
        },
        "drainage": {
            "total_drain_holes": pack.total_drain_holes,
            "applicable_panels": len(pack.drainage_specs),
            "specs": [
                {
                    "panel_index": d.panel_index,
                    "area_m2": d.panel_area_m2,
                    "slope_deg": d.slope_deg,
                    "drain_holes": d.drain_hole_count,
                    "hole_diameter_mm": d.drain_hole_diameter_mm,
                    "anti_dust_tape_mm": d.anti_dust_tape_length_mm,
                    "breather_tape_mm": d.breather_tape_length_mm,
                }
                for d in pack.drainage_specs
            ],
        },
        "eave_details": [
            {
                "node_index": e.node_index,
                "azimuth_deg": e.azimuth_deg,
                "drip_edge_length_mm": e.drip_edge_length_mm,
                "overhang_mm": e.overhang_mm,
            }
            for e in pack.eave_details
        ],
        "summary": {
            "total_gasket_m": pack.total_gasket_length_m,
            "total_drain_holes": pack.total_drain_holes,
            "eave_nodes": len(pack.eave_details),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Weather protection report written to %s", path)
