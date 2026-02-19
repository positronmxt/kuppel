"""Covering material system for the geodesic dome.

Generalises the glass-panel functionality in ``panels.py`` to support multiple
covering types:

- **glass** — monolithic float/tempered/laminated glass (legacy default).
- **polycarbonate_twin** — twin-wall (fluted) polycarbonate sheet, the most
  common greenhouse covering.  Comes in 4 / 6 / 8 / 10 / 16 mm thicknesses.
- **polycarbonate_triple** — three-wall polycarbonate, higher insulation.
- **polycarbonate_solid** — solid (monolithic) polycarbonate sheet.

Each covering type carries material property data used by:

- The geometry pipeline  (thickness, thermal-expansion gap adjustment).
- BOM / cost estimation  (weight, U-value, price range).
- IFC / metadata export  (material label, category).

The module also defines **attachment profiles** (H- and U-extrusions) used
to hold polycarbonate sheets in place between struts.

Usage::

    from freecad_dome.covering import CoveringSpec, COVERINGS, covering_for_params

    spec = covering_for_params(params)
    gap  = spec.effective_edge_gap_m(span_m=0.5, base_gap_m=0.01)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .parameters import DomeParameters

__all__ = [
    "CoveringSpec",
    "AttachmentProfile",
    "COVERINGS",
    "ATTACHMENT_PROFILES",
    "covering_for_params",
    "effective_edge_gap_m",
    "covering_bom_rows",
    "covering_cut_list",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CoveringSpec:
    """Physical properties of a covering material."""

    key: str                          # e.g. "polycarbonate_twin"
    label: str                        # Human-readable, e.g. "Twin-wall polycarbonate 8mm"
    category: str                     # "glass" | "polycarbonate"
    thickness_m: float                # Nominal sheet thickness
    density_kg_m3: float              # Bulk density (for weight calc)
    u_value_w_m2k: float              # Thermal transmittance
    thermal_expansion_mm_m_k: float   # Linear expansion coeff (mm/m/K)
    uv_coating: bool                  # Whether one side has UV protection
    min_bend_radius_m: float          # 0.0 for flat-only materials
    ifc_type: str                     # IFC entity type for export
    color_rgb: Tuple[float, float, float]  # Default display colour (0-1)
    transparency: int                 # 0-100 for FreeCAD display

    @property
    def weight_kg_m2(self) -> float:
        """Surface weight in kg/m²."""
        return self.density_kg_m3 * self.thickness_m

    def effective_edge_gap_m(
        self,
        span_m: float,
        base_gap_m: float,
        delta_t_k: float = 40.0,
    ) -> float:
        """Total edge gap accounting for thermal expansion.

        Parameters
        ----------
        span_m : float
            Panel edge span in metres (longest edge).
        base_gap_m : float
            Structural clearance gap (non-thermal).
        delta_t_k : float
            Design temperature swing in Kelvin (default 40 K covers
            typical greenhouse -10 °C … +30 °C).
        """
        thermal_gap = self.thermal_expansion_mm_m_k * span_m * delta_t_k / 1000.0
        return base_gap_m + thermal_gap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "category": self.category,
            "thickness_m": self.thickness_m,
            "weight_kg_m2": round(self.weight_kg_m2, 3),
            "u_value_w_m2k": self.u_value_w_m2k,
            "thermal_expansion_mm_m_k": self.thermal_expansion_mm_m_k,
            "uv_coating": self.uv_coating,
            "ifc_type": self.ifc_type,
        }


@dataclass(slots=True)
class AttachmentProfile:
    """Cross-section properties of a glazing bar / connecting profile."""

    key: str                  # "H_profile" | "U_profile" | "snap_cap"
    label: str
    width_m: float            # Profile total width
    depth_m: float            # How deep the sheet channel is
    grip_m: float             # How far the sheet overlaps into the channel
    material: str             # "aluminum" | "polycarbonate" | "rubber"
    weight_kg_m: float        # Linear weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "width_m": self.width_m,
            "depth_m": self.depth_m,
            "grip_m": self.grip_m,
            "material": self.material,
            "weight_kg_m": round(self.weight_kg_m, 4),
        }


# ---------------------------------------------------------------------------
# Material catalogue
# ---------------------------------------------------------------------------

COVERINGS: Dict[str, CoveringSpec] = {
    "glass": CoveringSpec(
        key="glass",
        label="Tempered glass 4mm",
        category="glass",
        thickness_m=0.004,
        density_kg_m3=2500.0,
        u_value_w_m2k=5.8,
        thermal_expansion_mm_m_k=0.009,   # ~9 µm/(m·K) — negligible
        uv_coating=False,
        min_bend_radius_m=0.0,
        ifc_type="IfcPlate",
        color_rgb=(0.2, 0.6, 1.0),
        transparency=80,
    ),
    "polycarbonate_twin_4": CoveringSpec(
        key="polycarbonate_twin_4",
        label="Twin-wall polycarbonate 4mm",
        category="polycarbonate",
        thickness_m=0.004,
        density_kg_m3=800.0,
        u_value_w_m2k=3.9,
        thermal_expansion_mm_m_k=0.065,   # ~65 µm/(m·K)
        uv_coating=True,
        min_bend_radius_m=0.7,
        ifc_type="IfcCovering",
        color_rgb=(0.85, 0.92, 0.97),
        transparency=70,
    ),
    "polycarbonate_twin_6": CoveringSpec(
        key="polycarbonate_twin_6",
        label="Twin-wall polycarbonate 6mm",
        category="polycarbonate",
        thickness_m=0.006,
        density_kg_m3=875.0,
        u_value_w_m2k=3.6,
        thermal_expansion_mm_m_k=0.065,
        uv_coating=True,
        min_bend_radius_m=1.05,
        ifc_type="IfcCovering",
        color_rgb=(0.85, 0.92, 0.97),
        transparency=68,
    ),
    "polycarbonate_twin_8": CoveringSpec(
        key="polycarbonate_twin_8",
        label="Twin-wall polycarbonate 8mm",
        category="polycarbonate",
        thickness_m=0.008,
        density_kg_m3=950.0,
        u_value_w_m2k=3.3,
        thermal_expansion_mm_m_k=0.065,
        uv_coating=True,
        min_bend_radius_m=1.4,
        ifc_type="IfcCovering",
        color_rgb=(0.85, 0.92, 0.97),
        transparency=65,
    ),
    "polycarbonate_twin_10": CoveringSpec(
        key="polycarbonate_twin_10",
        label="Twin-wall polycarbonate 10mm",
        category="polycarbonate",
        thickness_m=0.010,
        density_kg_m3=1050.0,
        u_value_w_m2k=3.0,
        thermal_expansion_mm_m_k=0.065,
        uv_coating=True,
        min_bend_radius_m=1.75,
        ifc_type="IfcCovering",
        color_rgb=(0.85, 0.92, 0.97),
        transparency=62,
    ),
    "polycarbonate_twin_16": CoveringSpec(
        key="polycarbonate_twin_16",
        label="Twin-wall polycarbonate 16mm",
        category="polycarbonate",
        thickness_m=0.016,
        density_kg_m3=1200.0,
        u_value_w_m2k=2.3,
        thermal_expansion_mm_m_k=0.065,
        uv_coating=True,
        min_bend_radius_m=2.8,
        ifc_type="IfcCovering",
        color_rgb=(0.82, 0.90, 0.96),
        transparency=55,
    ),
    "polycarbonate_triple": CoveringSpec(
        key="polycarbonate_triple",
        label="Triple-wall polycarbonate 16mm",
        category="polycarbonate",
        thickness_m=0.016,
        density_kg_m3=1400.0,
        u_value_w_m2k=2.0,
        thermal_expansion_mm_m_k=0.065,
        uv_coating=True,
        min_bend_radius_m=3.5,
        ifc_type="IfcCovering",
        color_rgb=(0.80, 0.88, 0.94),
        transparency=50,
    ),
    "polycarbonate_solid_3": CoveringSpec(
        key="polycarbonate_solid_3",
        label="Solid polycarbonate 3mm",
        category="polycarbonate",
        thickness_m=0.003,
        density_kg_m3=1200.0,
        u_value_w_m2k=5.4,
        thermal_expansion_mm_m_k=0.065,
        uv_coating=True,
        min_bend_radius_m=0.45,
        ifc_type="IfcCovering",
        color_rgb=(0.88, 0.94, 0.98),
        transparency=75,
    ),
}

ATTACHMENT_PROFILES: Dict[str, AttachmentProfile] = {
    "H_profile_alu": AttachmentProfile(
        key="H_profile_alu",
        label="Aluminium H-profile 10mm",
        width_m=0.040,
        depth_m=0.010,
        grip_m=0.008,
        material="aluminum",
        weight_kg_m=0.180,
    ),
    "H_profile_pc": AttachmentProfile(
        key="H_profile_pc",
        label="Polycarbonate H-profile 8mm",
        width_m=0.035,
        depth_m=0.008,
        grip_m=0.006,
        material="polycarbonate",
        weight_kg_m=0.050,
    ),
    "U_profile_alu": AttachmentProfile(
        key="U_profile_alu",
        label="Aluminium U-profile 10mm",
        width_m=0.020,
        depth_m=0.010,
        grip_m=0.008,
        material="aluminum",
        weight_kg_m=0.100,
    ),
    "snap_cap_alu": AttachmentProfile(
        key="snap_cap_alu",
        label="Snap-cap glazing bar 16mm",
        width_m=0.060,
        depth_m=0.016,
        grip_m=0.012,
        material="aluminum",
        weight_kg_m=0.350,
    ),
}


# ---------------------------------------------------------------------------
# Lookup / factory
# ---------------------------------------------------------------------------

def covering_for_params(params: DomeParameters) -> CoveringSpec:
    """Return the CoveringSpec matching the current parameters.

    Falls back to a glass spec derived from ``glass_thickness_m`` when
    ``covering_type`` is ``"glass"`` and no exact catalogue match exists.
    """
    key = params.covering_type

    # Direct catalogue match.
    if key in COVERINGS:
        spec = COVERINGS[key]
        # Override thickness from params if explicitly set.
        thickness = params.covering_thickness_m
        if thickness > 0 and abs(thickness - spec.thickness_m) > 1e-6:
            # Clone with custom thickness.
            import copy
            spec = copy.copy(spec)
            object.__setattr__(spec, "thickness_m", thickness)
        return spec

    # Legacy glass compatibility: use glass_thickness_m.
    if key == "glass" or key == "" or key == "none":
        thickness = params.covering_thickness_m
        if thickness <= 0:
            thickness = params.glass_thickness_m
        if thickness <= 0:
            thickness = 0.004  # default 4mm glass
        import copy
        spec = copy.copy(COVERINGS["glass"])
        object.__setattr__(spec, "thickness_m", thickness)
        return spec

    log.warning("Unknown covering_type '%s'; falling back to glass", key)
    return COVERINGS["glass"]


def effective_edge_gap_m(
    params: DomeParameters,
    span_m: float,
) -> float:
    """Compute the effective edge gap for a panel, including thermal expansion."""
    spec = covering_for_params(params)
    base_gap = params.covering_gap_m if params.covering_gap_m > 0 else params.glass_gap_m
    return spec.effective_edge_gap_m(span_m, base_gap, params.covering_delta_t_k)


# ---------------------------------------------------------------------------
# BOM helpers
# ---------------------------------------------------------------------------

def covering_bom_rows(
    panel_areas_m2: List[float],
    params: DomeParameters,
) -> List[Dict[str, Any]]:
    """Generate BOM rows for covering material.

    Parameters
    ----------
    panel_areas_m2 : list of float
        Area of each panel that gets a covering sheet.
    params : DomeParameters
        Current configuration.
    """
    spec = covering_for_params(params)
    total_area = sum(panel_areas_m2)
    total_weight = total_area * spec.weight_kg_m2

    rows: List[Dict[str, Any]] = [
        {
            "item": spec.label,
            "type": "covering",
            "material": spec.category,
            "quantity_panels": len(panel_areas_m2),
            "total_area_m2": round(total_area, 3),
            "thickness_m": spec.thickness_m,
            "weight_kg_total": round(total_weight, 2),
            "u_value_w_m2k": spec.u_value_w_m2k,
        },
    ]

    # Attachment profiles (if configured).
    profile_key = params.covering_profile_type
    if profile_key != "none" and profile_key in ATTACHMENT_PROFILES:
        profile = ATTACHMENT_PROFILES[profile_key]
        # Approximate total profile length: sum of panel perimeters.
        # Each internal edge is shared by two panels → /2 for internal,
        # but belt edges are single. A rough estimate:
        # total_perimeter ≈ total_area / avg_panel_size * avg_perimeter
        # Simpler: each panel's edges, halved for sharing.
        avg_edge = math.sqrt(total_area / max(len(panel_areas_m2), 1)) if panel_areas_m2 else 0
        n_edges_per_panel = 5.5  # mix of pentagons (5) and hexagons (6)
        total_length = avg_edge * n_edges_per_panel * len(panel_areas_m2) * 0.5
        rows.append({
            "item": profile.label,
            "type": "attachment_profile",
            "material": profile.material,
            "total_length_m": round(total_length, 2),
            "weight_kg_total": round(total_length * profile.weight_kg_m, 2),
        })

    return rows


def covering_cut_list(
    panel_data: List[Dict[str, Any]],
    params: DomeParameters,
) -> List[Dict[str, Any]]:
    """Generate a cut-list for covering sheets.

    Parameters
    ----------
    panel_data : list of dict
        Each dict must have ``panel_index``, ``area_m2``, ``edge_lengths_m``
        (list of edge lengths around the panel).
    params : DomeParameters
        Current configuration.

    Returns
    -------
    list of dict
        One entry per panel with effective dimensions and gaps.
    """
    spec = covering_for_params(params)
    base_gap = params.covering_gap_m if params.covering_gap_m > 0 else params.glass_gap_m

    result: List[Dict[str, Any]] = []
    for pd in panel_data:
        edges = pd.get("edge_lengths_m", [])
        max_span = max(edges) if edges else 0.0
        gap = spec.effective_edge_gap_m(max_span, base_gap, params.covering_delta_t_k)

        result.append({
            "panel_index": pd["panel_index"],
            "covering_type": spec.key,
            "area_m2": round(pd["area_m2"], 4),
            "max_span_m": round(max_span, 4),
            "effective_gap_m": round(gap, 4),
            "thermal_gap_m": round(gap - base_gap, 4),
            "thickness_m": spec.thickness_m,
            "uv_side": "outer" if spec.uv_coating else "none",
        })

    return result


def write_covering_report(
    panel_areas_m2: List[float],
    params: DomeParameters,
    path: Any,
) -> None:
    """Write a JSON report with covering material details and BOM."""
    import json
    from pathlib import Path

    spec = covering_for_params(params)
    bom = covering_bom_rows(panel_areas_m2, params)

    report = {
        "covering": spec.to_dict(),
        "panel_count": len(panel_areas_m2),
        "total_area_m2": round(sum(panel_areas_m2), 3),
        "total_weight_kg": round(sum(panel_areas_m2) * spec.weight_kg_m2, 2),
        "bom": bom,
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote covering report to %s", out)
