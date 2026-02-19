"""Configuration stack and parameter management for the dome generator.

This module centralizes parameter defaults, layering rules, and helpers for
writing them into FreeCAD VarSets or spreadsheets. The loader operates in three
layers ordered from lowest to highest precedence:

1. JSON file (primary) — persistent project configuration.
2. CLI overrides — runtime tweaks for automation/headless workflows.
3. Spreadsheet overrides — optional GUI editing inside FreeCAD documents.

The interface is pure Python so unit tests can run outside FreeCAD; VarSet and
Spreadsheet sync methods import FreeCAD lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import logging
import json

log = logging.getLogger(__name__)

__all__ = [
    "GeometryConfig",
    "StructureConfig",
    "CoveringConfig",
    "OpeningsConfig",
    "FoundationConfig",
    "ExportConfig",
    "CostingConfig",
    "MultiDomeConfig",
    "MaterialSpec",
    "DomeParameters",
    "load_json_config",
    "apply_overrides",
    "parse_cli_overrides",
    "load_parameters",
    "prompt_parameters_dialog",
    "write_varset",
]


@dataclass(slots=True)
class MaterialSpec:
    """Defines an IFC-ready material entry for struts.

    Strength values follow Eurocode material standards:
    - Timber: EN 338 strength classes (e.g. C24)
    - Steel: EN 10025 (e.g. S235)
    - Aluminum: EN 755 / EN 1999 (e.g. 6063-T5)
    """

    name: str
    density: float | None = None  # kg/m³
    elastic_modulus: float | None = None  # Pa
    # Structural strength properties (MPa).
    compressive_strength_mpa: float | None = None  # f_c,0,k (timber) or f_y (metal)
    tensile_strength_mpa: float | None = None  # f_t,0,k (timber) or f_y (metal)
    bending_strength_mpa: float | None = None  # f_m,k (timber) or f_y (metal)
    shear_strength_mpa: float | None = None  # f_v,k (timber) or 0.58·f_y (metal)
    # Partial safety factor γ_M for ULS capacity (EN 1995: 1.3 timber, EN 1993: 1.0 steel).
    gamma_m: float = 1.3
    ifc_label: str = "IfcMaterial"

    def to_ifc_dict(self) -> Dict[str, Any]:
        data = {
            "Name": self.name,
            "Category": self.ifc_label,
        }
        if self.density is not None:
            data["Density"] = self.density
        if self.elastic_modulus is not None:
            data["ElasticModulus"] = self.elastic_modulus
        return data


@dataclass
class GeometryConfig:
    """Sphere shape parameters."""

    radius_m: float = 3.0
    frequency: int = 4
    truncation_ratio: float = 0.18
    hemisphere_ratio: float = 0.625


@dataclass
class StructureConfig:
    """Materials, struts, and construction parameters."""

    stock_width_m: float = 0.05
    stock_height_m: float = 0.05
    kerf_m: float = 0.002
    clearance_m: float = 0.003
    material: str = "wood"
    use_bevels: bool = True
    use_truncation: bool = True
    panels_only: bool = False
    generate_struts: bool = True
    generate_belt_cap: bool = False
    node_fit_plane_mode: str = "radial"
    node_fit_use_separation_planes: bool = True
    node_fit_extension_m: float = 0.005
    node_fit_mode: str = "planar"
    node_fit_taper_ratio: float = 0.6
    split_struts_per_panel: bool = False
    # Strut geometry tuning factors (previously hardcoded).
    min_strut_length_factor: float = 0.5
    prism_only_length_factor: float = 3.0
    cap_length_factor: float = 2.0
    max_cap_ratio: float = 0.45
    split_keep_offset_factor: float = 0.35
    min_wedge_angle_deg: float = 15.0
    bevel_fillet_radius_m: float = 0.0
    cap_blend_mode: str = "sharp"
    strut_profile: str = "rectangular"
    # Connector-aware strut shortening.
    connector_strut_inset: bool = True
    generate_node_connectors: bool = False
    node_connector_type: str = "plate"
    node_connector_thickness_m: float = 0.006
    node_connector_bolt_diameter_m: float = 0.010
    node_connector_bolt_length_m: float = 0.060
    node_connector_washer_diameter_m: float = 0.020
    node_connector_bolt_offset_m: float = 0.025
    node_connector_lap_extension_m: float = 0.03
    materials: Dict[str, MaterialSpec] = field(
        default_factory=lambda: {
            "wood": MaterialSpec(
                name="Wood C24",
                density=420.0,
                elastic_modulus=11_000e6,  # 11 000 MPa → Pa
                compressive_strength_mpa=21.0,  # f_c,0,k  EN 338
                tensile_strength_mpa=14.5,  # f_t,0,k  EN 338
                bending_strength_mpa=24.0,  # f_m,k    EN 338
                shear_strength_mpa=4.0,  # f_v,k    EN 338
                gamma_m=1.3,  # EN 1995-1-1 §2.4.1
                ifc_label="IfcMaterialWood",
            ),
            "aluminum": MaterialSpec(
                name="Aluminum 6063-T5",
                density=2700.0,
                elastic_modulus=69_000e6,
                compressive_strength_mpa=185.0,  # f_0.2 proof
                tensile_strength_mpa=185.0,
                bending_strength_mpa=185.0,
                shear_strength_mpa=107.0,  # ≈ 0.58·f_y
                gamma_m=1.1,  # EN 1999-1-1
                ifc_label="IfcMaterial",
            ),
            "steel": MaterialSpec(
                name="Steel S235",
                density=7850.0,
                elastic_modulus=210_000e6,
                compressive_strength_mpa=235.0,
                tensile_strength_mpa=235.0,
                bending_strength_mpa=235.0,
                shear_strength_mpa=136.0,  # ≈ 0.58·f_y
                gamma_m=1.0,  # EN 1993-1-1
                ifc_label="IfcMaterial",
            ),
        }
    )


@dataclass
class CoveringConfig:
    """Panel covering and glazing parameters."""

    generate_panel_faces: bool = True
    generate_panel_frames: bool = False
    panel_frame_inset_m: float = 0.0
    panel_frame_profile_width_m: float = 0.04
    panel_frame_profile_height_m: float = 0.015
    glass_thickness_m: float = 0.0
    glass_gap_m: float = 0.01
    covering_type: str = "glass"
    covering_thickness_m: float = 0.0
    covering_gap_m: float = 0.0
    covering_delta_t_k: float = 40.0
    covering_profile_type: str = "none"
    generate_weather: bool = False
    gasket_type: str = "epdm_d_10x8"

    # --- Skylights / windows ---
    generate_skylights: bool = False
    skylight_count: int = 1
    skylight_position: str = "apex"           # "apex" | "ring" | "manual"
    skylight_panel_indices: List[int] = field(default_factory=list)
    skylight_glass_thickness_m: float = 0.006
    skylight_frame_width_m: float = 0.05
    skylight_hinge_side: str = "top"          # "top" | "bottom" | "left" | "right"
    skylight_material: str = "glass"          # "glass" | "polycarbonate"


@dataclass
class OpeningsConfig:
    """Doors, walls, porches, and ventilation parameters."""

    generate_base_wall: bool = False
    base_wall_height_m: float = 2.1
    base_wall_thickness_m: float = 0.15
    door_width_m: float = 0.9
    door_height_m: float = 2.1
    door_angle_deg: float = 0.0
    auto_door_angle: bool = False
    door_clearance_m: float = 0.01
    generate_entry_porch: bool = False
    porch_depth_m: float = 0.5
    porch_width_m: float = 1.2
    porch_height_m: float = 2.1
    porch_member_size_m: float = 0.045
    porch_glass_thickness_m: float = 0.006
    generate_ventilation: bool = False
    ventilation_mode: str = "auto"
    ventilation_target_ratio: float = 0.20
    ventilation_apex_count: int = 1
    ventilation_ring_count: int = 6
    ventilation_ring_height_ratio: float = 0.5
    ventilation_panel_indices: List[int] = field(default_factory=list)

    # --- Riser wall (pikendusring) ---
    generate_riser_wall: bool = False
    riser_height_m: float = 1.0
    riser_thickness_m: float = 0.15
    riser_material: str = "concrete"         # "concrete" | "wood" | "steel"
    riser_connection_type: str = "flange"    # "flange" | "embed" | "bolted"
    riser_door_integration: bool = True      # cut door through riser wall too
    riser_stud_spacing_m: float = 0.6        # stud spacing for wood riser
    riser_segments: int = 36                 # number of polygon segments for cylinder


@dataclass
class FoundationConfig:
    """Foundation system parameters."""

    generate_foundation: bool = False
    foundation_type: str = "strip"
    foundation_bolt_diameter_m: float = 0.016
    foundation_bolt_embed_m: float = 0.20
    foundation_bolt_protrusion_m: float = 0.10
    foundation_strip_width_m: float = 0.30
    foundation_strip_depth_m: float = 0.40
    foundation_pier_diameter_m: float = 0.30
    foundation_pier_depth_m: float = 0.60


@dataclass
class ExportConfig:
    """Export, analysis, and reporting parameters."""

    generate_spreadsheets: bool = False
    generate_production: bool = False
    generate_loads: bool = False
    load_wind_speed_ms: float = 25.0
    load_wind_terrain: str = "II"
    load_wind_direction_deg: float = 0.0
    load_snow_zone: str = "III"
    load_snow_exposure: float = 1.0
    load_snow_thermal: float = 1.0
    generate_structural_check: bool = False
    generate_cnc_export: bool = False
    # TechDraw
    generate_techdraw: bool = False
    techdraw_page_format: str = "A3"       # A2 / A3 / A4
    techdraw_scale: float = 0.02           # drawing scale (e.g. 1:50 = 0.02)
    techdraw_views: str = "all"            # "all" | "overview" | "parts" | "nodes"
    techdraw_project_name: str = ""
    techdraw_version: str = ""
    # Assembly guide
    generate_assembly_guide: bool = False
    assembly_time_per_strut_min: float = 15.0   # minutes per strut install
    assembly_time_per_node_min: float = 10.0    # minutes per node/connector
    assembly_time_per_panel_min: float = 20.0   # minutes per panel install
    assembly_workers: int = 2                   # crew size


@dataclass
class CostingConfig:
    """Cost estimation parameters.

    Unit-price overrides: when a price field is > 0 it overrides the
    corresponding ``DEFAULT_PRICE_CATALOGUE`` entry.  Setting a value
    to 0.0 (the default) means "use catalogue default".

    An external JSON price catalogue can be loaded via
    ``price_catalogue_path`` — it is merged on top of the defaults and
    below explicit overrides.
    """

    generate_costing: bool = False

    # --- currency ---
    currency: str = "EUR"  # EUR / USD / GBP

    # --- unit-price overrides (0 = use catalogue default) ---
    timber_price_per_m: float = 0.0
    covering_price_per_m2: float = 0.0
    connector_plate_price_per_m2: float = 0.0
    gasket_price_per_m: float = 0.0
    bolt_price_each: float = 0.0

    # --- waste / markup (percentages) ---
    waste_timber_pct: float = 10.0
    waste_covering_pct: float = 8.0
    overhead_pct: float = 0.0  # general overhead markup on total

    # --- labour ---
    labor_install_eur_h: float = 0.0  # installation €/h
    labor_cnc_eur_h: float = 0.0      # CNC processing €/h
    estimated_install_hours: float = 0.0
    estimated_cnc_hours: float = 0.0

    # --- external catalogue ---
    price_catalogue_path: str = ""  # path to JSON file


@dataclass
class MultiDomeConfig:
    """Multi-dome project parameters.

    When ``multi_dome_enabled`` is True the project contains multiple domes
    that can be connected by corridors.  Each secondary dome is defined as
    a JSON-serialisable dict with positional offset and optional parameter
    overrides (e.g. a smaller annex dome).

    Corridors are rectangular passages connecting pairs of domes at their
    belt level.  The ``corridor_definitions`` list holds JSON dicts with
    keys: ``from_dome`` (int), ``to_dome`` (int), ``width_m``, ``height_m``.
    """

    multi_dome_enabled: bool = False
    # JSON-encoded list of dome instance dicts:
    #   [{"label": "Annex", "offset_x_m": 8.0, "offset_y_m": 0.0,
    #     "rotation_deg": 0.0, "overrides": {"radius_m": 3.0}}]
    dome_instances_json: str = "[]"

    # Corridor geometry
    corridor_width_m: float = 1.2      # passage width
    corridor_height_m: float = 2.1     # passage height
    corridor_wall_thickness_m: float = 0.15  # wall/frame thickness
    corridor_material: str = "wood"    # "wood" | "steel" | "glass"

    # JSON-encoded list of corridor definitions:
    #   [{"from_dome": 0, "to_dome": 1}]
    corridor_definitions_json: str = "[]"

    # Shared outputs
    merge_foundation: bool = True      # combine all dome foundations
    merge_bom: bool = True             # combine BOM across domes


# ---------------------------------------------------------------------------
# Field → sub-config mapping (built once at import time)
# ---------------------------------------------------------------------------

_SUB_CONFIGS: List[Tuple[str, type]] = [
    ("geometry", GeometryConfig),
    ("structure", StructureConfig),
    ("covering", CoveringConfig),
    ("openings", OpeningsConfig),
    ("foundation", FoundationConfig),
    ("export_config", ExportConfig),
    ("costing", CostingConfig),
    ("multi_dome", MultiDomeConfig),
]

_SUB_CONFIG_ATTRS = frozenset(name for name, _ in _SUB_CONFIGS)

_FIELD_MAP: Dict[str, str] = {}
for _attr_name, _cfg_cls in _SUB_CONFIGS:
    for _f in dc_fields(_cfg_cls):
        _FIELD_MAP[_f.name] = _attr_name


class DomeParameters:
    """Canonical set of adjustable dome parameters.

    Internally organized into hierarchical sub-configurations
    (:class:`GeometryConfig`, :class:`StructureConfig`, :class:`CoveringConfig`,
    :class:`OpeningsConfig`, :class:`FoundationConfig`, :class:`ExportConfig`,
    :class:`CostingConfig`, :class:`MultiDomeConfig`).

    All fields remain accessible as flat attributes for backward compatibility::

        params.radius_m          # equivalent to params.geometry.radius_m
        params.stock_width_m     # equivalent to params.structure.stock_width_m
    """

    def __init__(self, **kwargs: Any) -> None:
        sub_kwargs: Dict[str, Dict[str, Any]] = {n: {} for n, _ in _SUB_CONFIGS}
        direct: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in _SUB_CONFIG_ATTRS:
                direct[key] = value
            elif key in _FIELD_MAP:
                sub_kwargs[_FIELD_MAP[key]][key] = value
            else:
                raise TypeError(
                    f"DomeParameters() got an unexpected keyword argument '{key}'"
                )
        for attr_name, cfg_cls in _SUB_CONFIGS:
            if attr_name in direct:
                if sub_kwargs.get(attr_name):
                    raise TypeError(
                        f"Cannot mix '{attr_name}=' with flat field kwargs "
                        f"({', '.join(sub_kwargs[attr_name])})"
                    )
                object.__setattr__(self, attr_name, direct[attr_name])
            else:
                object.__setattr__(
                    self, attr_name, cfg_cls(**sub_kwargs.get(attr_name, {}))
                )
        # panels_only implies generate_struts = False
        if self.panels_only:
            self.generate_struts = False

    # ------------------------------------------------------------------
    # Transparent flat-attribute access (backward compatibility)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        cfg_attr = _FIELD_MAP.get(name)
        if cfg_attr is not None:
            sub = object.__getattribute__(self, cfg_attr)
            return getattr(sub, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _SUB_CONFIG_ATTRS:
            object.__setattr__(self, name, value)
        elif name in _FIELD_MAP:
            sub = object.__getattribute__(self, _FIELD_MAP[name])
            object.__setattr__(sub, name, value)
        else:
            object.__setattr__(self, name, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DomeParameters):
            return NotImplemented
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr, _ in _SUB_CONFIGS
        )

    def __repr__(self) -> str:
        parts: List[str] = []
        for attr_name, cfg_cls in _SUB_CONFIGS:
            sub = object.__getattribute__(self, attr_name)
            for f in dc_fields(cfg_cls):
                parts.append(f"{f.name}={getattr(sub, f.name)!r}")
        return f"DomeParameters({', '.join(parts)})"

    def validate(self) -> None:
        if self.radius_m <= 0:
            raise ValueError("Radius must be positive")
        if self.frequency < 1:
            raise ValueError("Frequency must be at least 1")
        if not 0 < self.truncation_ratio < 1:
            raise ValueError("Truncation ratio must be within (0, 1)")
        if not 0 < self.hemisphere_ratio <= 1:
            raise ValueError("Hemisphere ratio must be within (0, 1]")
        if min(self.stock_width_m, self.stock_height_m) <= 0:
            raise ValueError("Stock dimensions must be positive")
        if self.kerf_m < 0:
            raise ValueError("Kerf cannot be negative")
        if self.clearance_m < 0:
            raise ValueError("Clearance cannot be negative")
        if self.material not in self.materials:
            raise ValueError(f"Unknown material '{self.material}'")
        if self.panel_frame_inset_m < 0:
            raise ValueError("Panel frame inset cannot be negative")
        if self.panel_frame_profile_width_m <= 0:
            raise ValueError("Panel frame profile width must be positive")
        if self.panel_frame_profile_height_m <= 0:
            raise ValueError("Panel frame profile height must be positive")
        if self.glass_thickness_m < 0:
            raise ValueError("Glass thickness cannot be negative")
        if self.glass_gap_m < 0:
            raise ValueError("Glass gap cannot be negative")

        if self.node_fit_plane_mode not in {"radial", "axis", "miter"}:
            raise ValueError("node_fit_plane_mode must be 'radial', 'axis', or 'miter'")
        if self.node_fit_extension_m < 0:
            raise ValueError("node_fit_extension_m cannot be negative")
        if self.node_fit_mode not in {"planar", "tapered", "voronoi"}:
            raise ValueError("node_fit_mode must be 'planar', 'tapered', or 'voronoi'")
        if self.node_fit_taper_ratio <= 0 or self.node_fit_taper_ratio >= 1.0:
            raise ValueError("node_fit_taper_ratio must be between 0 and 1 (exclusive)")

        if self.strut_profile not in {"rectangular", "round", "trapezoidal"}:
            raise ValueError("strut_profile must be 'rectangular', 'round', or 'trapezoidal'")

        if self.cap_blend_mode not in {"sharp", "chamfer", "fillet"}:
            raise ValueError("cap_blend_mode must be 'sharp', 'chamfer', or 'fillet'")

        # Connector-strut compatibility warning.
        if self.generate_node_connectors and self.node_fit_plane_mode == "axis":
            log.warning(
                "node_fit_plane_mode='axis' is not compatible with plate connectors "
                "(strut end face won't be parallel to the plate). "
                "Consider switching to 'radial'."
            )
        if self.generate_node_connectors and self.node_fit_plane_mode == "miter":
            log.warning(
                "node_fit_plane_mode='miter' with plate connectors: strut end face "
                "won't be parallel to the plate. Consider using 'radial' mode."
            )

        if bool(getattr(self, "generate_base_wall", False)):
            if float(getattr(self, "base_wall_height_m", 0.0) or 0.0) <= 0:
                raise ValueError("base_wall_height_m must be positive when generate_base_wall is enabled")
            if float(getattr(self, "base_wall_thickness_m", 0.0) or 0.0) <= 0:
                raise ValueError("base_wall_thickness_m must be positive when generate_base_wall is enabled")
            if float(getattr(self, "door_width_m", 0.0) or 0.0) <= 0:
                raise ValueError("door_width_m must be positive when generate_base_wall is enabled")
            if float(getattr(self, "door_height_m", 0.0) or 0.0) <= 0:
                raise ValueError("door_height_m must be positive when generate_base_wall is enabled")
            if float(getattr(self, "door_clearance_m", 0.0) or 0.0) < 0:
                raise ValueError("door_clearance_m cannot be negative")

        if bool(getattr(self, "generate_base_wall", False)) and bool(getattr(self, "generate_entry_porch", False)):
            raise ValueError("Enable either base wall or entry porch (not both)")

        if bool(getattr(self, "generate_entry_porch", False)):
            if float(getattr(self, "porch_depth_m", 0.0) or 0.0) <= 0:
                raise ValueError("porch_depth_m must be positive when generate_entry_porch is enabled")
            if float(getattr(self, "porch_depth_m", 0.0) or 0.0) > 0.5 + 1e-9:
                raise ValueError("porch_depth_m must be <= 0.5m")
            if float(getattr(self, "porch_width_m", 0.0) or 0.0) <= 0:
                raise ValueError("porch_width_m must be positive when generate_entry_porch is enabled")
            if float(getattr(self, "porch_height_m", 0.0) or 0.0) <= 0:
                raise ValueError("porch_height_m must be positive when generate_entry_porch is enabled")
            if float(getattr(self, "porch_member_size_m", 0.0) or 0.0) <= 0:
                raise ValueError("porch_member_size_m must be positive when generate_entry_porch is enabled")
            if float(getattr(self, "porch_glass_thickness_m", 0.0) or 0.0) < 0:
                raise ValueError("porch_glass_thickness_m cannot be negative")

        if self.ventilation_mode not in {"auto", "apex", "ring", "manual"}:
            raise ValueError("ventilation_mode must be 'auto', 'apex', 'ring', or 'manual'")
        if self.ventilation_target_ratio < 0 or self.ventilation_target_ratio > 1:
            raise ValueError("ventilation_target_ratio must be in [0, 1]")
        if self.ventilation_apex_count < 0:
            raise ValueError("ventilation_apex_count cannot be negative")
        if self.ventilation_ring_count < 0:
            raise ValueError("ventilation_ring_count cannot be negative")

        if self.node_connector_type not in {"plate", "ball", "pipe", "lapjoint"}:
            raise ValueError("node_connector_type must be 'plate', 'ball', 'pipe', or 'lapjoint'")
        if self.node_connector_thickness_m <= 0:
            raise ValueError("node_connector_thickness_m must be positive")
        if self.node_connector_bolt_diameter_m <= 0:
            raise ValueError("node_connector_bolt_diameter_m must be positive")
        if self.node_connector_bolt_offset_m <= 0:
            raise ValueError("node_connector_bolt_offset_m must be positive")

        if self.covering_thickness_m < 0:
            raise ValueError("covering_thickness_m cannot be negative")
        if self.covering_gap_m < 0:
            raise ValueError("covering_gap_m cannot be negative")
        if self.covering_delta_t_k < 0:
            raise ValueError("covering_delta_t_k cannot be negative")

        if self.foundation_type not in {"strip", "point", "screw_anchor"}:
            raise ValueError("foundation_type must be 'strip', 'point', or 'screw_anchor'")
        if self.foundation_bolt_diameter_m <= 0:
            raise ValueError("foundation_bolt_diameter_m must be positive")
        if self.foundation_strip_width_m <= 0:
            raise ValueError("foundation_strip_width_m must be positive")
        if self.foundation_strip_depth_m <= 0:
            raise ValueError("foundation_strip_depth_m must be positive")

        if self.gasket_type not in {
            "epdm_d_10x8", "epdm_p_9x5", "silicone_12x8",
            "neoprene_flat_15x3", "butyl_tape_20x1",
        }:
            raise ValueError("gasket_type must be a known gasket profile")

        if self.load_wind_speed_ms < 0:
            raise ValueError("load_wind_speed_ms cannot be negative")
        if self.load_wind_terrain not in {"0", "I", "II", "III", "IV"}:
            raise ValueError("load_wind_terrain must be '0', 'I', 'II', 'III', or 'IV'")
        if self.load_snow_zone not in {"I", "II", "III", "IV", "V"}:
            raise ValueError("load_snow_zone must be 'I' through 'V'")

    def material_spec(self) -> MaterialSpec:
        return self.materials[self.material]

    def to_dict(self) -> Dict[str, Any]:
        """Return flat dict of all parameters (backward compatible)."""
        result: Dict[str, Any] = {}
        for attr_name, cfg_cls in _SUB_CONFIGS:
            sub = object.__getattribute__(self, attr_name)
            for f in dc_fields(cfg_cls):
                val = getattr(sub, f.name)
                if f.name == "materials" and isinstance(val, dict):
                    val = {
                        k: (v.to_ifc_dict() if isinstance(v, MaterialSpec) else v)
                        for k, v in val.items()
                    }
                elif isinstance(val, list):
                    val = list(val)
                result[f.name] = val
        return result

    def to_nested_dict(self) -> Dict[str, Any]:
        """Return hierarchically grouped dict keyed by sub-config name."""
        result: Dict[str, Any] = {}
        for attr_name, cfg_cls in _SUB_CONFIGS:
            sub = object.__getattribute__(self, attr_name)
            section: Dict[str, Any] = {}
            for f in dc_fields(cfg_cls):
                val = getattr(sub, f.name)
                if f.name == "materials" and isinstance(val, dict):
                    val = {
                        k: (v.to_ifc_dict() if isinstance(v, MaterialSpec) else v)
                        for k, v in val.items()
                    }
                elif isinstance(val, list):
                    val = list(val)
                section[f.name] = val
            result[attr_name] = section
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DomeParameters":
        """Create from dict — accepts both flat and nested (hierarchical) formats.

        Nested format uses sub-config keys (geometry, structure, covering,
        openings, foundation, export_config, costing) as top-level entries.
        Flat format passes field names directly.
        Legacy JSON files (flat) are automatically migrated.
        """
        # Flatten nested format if detected
        flat_data: Dict[str, Any] = {}
        for key, value in data.items():
            if key in _SUB_CONFIG_ATTRS and isinstance(value, Mapping):
                flat_data.update(value)
            else:
                flat_data[key] = value
        # Merge with defaults
        base = cls()
        merged = {**base.to_dict(), **flat_data}
        # Parse materials
        materials_raw = merged.pop("materials", base.structure.materials)
        materials: Dict[str, MaterialSpec] = {}
        if isinstance(materials_raw, Mapping):
            for key, value in materials_raw.items():
                if isinstance(value, MaterialSpec):
                    materials[key] = value
                else:
                    materials[key] = MaterialSpec(
                        name=value.get("Name", key.title()),
                        density=value.get("Density"),
                        elastic_modulus=value.get("ElasticModulus"),
                        ifc_label=value.get("Category", "IfcMaterial"),
                    )
        merged["materials"] = materials
        params = cls(**merged)
        params.validate()
        return params


def load_json_config(path: Path | str | None) -> Dict[str, Any]:
    """Load the JSON config file or return an empty dict if missing."""

    if path is None:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Config file not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Top-level JSON config must be an object")
    return dict(data)


def apply_overrides(base: DomeParameters, overrides: Mapping[str, Any]) -> DomeParameters:
    """Return a copy of ``base`` with overrides applied."""

    merged = base.to_dict()
    for key, value in overrides.items():
        if key not in merged:
            raise KeyError(f"Unknown parameter '{key}'")
        merged[key] = value
    return DomeParameters.from_dict(merged)


def parse_cli_overrides(
    args: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], Any]:
    """Parse CLI-style overrides using argparse conventions."""

    import argparse

    parser = argparse.ArgumentParser(description="Geodesic dome generator")
    parser.add_argument("--config", type=str, help="Path to JSON config", default=None)
    parser.add_argument("--out-dir", type=str, default="exports", help="Export folder")
    parser.add_argument("--manifest-name", type=str, default="dome_manifest.json")
    parser.add_argument("--ifc-name", type=str, default="dome.ifc")
    parser.add_argument("--stl-name", type=str, default="dome.stl")
    parser.add_argument("--dxf-name", type=str, default="dome.dxf")
    parser.add_argument("--skip-ifc", action="store_true", help="Disable IFC export")
    parser.add_argument("--skip-stl", action="store_true", help="Disable STL export")
    parser.add_argument("--skip-dxf", action="store_true", help="Disable DXF export")
    parser.add_argument("--radius", type=float, help="Override radius in meters")
    parser.add_argument("--frequency", type=int, help="Icosahedron subdivision order")
    parser.add_argument("--truncation", type=float, help="Truncation ratio (0-1)")
    parser.add_argument("--segment", type=float, help="Portion of sphere kept (0-1]")
    parser.add_argument(
        "--stock-size",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Rectangular stock dimensions in meters",
    )
    parser.add_argument("--kerf", type=float, help="Saw kerf in meters")
    parser.add_argument("--clearance", type=float, help="Clearance offset in meters")
    parser.add_argument(
        "--material",
        type=str,
        choices=["wood", "aluminum", "steel"],
        help="Material selection",
    )
    parser.add_argument(
        "--no-bevels",
        action="store_true",
        help="Disable beveled cuts (use simple prisms)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip the graphical parameter dialog",
    )
    parser.add_argument(
        "--no-truncation",
        action="store_true",
        help="Skip the truncation step and keep full icosahedron",
    )
    parser.add_argument(
        "--panels-only",
        action="store_true",
        help="Generate panels without creating any strut solids",
    )
    parser.add_argument(
        "--no-struts",
        action="store_true",
        help="Skip strut solid generation regardless of panels-only mode",
    )
    parser.add_argument(
        "--no-panel-surfaces",
        action="store_true",
        help="Skip creation of base panel faces (use frames only)",
    )
    parser.add_argument(
        "--panel-frames",
        action="store_true",
        help="Generate inset frames for every panel",
    )
    parser.add_argument(
        "--no-panel-frames",
        action="store_true",
        help="Disable panel frame generation",
    )
    parser.add_argument(
        "--panel-frame-inset",
        type=float,
        help="Inset distance from panel edges for frame placement (m)",
    )
    parser.add_argument(
        "--panel-frame-profile",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Panel frame profile width/height in meters",
    )
    parser.add_argument(
        "--panel-frames-only",
        action="store_true",
        help="Enable panel frames while disabling struts and panel faces",
    )

    parser.add_argument(
        "--glass-thickness",
        type=float,
        help="Glass panel thickness in meters (0 disables glass solids)",
    )
    parser.add_argument(
        "--glass-gap",
        type=float,
        help="Target edge-to-edge gap between adjacent glass panels in meters",
    )

    parser.add_argument(
        "--node-fit-plane",
        type=str,
        choices=["radial", "axis", "miter"],
        help="Strut node-fit end plane mode: 'radial' (tangent), 'axis' (square cut), or 'miter' (tight compound miter)",
    )
    parser.add_argument(
        "--node-fit-extension",
        type=float,
        help="Extend struts past the node by this amount (metres) so node-fit planes have material to carve (default 0.005)",
    )
    parser.add_argument(
        "--node-fit-mode",
        type=str,
        choices=["planar", "tapered", "voronoi"],
        help="Node-fit algorithm: 'planar' (cut planes), 'tapered' (CNC tapered ends), 'voronoi' (perfect angular partition)",
    )
    parser.add_argument(
        "--node-fit-taper-ratio",
        type=float,
        help="End cross-section ratio (0-1) for tapered mode (default 0.6)",
    )
    parser.add_argument(
        "--no-node-fit-separation",
        action="store_true",
        help="Disable node-fit separation planes (keep only the end plane cut)",
    )

    parser.add_argument(
        "--split-struts-per-panel",
        action="store_true",
        help="Split non-belt struts lengthwise so each panel edge has its own strut",
    )

    parser.add_argument(
        "--spreadsheets",
        action="store_true",
        help="Generate FreeCAD spreadsheets with parameters/parts (requires FreeCAD)",
    )

    parser.add_argument(
        "--base-wall",
        action="store_true",
        help="Generate a cylindrical base wall (knee wall) under the dome (requires FreeCAD)",
    )
    parser.add_argument(
        "--base-wall-height",
        type=float,
        help="Base wall height in meters (also sets default door height if not overridden)",
    )
    parser.add_argument(
        "--base-wall-thickness",
        type=float,
        help="Base wall thickness in meters",
    )
    parser.add_argument(
        "--door",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Door opening size in meters (e.g. 0.9 2.1)",
    )
    parser.add_argument(
        "--door-angle",
        type=float,
        help="Door azimuth angle in degrees (0 = +X axis)",
    )
    parser.add_argument(
        "--auto-door-angle",
        action="store_true",
        help="Choose door angle automatically based on dome geometry",
    )
    parser.add_argument(
        "--door-clearance",
        type=float,
        help="Extra clearance around door cutout in meters",
    )

    parser.add_argument(
        "--entry-porch",
        action="store_true",
        help="Generate a small framed/glazed entry porch at the belt opening (requires FreeCAD)",
    )
    parser.add_argument(
        "--porch-depth",
        type=float,
        help="Porch projection depth in meters (max 0.5)",
    )
    parser.add_argument(
        "--porch-width",
        type=float,
        help="Porch outer width in meters",
    )
    parser.add_argument(
        "--porch-height",
        type=float,
        help="Porch outer height in meters (measured up from belt plane)",
    )
    parser.add_argument(
        "--porch-member",
        type=float,
        help="Porch frame member size in meters (square, e.g. 0.045)",
    )
    parser.add_argument(
        "--porch-glass",
        type=float,
        help="Porch glass thickness in meters (0 disables porch glazing solids)",
    )

    # Ventilation
    parser.add_argument(
        "--ventilation",
        action="store_true",
        help="Generate ventilation plan (vent panel selection + metadata)",
    )
    parser.add_argument(
        "--ventilation-mode",
        type=str,
        choices=["auto", "apex", "ring", "manual"],
        help="Vent placement strategy",
    )
    parser.add_argument(
        "--ventilation-target",
        type=float,
        help="Target vent area as fraction of floor area (e.g. 0.20)",
    )
    parser.add_argument(
        "--ventilation-apex-count",
        type=int,
        help="Number of apex vent panels",
    )
    parser.add_argument(
        "--ventilation-ring-count",
        type=int,
        help="Number of ring vent panels",
    )
    parser.add_argument(
        "--ventilation-ring-height",
        type=float,
        help="Ring vent height ratio (0=belt, 1=apex)",
    )
    parser.add_argument(
        "--ventilation-panels",
        type=int,
        nargs="+",
        help="Explicit panel indices for manual ventilation mode",
    )

    # Node connector flags.
    parser.add_argument(
        "--node-connectors",
        action="store_true",
        help="Generate node connector (hub) geometry and BOM",
    )
    parser.add_argument(
        "--connector-type",
        type=str,
        choices=["plate", "ball", "pipe"],
        help="Node connector type",
    )
    parser.add_argument(
        "--connector-thickness",
        type=float,
        help="Connector plate thickness in meters (e.g. 0.006)",
    )
    parser.add_argument(
        "--connector-bolt-diameter",
        type=float,
        help="Bolt diameter in meters (e.g. 0.010 for M10)",
    )
    parser.add_argument(
        "--connector-bolt-offset",
        type=float,
        help="Bolt hole offset from node center in meters",
    )

    # Covering material flags.
    parser.add_argument(
        "--covering-type",
        type=str,
        help="Covering material key (e.g. glass, polycarbonate_twin_8)",
    )
    parser.add_argument(
        "--covering-thickness",
        type=float,
        help="Covering thickness in meters (overrides catalogue default)",
    )
    parser.add_argument(
        "--covering-gap",
        type=float,
        help="Covering edge gap in meters",
    )
    parser.add_argument(
        "--covering-delta-t",
        type=float,
        help="Design temperature swing in K for thermal expansion gap",
    )
    parser.add_argument(
        "--covering-profile",
        type=str,
        help="Attachment profile key (e.g. H_profile_alu, U_profile_alu, none)",
    )

    # Foundation
    parser.add_argument(
        "--foundation",
        action="store_true",
        help="Generate foundation layout with anchor bolt positions",
    )
    parser.add_argument(
        "--foundation-type",
        type=str,
        choices=["strip", "point", "screw_anchor"],
        help="Foundation type: strip footing, point piers, or screw anchors",
    )
    parser.add_argument(
        "--foundation-bolt-diameter",
        type=float,
        help="Anchor bolt diameter in meters (e.g. 0.016 for M16)",
    )
    parser.add_argument(
        "--foundation-strip-width",
        type=float,
        help="Strip footing width in meters",
    )
    parser.add_argument(
        "--foundation-strip-depth",
        type=float,
        help="Strip footing depth below grade in meters",
    )
    parser.add_argument(
        "--foundation-pier-diameter",
        type=float,
        help="Point pier diameter in meters",
    )
    parser.add_argument(
        "--foundation-pier-depth",
        type=float,
        help="Point pier depth in meters",
    )

    # Cost estimation
    parser.add_argument(
        "--costing",
        action="store_true",
        help="Generate cost estimate and full BOM",
    )

    # Weather protection
    parser.add_argument(
        "--weather",
        action="store_true",
        help="Generate weather protection data (gaskets, drainage, eave details)",
    )
    parser.add_argument(
        "--gasket-type",
        type=str,
        choices=["epdm_d_10x8", "epdm_p_9x5", "silicone_12x8", "neoprene_flat_15x3", "butyl_tape_20x1"],
        help="Gasket profile type",
    )

    # Production drawings
    parser.add_argument(
        "--production",
        action="store_true",
        help="Generate production data (cut-lists, saw table, plates, assembly plan)",
    )

    # Load calculations
    parser.add_argument(
        "--loads",
        action="store_true",
        help="Compute structural loads (dead/snow/wind) and export FEM data",
    )
    parser.add_argument(
        "--wind-speed",
        type=float,
        help="Reference wind speed in m/s (e.g. 25.0)",
    )
    parser.add_argument(
        "--wind-terrain",
        type=str,
        choices=["0", "I", "II", "III", "IV"],
        help="Wind terrain category (0=sea, II=farmland, IV=urban)",
    )
    parser.add_argument(
        "--wind-direction",
        type=float,
        help="Wind direction azimuth in degrees (0 = +X)",
    )
    parser.add_argument(
        "--snow-zone",
        type=str,
        choices=["I", "II", "III", "IV", "V"],
        help="Snow zone (I=1.0, III=2.0, V=3.0 kN/m²)",
    )
    parser.add_argument(
        "--snow-exposure",
        type=float,
        help="Snow exposure coefficient Ce (default 1.0)",
    )

    parsed, unknown = parser.parse_known_args(args=args)
    if unknown:
        logging.info("Ignoring unknown CLI args: %s", " ".join(unknown))
    overrides: Dict[str, Any] = {}
    if parsed.radius is not None:
        overrides["radius_m"] = parsed.radius
    if parsed.frequency is not None:
        overrides["frequency"] = parsed.frequency
    if parsed.truncation is not None:
        overrides["truncation_ratio"] = parsed.truncation
    if parsed.segment is not None:
        overrides["hemisphere_ratio"] = parsed.segment
    if parsed.stock_size is not None:
        overrides["stock_width_m"], overrides["stock_height_m"] = parsed.stock_size
    if parsed.kerf is not None:
        overrides["kerf_m"] = parsed.kerf
    if parsed.clearance is not None:
        overrides["clearance_m"] = parsed.clearance
    if parsed.material is not None:
        overrides["material"] = parsed.material
    if parsed.no_bevels:
        overrides["use_bevels"] = False
    if parsed.no_truncation:
        overrides["use_truncation"] = False
    if parsed.panels_only:
        overrides["panels_only"] = True
        overrides["generate_struts"] = False
    if parsed.no_struts:
        overrides["generate_struts"] = False
    if parsed.no_panel_surfaces:
        overrides["generate_panel_faces"] = False
    if parsed.panel_frames:
        overrides["generate_panel_frames"] = True
    if parsed.no_panel_frames:
        overrides["generate_panel_frames"] = False
    if parsed.panel_frames_only:
        overrides["generate_panel_frames"] = True
        overrides["generate_panel_faces"] = False
        overrides["generate_struts"] = False
    if parsed.panel_frame_inset is not None:
        overrides["panel_frame_inset_m"] = parsed.panel_frame_inset
    if parsed.panel_frame_profile is not None:
        overrides["panel_frame_profile_width_m"] = parsed.panel_frame_profile[0]
        overrides["panel_frame_profile_height_m"] = parsed.panel_frame_profile[1]
    if parsed.glass_thickness is not None:
        overrides["glass_thickness_m"] = parsed.glass_thickness
    if parsed.glass_gap is not None:
        overrides["glass_gap_m"] = parsed.glass_gap

    if getattr(parsed, "node_fit_plane", None) is not None:
        overrides["node_fit_plane_mode"] = parsed.node_fit_plane
    if getattr(parsed, "node_fit_extension", None) is not None:
        overrides["node_fit_extension_m"] = parsed.node_fit_extension
    if getattr(parsed, "node_fit_mode", None) is not None:
        overrides["node_fit_mode"] = parsed.node_fit_mode
    if getattr(parsed, "node_fit_taper_ratio", None) is not None:
        overrides["node_fit_taper_ratio"] = parsed.node_fit_taper_ratio
    if getattr(parsed, "no_node_fit_separation", False):
        overrides["node_fit_use_separation_planes"] = False

    if getattr(parsed, "split_struts_per_panel", False):
        overrides["split_struts_per_panel"] = True

    if getattr(parsed, "spreadsheets", False):
        overrides["generate_spreadsheets"] = True

    if getattr(parsed, "base_wall", False):
        overrides["generate_base_wall"] = True
    if getattr(parsed, "base_wall_height", None) is not None:
        overrides["base_wall_height_m"] = float(parsed.base_wall_height)
        # If door height wasn't explicitly provided, keep it in sync.
        if getattr(parsed, "door", None) is None:
            overrides["door_height_m"] = float(parsed.base_wall_height)
    if getattr(parsed, "base_wall_thickness", None) is not None:
        overrides["base_wall_thickness_m"] = float(parsed.base_wall_thickness)
    if getattr(parsed, "door", None) is not None:
        overrides["door_width_m"] = float(parsed.door[0])
        overrides["door_height_m"] = float(parsed.door[1])
    if getattr(parsed, "door_angle", None) is not None:
        overrides["door_angle_deg"] = float(parsed.door_angle)
    if getattr(parsed, "auto_door_angle", False):
        overrides["auto_door_angle"] = True
    if getattr(parsed, "door_clearance", None) is not None:
        overrides["door_clearance_m"] = float(parsed.door_clearance)

    if getattr(parsed, "entry_porch", False):
        overrides["generate_entry_porch"] = True
    if getattr(parsed, "porch_depth", None) is not None:
        overrides["porch_depth_m"] = float(parsed.porch_depth)
    if getattr(parsed, "porch_width", None) is not None:
        overrides["porch_width_m"] = float(parsed.porch_width)
    if getattr(parsed, "porch_height", None) is not None:
        overrides["porch_height_m"] = float(parsed.porch_height)
    if getattr(parsed, "porch_member", None) is not None:
        overrides["porch_member_size_m"] = float(parsed.porch_member)
    if getattr(parsed, "porch_glass", None) is not None:
        overrides["porch_glass_thickness_m"] = float(parsed.porch_glass)

    if getattr(parsed, "ventilation", False):
        overrides["generate_ventilation"] = True
    if getattr(parsed, "ventilation_mode", None) is not None:
        overrides["ventilation_mode"] = parsed.ventilation_mode
    if getattr(parsed, "ventilation_target", None) is not None:
        overrides["ventilation_target_ratio"] = float(parsed.ventilation_target)
    if getattr(parsed, "ventilation_apex_count", None) is not None:
        overrides["ventilation_apex_count"] = int(parsed.ventilation_apex_count)
    if getattr(parsed, "ventilation_ring_count", None) is not None:
        overrides["ventilation_ring_count"] = int(parsed.ventilation_ring_count)
    if getattr(parsed, "ventilation_ring_height", None) is not None:
        overrides["ventilation_ring_height_ratio"] = float(parsed.ventilation_ring_height)
    if getattr(parsed, "ventilation_panels", None) is not None:
        overrides["ventilation_panel_indices"] = list(parsed.ventilation_panels)

    if getattr(parsed, "node_connectors", False):
        overrides["generate_node_connectors"] = True
    if getattr(parsed, "connector_type", None) is not None:
        overrides["node_connector_type"] = parsed.connector_type
    if getattr(parsed, "connector_thickness", None) is not None:
        overrides["node_connector_thickness_m"] = float(parsed.connector_thickness)
    if getattr(parsed, "connector_bolt_diameter", None) is not None:
        overrides["node_connector_bolt_diameter_m"] = float(parsed.connector_bolt_diameter)
    if getattr(parsed, "connector_bolt_offset", None) is not None:
        overrides["node_connector_bolt_offset_m"] = float(parsed.connector_bolt_offset)

    if getattr(parsed, "covering_type", None) is not None:
        overrides["covering_type"] = parsed.covering_type
    if getattr(parsed, "covering_thickness", None) is not None:
        overrides["covering_thickness_m"] = float(parsed.covering_thickness)
    if getattr(parsed, "covering_gap", None) is not None:
        overrides["covering_gap_m"] = float(parsed.covering_gap)
    if getattr(parsed, "covering_delta_t", None) is not None:
        overrides["covering_delta_t_k"] = float(parsed.covering_delta_t)
    if getattr(parsed, "covering_profile", None) is not None:
        overrides["covering_profile_type"] = parsed.covering_profile

    if getattr(parsed, "foundation", False):
        overrides["generate_foundation"] = True
    if getattr(parsed, "foundation_type", None) is not None:
        overrides["foundation_type"] = parsed.foundation_type
    if getattr(parsed, "foundation_bolt_diameter", None) is not None:
        overrides["foundation_bolt_diameter_m"] = float(parsed.foundation_bolt_diameter)
    if getattr(parsed, "foundation_strip_width", None) is not None:
        overrides["foundation_strip_width_m"] = float(parsed.foundation_strip_width)
    if getattr(parsed, "foundation_strip_depth", None) is not None:
        overrides["foundation_strip_depth_m"] = float(parsed.foundation_strip_depth)
    if getattr(parsed, "foundation_pier_diameter", None) is not None:
        overrides["foundation_pier_diameter_m"] = float(parsed.foundation_pier_diameter)
    if getattr(parsed, "foundation_pier_depth", None) is not None:
        overrides["foundation_pier_depth_m"] = float(parsed.foundation_pier_depth)

    if getattr(parsed, "loads", False):
        overrides["generate_loads"] = True
    if getattr(parsed, "wind_speed", None) is not None:
        overrides["load_wind_speed_ms"] = float(parsed.wind_speed)
    if getattr(parsed, "wind_terrain", None) is not None:
        overrides["load_wind_terrain"] = parsed.wind_terrain
    if getattr(parsed, "wind_direction", None) is not None:
        overrides["load_wind_direction_deg"] = float(parsed.wind_direction)
    if getattr(parsed, "snow_zone", None) is not None:
        overrides["load_snow_zone"] = parsed.snow_zone
    if getattr(parsed, "snow_exposure", None) is not None:
        overrides["load_snow_exposure"] = float(parsed.snow_exposure)

    if getattr(parsed, "costing", False):
        overrides["generate_costing"] = True

    if getattr(parsed, "weather", False):
        overrides["generate_weather"] = True
    if getattr(parsed, "gasket_type", None) is not None:
        overrides["gasket_type"] = parsed.gasket_type

    if getattr(parsed, "production", False):
        overrides["generate_production"] = True

    return overrides, parsed


def load_parameters(
    config_path: Path | str | None,
    cli_overrides: Mapping[str, Any] | None = None,
    spreadsheet_values: Mapping[str, Any] | None = None,
) -> DomeParameters:
    """Load parameters using the JSON → CLI → spreadsheet precedence chain."""

    data = load_json_config(config_path)
    params = DomeParameters.from_dict(data)
    if cli_overrides:
        params = apply_overrides(params, cli_overrides)
    if spreadsheet_values:
        params = apply_overrides(params, spreadsheet_values)
    return params


# Re-export for backward compatibility — the dialog now lives in gui_dialog.
from .gui_dialog import prompt_parameters_dialog as prompt_parameters_dialog  # noqa: F811



def write_varset(obj: Any, params: DomeParameters) -> None:
    """Populate an App::VarSet or Spreadsheet object with parameter data.

    ``obj`` can be:
    - ``App.VarSet`` instance (preferred in FreeCAD 1.0.2)
    - ``App.DocumentObject`` hosting a ``VarSet`` property
    - ``App.Spreadsheet`` object for manual editing

    The function degrades gracefully when FreeCAD modules are unavailable, so
    tests can run in plain Python.
    """

    try:
        import FreeCAD  # type: ignore
    except ImportError:  # pragma: no cover - plain Python environment
        return

    data = params.to_dict()

    if hasattr(obj, "TypeId") and obj.TypeId == "App::VarSet":
        for key, value in data.items():
            obj.setExpression(key, None)
            obj.setProperty("App::PropertyFloat", key, value)
        return

    if getattr(obj, "TypeId", "") == "Spreadsheet::Sheet":
        for row, (key, value) in enumerate(sorted(data.items()), start=1):
            obj.set("A%d" % row, key)
            obj.set("B%d" % row, str(value))
        return

    if hasattr(obj, "VarSet"):
        write_varset(obj.VarSet, params)
        return

    raise TypeError("Unsupported object for VarSet writing")
