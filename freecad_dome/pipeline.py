"""Pipeline architecture for the geodesic dome generator.

Breaks the monolithic generation flow into composable, testable steps.
Each step receives a shared ``PipelineContext`` and can read/write its fields.
Steps declare their own ``should_run`` predicate so the pipeline runner
automatically skips irrelevant stages.

Usage::

    from freecad_dome.pipeline import DomePipeline, PipelineContext

    ctx = PipelineContext(params=my_params, out_dir=Path("exports"))
    pipeline = DomePipeline()          # default steps
    pipeline.run(ctx)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import icosahedron, tessellation
from .parameters import DomeParameters
from .tessellation import TessellatedDome

__all__ = [
    "PipelineContext",
    "PipelineStep",
    "DomePipeline",
    "TessellationStep",
    "AutoDoorAngleStep",
    "EntryPorchStep",
    "BaseWallStep",
    "RiserWallStep",
    "StrutGenerationStep",
    "PanelGenerationStep",
    "GlassPanelStep",
    "DoorOpeningStep",
    "PanelAccuracyReportStep",
    "SpreadsheetStep",
    "ManifestExportStep",
    "ModelExportStep",
    "VentilationStep",
    "SkylightStep",
    "NodeConnectorStep",
    "CoveringReportStep",
    "FoundationStep",
    "LoadCalculationStep",
    "StructuralCheckStep",
    "ProductionDrawingsStep",
    "CncExportStep",
    "TechDrawStep",
    "AssemblyGuideStep",
    "WeatherProtectionStep",
    "CostEstimationStep",
    "default_steps",
]


# ---------------------------------------------------------------------------
# Pipeline context — shared state between steps
# ---------------------------------------------------------------------------


@dataclass
class PipelineContext:
    """Mutable state bag passed through every pipeline step."""

    params: DomeParameters
    out_dir: Path = field(default_factory=lambda: Path("exports"))

    # Export control flags (typically populated from CLI).
    skip_ifc: bool = False
    skip_stl: bool = False
    skip_dxf: bool = False
    manifest_name: str = "dome_manifest.json"
    ifc_name: str = "dome.ifc"
    stl_name: str = "dome.stl"
    dxf_name: str = "dome.dxf"

    # Populated by TessellationStep.
    mesh: Any = None
    dome: TessellatedDome | None = None

    # Populated by builder steps.
    document: Any = None  # Shared FreeCAD document.
    strut_builder: Any = None
    strut_instances: List[Any] = field(default_factory=list)
    panel_builder: Any = None
    panel_instances: List[Any] = field(default_factory=list)

    # Validation report from tessellation.
    validation: Dict[str, Any] = field(default_factory=dict)

    def _set_document_if_missing(self, doc: Any) -> None:
        """Track the first FreeCAD document created by any builder."""
        if self.document is None and doc is not None:
            self.document = doc


# ---------------------------------------------------------------------------
# Step base class
# ---------------------------------------------------------------------------


class PipelineStep(ABC):
    """A single composable stage of the dome generation pipeline."""

    name: str = "unnamed"

    def should_run(self, ctx: PipelineContext) -> bool:
        """Return ``False`` to skip this step for the current context."""
        return True

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> None:
        """Perform the step's work, mutating *ctx* as needed."""
        ...


# ---------------------------------------------------------------------------
# Concrete steps
# ---------------------------------------------------------------------------


class TessellationStep(PipelineStep):
    """Build icosahedron mesh and tessellate into nodes/struts/panels."""

    name = "tessellation"

    def execute(self, ctx: PipelineContext) -> None:
        ctx.mesh = icosahedron.build_icosahedron(ctx.params.radius_m)
        if ctx.params.use_truncation and ctx.params.truncation_ratio > 0:
            logging.info(
                "Truncation enabled (ratio=%.3f); handled during tessellation",
                ctx.params.truncation_ratio,
            )
        else:
            logging.info("Truncation disabled; using full icosahedron")
        ctx.dome = tessellation.tessellate(ctx.mesh, ctx.params)
        logging.info("Tessellation summary: %s", ctx.dome.summary())
        ctx.validation = tessellation.validate_structure(ctx.dome, ctx.params)
        _log_validation_report(ctx.validation)


class AutoDoorAngleStep(PipelineStep):
    """Choose optimal door angle based on dome geometry."""

    name = "auto_door_angle"

    def should_run(self, ctx: PipelineContext) -> bool:
        return (
            (ctx.params.generate_base_wall or ctx.params.generate_entry_porch)
            and ctx.params.auto_door_angle
        )

    def execute(self, ctx: PipelineContext) -> None:
        from . import base_wall

        try:
            suggested = base_wall.suggest_door_angle_deg(ctx.params, ctx.dome)
            if suggested is not None:
                ctx.params.door_angle_deg = float(suggested)
                logging.info("Auto door angle: %.2f deg", float(suggested))
        except Exception as exc:
            logging.warning("Auto door angle selection failed: %s", exc)


class EntryPorchStep(PipelineStep):
    """Generate framed + glazed vestibule at the belt opening."""

    name = "entry_porch"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.params.generate_entry_porch

    def execute(self, ctx: PipelineContext) -> None:
        from . import entry_porch

        try:
            builder = entry_porch.EntryPorchBuilder(ctx.params)
            result = builder.create_entry_porch(ctx.dome)
            ctx._set_document_if_missing(builder.document)
            if result is not None:
                logging.info(
                    "Generated entry porch: angle=%.2f deg belt_z=%.3fm",
                    float(result.angle_deg),
                    float(result.belt_height_m),
                )
        except Exception as exc:
            logging.warning("Entry porch generation failed: %s", exc)


class BaseWallStep(PipelineStep):
    """Generate cylindrical knee wall with door cutout."""

    name = "base_wall"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.params.generate_base_wall

    def execute(self, ctx: PipelineContext) -> None:
        from . import base_wall

        try:
            builder = base_wall.BaseWallBuilder(ctx.params)
            result = builder.create_base_wall()
            ctx._set_document_if_missing(builder.document)
            if result is not None:
                logging.info(
                    "Generated base wall: radius=%.3fm belt_z=%.3fm (wall_z=[%.3f..%.3f])",
                    result.wall_radius_m,
                    result.belt_height_m,
                    result.wall_bottom_z_m,
                    result.wall_top_z_m,
                )
        except Exception as exc:
            logging.warning("Base wall generation failed: %s", exc)


class RiserWallStep(PipelineStep):
    """Plan riser wall (pikendusring) geometry and write report."""

    name = "riser_wall"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.params.generate_riser_wall and ctx.dome is not None

    def execute(self, ctx: PipelineContext) -> None:
        from .riser_wall import plan_riser_wall, write_riser_report

        try:
            plan = plan_riser_wall(ctx.dome, ctx.params)
            ctx.riser_wall_plan = plan  # type: ignore[attr-defined]
            logging.info("Riser wall plan: %s", plan.summary())
            write_riser_report(plan, ctx.out_dir)
        except Exception as exc:
            logging.warning("Riser wall planning failed: %s", exc)


class StrutGenerationStep(PipelineStep):
    """Generate strut solids with bevels and node-fit cuts."""

    name = "strut_generation"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.params.generate_struts

    def execute(self, ctx: PipelineContext) -> None:
        from .node_fit import compute_node_fit_data
        from .struts import StrutBuilder

        # Compute shared node-fit data once for both struts and connectors.
        if not hasattr(ctx, "node_fit_data") or ctx.node_fit_data is None:  # type: ignore[attr-defined]
            ctx.node_fit_data = compute_node_fit_data(  # type: ignore[attr-defined]
                ctx.dome,
                hemisphere_ratio=ctx.params.hemisphere_ratio,
                radius_m=ctx.params.radius_m,
            )

        ctx.strut_builder = StrutBuilder(ctx.params)
        ctx.strut_instances = ctx.strut_builder.create_struts(ctx.dome)
        ctx._set_document_if_missing(ctx.strut_builder.document)
        logging.info("Generated %d struts", len(ctx.strut_instances))


class NodeConnectorStep(PipelineStep):
    """Generate node connector (hub) geometry and metadata."""

    name = "node_connectors"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.params.generate_node_connectors and ctx.dome is not None

    def execute(self, ctx: PipelineContext) -> None:
        from .node_connectors import NodeConnectorBuilder, write_connector_report
        from .node_fit import compute_node_fit_data

        # Ensure shared node-fit data is available (may already exist from strut step).
        if not hasattr(ctx, "node_fit_data") or ctx.node_fit_data is None:  # type: ignore[attr-defined]
            ctx.node_fit_data = compute_node_fit_data(  # type: ignore[attr-defined]
                ctx.dome,
                hemisphere_ratio=ctx.params.hemisphere_ratio,
                radius_m=ctx.params.radius_m,
            )

        try:
            builder = NodeConnectorBuilder(ctx.params)
            connectors = builder.create_connectors(
                ctx.dome,
                node_fit_data=getattr(ctx, "node_fit_data", None),
            )
            ctx.node_connectors = connectors  # type: ignore[attr-defined]

            # Create FreeCAD 3-D solids (plates with bolt holes).
            solids = builder.create_connector_solids(connectors)
            ctx._set_document_if_missing(builder.document)

            # Write connector report JSON.
            report_path = ctx.out_dir / "node_connector_report.json"
            write_connector_report(connectors, report_path)

            logging.info(
                "Generated %d node connectors (%d solids)",
                len(connectors),
                len(solids),
            )
        except Exception as exc:
            logging.warning("Node connector generation failed: %s", exc)


class StrutBoltHoleStep(PipelineStep):
    """Drill bolt holes in strut ends to match connector bolt positions (E8)."""

    name = "strut_bolt_holes"

    def should_run(self, ctx: PipelineContext) -> bool:
        return (
            ctx.params.generate_node_connectors
            and ctx.params.generate_struts
            and hasattr(ctx, "node_connectors")
            and hasattr(ctx, "strut_builder")
            and ctx.strut_builder is not None
        )

    def execute(self, ctx: PipelineContext) -> None:
        connectors = getattr(ctx, "node_connectors", None)
        if not connectors:
            return
        modified = ctx.strut_builder.drill_connector_bolt_holes(connectors)
        logging.info("Drilled bolt holes in %d struts", modified)


class PanelGenerationStep(PipelineStep):
    """Generate panel faces and/or panel frames."""

    name = "panel_generation"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.params.generate_panel_faces or ctx.params.generate_panel_frames

    def execute(self, ctx: PipelineContext) -> None:
        from .panels import PanelBuilder

        ctx.panel_builder = PanelBuilder(ctx.params, document=ctx.document)
        ctx.panel_instances = ctx.panel_builder.create_panels(ctx.dome)
        ctx._set_document_if_missing(ctx.panel_builder.document)
        logging.info("Generated %d panels", len(ctx.panel_instances))


class GlassPanelStep(PipelineStep):
    """Generate glass panel solids."""

    name = "glass_panels"

    def should_run(self, ctx: PipelineContext) -> bool:
        return float(ctx.params.glass_thickness_m) > 0

    def execute(self, ctx: PipelineContext) -> None:
        from .panels import PanelBuilder

        if ctx.panel_builder is None:
            ctx.panel_builder = PanelBuilder(ctx.params, document=ctx.document)
        created = ctx.panel_builder.create_glass_panels(ctx.dome)
        ctx._set_document_if_missing(ctx.panel_builder.document)
        if created:
            logging.info("Generated %d glass panels", created)


class DoorOpeningStep(PipelineStep):
    """Cut a door opening through the dome shell."""

    name = "door_opening"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.document is not None and ctx.params.generate_entry_porch

    def execute(self, ctx: PipelineContext) -> None:
        from . import door_opening

        try:
            res = door_opening.apply_door_opening(
                ctx.document, params=ctx.params, dome=ctx.dome
            )
            if res is not None:
                logging.info("Cut door opening (angle=%.2f deg)", float(res.angle_deg))
        except Exception as exc:
            logging.warning("Door opening cut failed: %s", exc)


class VentilationStep(PipelineStep):
    """Plan ventilation layout and log the result."""

    name = "ventilation"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.params.generate_ventilation and ctx.dome is not None

    def execute(self, ctx: PipelineContext) -> None:
        from .ventilation import plan_ventilation

        try:
            plan = plan_ventilation(ctx.dome, ctx.params)
            ctx.ventilation_plan = plan  # type: ignore[attr-defined]
            logging.info("Ventilation plan: %s", plan.summary())
            if not plan.meets_target:
                logging.warning(
                    "Ventilation ratio %.1f%% is outside target range %.0f–%.0f%%",
                    plan.vent_ratio * 100,
                    plan.target_ratio_min * 100,
                    plan.target_ratio_max * 100,
                )

            # Write ventilation plan JSON.
            import json
            report_path = ctx.out_dir / "ventilation_plan.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(plan.to_dict(), indent=2), encoding="utf-8"
            )
            logging.info("Wrote ventilation plan %s", report_path)
        except Exception as exc:
            logging.warning("Ventilation planning failed: %s", exc)


class SkylightStep(PipelineStep):
    """Plan skylight / window layout and write report."""

    name = "skylights"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.params.generate_skylights and ctx.dome is not None

    def execute(self, ctx: PipelineContext) -> None:
        from .skylight import plan_skylights, write_skylight_report

        try:
            plan = plan_skylights(ctx.dome, ctx.params)
            ctx.skylight_plan = plan  # type: ignore[attr-defined]
            logging.info("Skylight plan: %s", plan.summary())
            write_skylight_report(plan, ctx.out_dir)
        except Exception as exc:
            logging.warning("Skylight planning failed: %s", exc)


class CoveringReportStep(PipelineStep):
    """Generate covering material report and BOM."""

    name = "covering_report"

    def should_run(self, ctx: PipelineContext) -> bool:
        # Run when covering thickness is set or a non-glass type is selected.
        has_covering = (
            float(ctx.params.covering_thickness_m) > 0
            or float(ctx.params.glass_thickness_m) > 0
            or ctx.params.covering_type != "glass"
        )
        return has_covering

    def execute(self, ctx: PipelineContext) -> None:
        from .covering import covering_for_params, write_covering_report
        from .ventilation import floor_area_m2

        try:
            spec = covering_for_params(ctx.params)
            # Collect panel areas.
            panel_areas: list[float] = []
            for panel in ctx.dome.panels:
                pts = [ctx.dome.nodes[i] for i in panel.node_indices]
                if len(pts) >= 3:
                    area = _polygon_area_3d(pts)
                    panel_areas.append(area)

            report_path = ctx.out_dir / "covering_report.json"
            write_covering_report(panel_areas, ctx.params, report_path)
            total_weight = sum(panel_areas) * spec.weight_kg_m2
            logging.info(
                "Covering: %s, %d panels, %.1f m², %.1f kg",
                spec.label,
                len(panel_areas),
                sum(panel_areas),
                total_weight,
            )
        except Exception as exc:
            logging.warning("Covering report failed: %s", exc)


class FoundationStep(PipelineStep):
    """Generate foundation layout with anchor bolt positions."""

    name = "foundation"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(getattr(ctx.params, "generate_foundation", False)) and ctx.dome is not None

    def execute(self, ctx: PipelineContext) -> None:
        from .foundation import foundation_for_params, write_foundation_report, create_foundation_solids

        try:
            plan = foundation_for_params(ctx.dome, ctx.params)
            ctx.foundation_plan = plan  # type: ignore[attr-defined]
            report_path = ctx.out_dir / "foundation_report.json"
            write_foundation_report(plan, ctx.params, report_path)

            # Create FreeCAD 3-D solids (strip / piers / anchors).
            solids = create_foundation_solids(plan, ctx.params, document=ctx.document)
            if solids:
                ctx._set_document_if_missing(solids[0].Document)
        except Exception as exc:
            logging.warning("Foundation layout failed: %s", exc)


class LoadCalculationStep(PipelineStep):
    """Compute structural loads (dead/snow/wind) and export results."""

    name = "load_calculation"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(getattr(ctx.params, "generate_loads", False)) and ctx.dome is not None

    def execute(self, ctx: PipelineContext) -> None:
        from .loads import compute_loads, write_load_report, write_load_csv

        try:
            result = compute_loads(ctx.dome, ctx.params)
            ctx.load_result = result  # type: ignore[attr-defined]
            report_path = ctx.out_dir / "load_report.json"
            write_load_report(result, ctx.params, report_path)
            csv_path = ctx.out_dir / "load_nodes.csv"
            write_load_csv(result, csv_path)
        except Exception as exc:
            logging.warning("Load calculation failed: %s", exc)


class StructuralCheckStep(PipelineStep):
    """Check strut capacity against computed loads (Eurocode)."""

    name = "structural_check"

    def should_run(self, ctx: PipelineContext) -> bool:
        return (
            bool(getattr(ctx.params, "generate_structural_check", False))
            and ctx.dome is not None
            and getattr(ctx, "load_result", None) is not None
        )

    def execute(self, ctx: PipelineContext) -> None:
        from .structural_check import run_structural_check, write_check_report

        try:
            result = run_structural_check(ctx.dome, ctx.params, ctx.load_result)
            ctx.structural_check_result = result  # type: ignore[attr-defined]
            report_path = ctx.out_dir / "structural_check.json"
            write_check_report(result, report_path)
        except Exception as exc:
            logging.warning("Structural check failed: %s", exc)


class ProductionDrawingsStep(PipelineStep):
    """Generate production data: cut-lists, saw table, node plates, assembly plan."""

    name = "production_drawings"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(getattr(ctx.params, "generate_production", False)) and ctx.dome is not None

    def execute(self, ctx: PipelineContext) -> None:
        from .production import (
            production_for_params,
            write_production_report,
            write_saw_table_csv,
            write_node_plate_dxf,
            write_node_plate_svg,
        )

        try:
            pack = production_for_params(ctx.dome, ctx.params)
            ctx.production_pack = pack  # type: ignore[attr-defined]

            # JSON report
            report_path = ctx.out_dir / "production_report.json"
            write_production_report(pack, ctx.params, report_path)

            # Saw table CSV
            csv_path = ctx.out_dir / "saw_table.csv"
            write_saw_table_csv(pack.saw_table, csv_path)

            # Node plate DXF/SVG (one per unique valence to avoid large output)
            plates_dir = ctx.out_dir / "node_plates"
            seen_valences: set = set()
            for plate in pack.node_plates:
                if plate.valence in seen_valences:
                    continue
                seen_valences.add(plate.valence)
                write_node_plate_dxf(plate, plates_dir / f"plate_v{plate.valence}_n{plate.node_index}.dxf")
                write_node_plate_svg(plate, plates_dir / f"plate_v{plate.valence}_n{plate.node_index}.svg")

            logging.info(
                "Production: %d struts (%d types), %d plates, %d stages",
                pack.total_struts,
                pack.unique_types,
                len(pack.node_plates),
                len(pack.assembly_stages),
            )
        except Exception as exc:
            logging.warning("Production drawings failed: %s", exc)


class CncExportStep(PipelineStep):
    """Export per-strut STEP files and cutting table for CNC manufacturing."""

    name = "cnc_export"

    def should_run(self, ctx: PipelineContext) -> bool:
        return (
            bool(getattr(ctx.params, "generate_cnc_export", False))
            and ctx.dome is not None
        )

    def execute(self, ctx: PipelineContext) -> None:
        from .cnc_export import cnc_export_for_dome

        try:
            cnc_dir = ctx.out_dir / "cnc_export"
            result = cnc_export_for_dome(
                ctx.dome,
                ctx.params,
                cnc_dir,
                doc=ctx.document,
            )
            ctx.cnc_export_result = result  # type: ignore[attr-defined]
        except Exception as exc:
            logging.warning("CNC export failed: %s", exc)


class TechDrawStep(PipelineStep):
    """Generate TechDraw pages (overview, parts, node details) with PDF output."""

    name = "techdraw"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(getattr(ctx.params, "generate_techdraw", False))

    def execute(self, ctx: PipelineContext) -> None:
        from .techdraw import generate_techdraw_for_dome

        try:
            result = generate_techdraw_for_dome(
                ctx.params,
                ctx.out_dir,
                doc=ctx.document,
            )
            ctx.techdraw_result = result  # type: ignore[attr-defined]
            logging.info(
                "TechDraw: %d pages created, %d PDF exported, format %s, mode %s",
                result.pages_created,
                result.pdf_exported,
                result.page_format,
                result.views_mode,
            )
        except Exception as exc:
            logging.warning("TechDraw generation failed: %s", exc)


class AssemblyGuideStep(PipelineStep):
    """Generate ring-by-ring assembly instructions with BOM and SVG diagrams."""

    name = "assembly_guide"

    def should_run(self, ctx: PipelineContext) -> bool:
        return (
            bool(getattr(ctx.params, "generate_assembly_guide", False))
            and ctx.dome is not None
        )

    def execute(self, ctx: PipelineContext) -> None:
        from .assembly import assembly_guide_for_dome, write_assembly_report, write_assembly_svg

        try:
            guide = assembly_guide_for_dome(ctx.dome, ctx.params)
            ctx.assembly_guide = guide  # type: ignore[attr-defined]

            assembly_dir = ctx.out_dir / "assembly"
            report_path = assembly_dir / "assembly_guide.json"
            write_assembly_report(guide, ctx.params, report_path)
            guide.report_path = str(report_path)

            svg_paths = write_assembly_svg(guide, ctx.dome, assembly_dir)
            guide.svg_paths = svg_paths

            logging.info(
                "Assembly guide: %d stages, %.1f h estimated (%d workers), %d SVGs",
                guide.total_stages,
                guide.total_estimated_hours,
                guide.workers,
                len(svg_paths),
            )
        except Exception as exc:
            logging.warning("Assembly guide generation failed: %s", exc)


class MultiDomeStep(PipelineStep):
    """Plan multi-dome layout with corridors, merged foundation and BOM."""

    name = "multi_dome"

    def should_run(self, ctx: PipelineContext) -> bool:
        return (
            bool(getattr(ctx.params, "multi_dome_enabled", False))
            and ctx.dome is not None
        )

    def execute(self, ctx: PipelineContext) -> None:
        from .multi_dome import plan_multi_dome, write_multi_dome_report

        try:
            plan = plan_multi_dome(ctx.dome, ctx.params)
            ctx.multi_dome_plan = plan  # type: ignore[attr-defined]
            logging.info("Multi-dome plan: %s", plan.summary())
            write_multi_dome_report(plan, ctx.out_dir)
        except Exception as exc:
            logging.warning("Multi-dome planning failed: %s", exc)


class WeatherProtectionStep(PipelineStep):
    """Generate weather protection data: gaskets, drainage, eave details."""

    name = "weather_protection"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(getattr(ctx.params, "generate_weather", False)) and ctx.dome is not None

    def execute(self, ctx: PipelineContext) -> None:
        from .weather import weather_for_params, write_weather_report

        try:
            pack = weather_for_params(ctx.dome, ctx.params)
            ctx.weather_pack = pack  # type: ignore[attr-defined]

            report_path = ctx.out_dir / "weather_report.json"
            write_weather_report(pack, ctx.params, report_path)

            logging.info(
                "Weather: %s, %.1f m gasket, %d drain holes, %d eave nodes",
                pack.gasket_profile.name,
                pack.total_gasket_length_m,
                pack.total_drain_holes,
                len(pack.eave_details),
            )
        except Exception as exc:
            logging.warning("Weather protection failed: %s", exc)


class CostEstimationStep(PipelineStep):
    """Generate cost estimate and full BOM."""

    name = "cost_estimation"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(getattr(ctx.params, "generate_costing", False)) and ctx.dome is not None

    def execute(self, ctx: PipelineContext) -> None:
        from .costing import cost_estimate_for_params, write_cost_report, write_cost_csv

        try:
            estimate = cost_estimate_for_params(ctx.dome, ctx.params)
            ctx.cost_estimate = estimate  # type: ignore[attr-defined]

            report_path = ctx.out_dir / "cost_report.json"
            write_cost_report(estimate, ctx.params, report_path)

            csv_path = ctx.out_dir / "cost_bom.csv"
            write_cost_csv(estimate, csv_path)

            logging.info(
                "Cost (%s): %.2f total (%.2f material, %.2f hardware, "
                "%.2f labour, %.2f overhead), %d BOM items",
                estimate.currency,
                estimate.total_eur,
                estimate.total_material_eur,
                estimate.total_hardware_eur,
                estimate.total_labour_eur,
                estimate.total_overhead_eur,
                len(estimate.bom),
            )
        except Exception as exc:
            logging.warning("Cost estimation failed: %s", exc)


class SpreadsheetStep(PipelineStep):
    """Generate FreeCAD spreadsheets with parameters and BOM."""

    name = "spreadsheets"

    def should_run(self, ctx: PipelineContext) -> bool:
        return ctx.document is not None and ctx.params.generate_spreadsheets

    def execute(self, ctx: PipelineContext) -> None:
        from . import spreadsheets

        try:
            spreadsheets.export_dome_spreadsheets(
                ctx.document,
                params=ctx.params,
                dome=ctx.dome,
                struts=ctx.strut_instances,
            )
            logging.info("Updated FreeCAD spreadsheets (Dome_*)")
        except Exception as exc:
            logging.warning("Spreadsheet export failed: %s", exc)


class PanelAccuracyReportStep(PipelineStep):
    """Write a JSON report comparing geometry to mathematical panel planes."""

    name = "panel_accuracy_report"

    def should_run(self, ctx: PipelineContext) -> bool:
        if ctx.document is None:
            return False
        return bool(
            ctx.panel_instances
            or float(ctx.params.glass_thickness_m) > 0
            or ctx.params.generate_panel_faces
            or ctx.params.generate_panel_frames
        )

    def execute(self, ctx: PipelineContext) -> None:
        from .export import write_panel_accuracy_report

        try:
            report_path = ctx.out_dir / "panel_accuracy_report.json"
            write_panel_accuracy_report(
                ctx.document, ctx.dome, ctx.params, report_path
            )
            logging.info("Wrote panel accuracy report %s", report_path)
        except Exception as exc:
            logging.warning("Panel accuracy report failed: %s", exc)


class ManifestExportStep(PipelineStep):
    """Write per-strut JSON manifest."""

    name = "manifest_export"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(ctx.strut_instances)

    def execute(self, ctx: PipelineContext) -> None:
        from .export import export_manifest

        manifest_path = ctx.out_dir / ctx.manifest_name
        export_manifest(ctx.strut_instances, manifest_path)


class ModelExportStep(PipelineStep):
    """Export IFC / STL / DXF files."""

    name = "model_export"

    def should_run(self, ctx: PipelineContext) -> bool:
        if not ctx.params.generate_struts:
            if ctx.params.panels_only:
                logging.info("Panels-only mode complete; skipping IFC/STL/DXF exports")
            else:
                logging.info("Strut generation disabled; skipping IFC/STL/DXF exports")
            return False
        if ctx.document is None:
            logging.info("No FreeCAD document detected; skipping IFC/STL/DXF exports")
            return False
        return True

    def execute(self, ctx: PipelineContext) -> None:
        from .export import collect_structural_objects, export_dxf, export_ifc, export_stl

        structural_objects = collect_structural_objects(ctx.document)
        if not structural_objects:
            logging.info("No structural objects detected; skipping IFC/STL/DXF exports")
            return

        if not ctx.skip_ifc:
            export_ifc(structural_objects, ctx.out_dir / ctx.ifc_name)
        if not ctx.skip_stl:
            export_stl(structural_objects, ctx.out_dir / ctx.stl_name)
        if not ctx.skip_dxf:
            export_dxf(structural_objects, ctx.out_dir / ctx.dxf_name)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def default_steps() -> List[PipelineStep]:
    """Return the standard ordered list of pipeline steps."""
    return [
        TessellationStep(),
        AutoDoorAngleStep(),
        EntryPorchStep(),
        BaseWallStep(),
        RiserWallStep(),
        StrutGenerationStep(),
        NodeConnectorStep(),
        StrutBoltHoleStep(),
        PanelGenerationStep(),
        GlassPanelStep(),
        DoorOpeningStep(),
        VentilationStep(),
        SkylightStep(),
        CoveringReportStep(),
        FoundationStep(),
        LoadCalculationStep(),
        StructuralCheckStep(),
        ProductionDrawingsStep(),
        CncExportStep(),
        TechDrawStep(),
        AssemblyGuideStep(),
        MultiDomeStep(),
        WeatherProtectionStep(),
        CostEstimationStep(),
        SpreadsheetStep(),
        PanelAccuracyReportStep(),
        ManifestExportStep(),
        ModelExportStep(),
    ]


class DomePipeline:
    """Orchestrates the full dome generation flow.

    The default step order matches the previous monolithic ``main()`` logic.
    Users can supply a custom step list to re-order, insert, or remove stages.
    """

    def __init__(self, steps: List[PipelineStep] | None = None) -> None:
        self.steps = steps if steps is not None else default_steps()

    def run(self, ctx: PipelineContext) -> None:
        """Execute all enabled steps in order."""
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        for step in self.steps:
            if step.should_run(ctx):
                logging.info("[pipeline] %s", step.name)
                step.execute(ctx)

    def insert_before(self, reference_name: str, step: PipelineStep) -> None:
        """Insert *step* immediately before the step named *reference_name*."""
        for i, existing in enumerate(self.steps):
            if existing.name == reference_name:
                self.steps.insert(i, step)
                return
        self.steps.append(step)

    def insert_after(self, reference_name: str, step: PipelineStep) -> None:
        """Insert *step* immediately after the step named *reference_name*."""
        for i, existing in enumerate(self.steps):
            if existing.name == reference_name:
                self.steps.insert(i + 1, step)
                return
        self.steps.append(step)

    def remove(self, step_name: str) -> None:
        """Remove the step with the given name, if present."""
        self.steps = [s for s in self.steps if s.name != step_name]

    def replace(self, step_name: str, new_step: PipelineStep) -> None:
        """Replace an existing step with *new_step*."""
        for i, existing in enumerate(self.steps):
            if existing.name == step_name:
                self.steps[i] = new_step
                return
        self.steps.append(new_step)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_validation_report(report: Dict[str, Any]) -> None:
    max_error = report.get("max_radius_error", 0)
    logging.info("Max node radius deviation: %.6f", max_error)
    stray = report.get("stray_endpoints", [])
    if stray:
        sample = stray[:5]
        logging.error("%d endpoints drifted from nodes (sample: %s)", len(stray), sample)
    low_nodes = report.get("low_valence_nodes", [])
    if low_nodes:
        logging.warning("%d nodes have low valence", len(low_nodes))
    lengths = report.get("length_histogram", {})
    if lengths:
        logging.info("Strut length distribution: %s", lengths)
    missing_panels = report.get("missing_panel_edges", [])
    if missing_panels:
        logging.error(
            "Panel mismatches detected (%d edges without struts)", len(missing_panels)
        )
    panel_stats = report.get("panel_area_stats")
    if panel_stats:
        logging.info("Panel area stats: %s", panel_stats)


def _polygon_area_3d(points: list) -> float:
    """Compute area of a 3D polygon using Newell's method."""
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
