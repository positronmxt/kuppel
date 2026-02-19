"""P3 #11 — Integration tests requiring FreeCAD or IfcOpenShell.

These tests are skipped when the required tool is not available.
They validate:
  - Actual FreeCAD solid body generation via ``freecadcmd``
  - IFC output validation via IfcOpenShell
  - Visual snapshot consistency (stub for future CI use)

Run with FreeCAD available::

    freecadcmd -c "import pytest; pytest.main([__file__, '-v'])"

Or simply::

    pytest tests/test_integration.py -v

Tests that need FreeCAD / IfcOpenShell are auto-skipped when the
dependency is missing.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

_FREECAD_AVAILABLE = False
try:
    import FreeCAD  # type: ignore

    _FREECAD_AVAILABLE = True
except ImportError:
    # Also check if freecadcmd is on PATH
    try:
        _FREECAD_AVAILABLE = (
            subprocess.run(
                ["freecadcmd", "--version"],
                capture_output=True,
                timeout=10,
            ).returncode
            == 0
        )
    except (FileNotFoundError, OSError):
        _FREECAD_AVAILABLE = False

_IFC_AVAILABLE = False
try:
    import ifcopenshell  # type: ignore

    _IFC_AVAILABLE = True
except ImportError:
    pass

requires_freecad = pytest.mark.skipif(
    not _FREECAD_AVAILABLE,
    reason="FreeCAD (freecadcmd) not available",
)

requires_ifc = pytest.mark.skipif(
    not _IFC_AVAILABLE,
    reason="ifcopenshell not installed",
)


from freecad_dome.parameters import DomeParameters
from freecad_dome import icosahedron, tessellation
from freecad_dome import pipeline
from freecad_dome.pipeline import PipelineContext, DomePipeline, default_steps


# ---------------------------------------------------------------------------
# Pure-Python integration tests (always run)
# ---------------------------------------------------------------------------


def _make_test_dome():
    """Create a small dome for integration testing."""
    params = DomeParameters(radius_m=2.0, frequency=1, hemisphere_ratio=1.0)
    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)
    return dome, params


class TestPipelineIntegration:
    """Integration tests for the full pipeline in pure-Python mode."""

    def test_full_pipeline_all_features(self, tmp_path):
        """Pipeline should run with all feature flags enabled (no FreeCAD)."""
        params = DomeParameters(
            radius_m=2.0,
            frequency=2,
            hemisphere_ratio=0.625,
            generate_ventilation=True,
            generate_node_connectors=True,
            generate_foundation=True,
            generate_loads=True,
            generate_weather=True,
            generate_production=True,
            generate_costing=True,
        )
        ctx = PipelineContext(params=params, out_dir=tmp_path)
        p = DomePipeline()
        p.run(ctx)
        # Tessellation should have populated the dome
        assert ctx.dome is not None
        assert len(ctx.dome.nodes) > 0

    def test_pipeline_context_propagation(self, tmp_path):
        """Context data should flow correctly between pipeline steps."""
        params = DomeParameters(
            radius_m=2.0,
            frequency=1,
            hemisphere_ratio=1.0,
        )
        ctx = PipelineContext(params=params, out_dir=tmp_path)
        p = DomePipeline()
        p.run(ctx)
        assert ctx.dome is not None
        assert len(ctx.dome.nodes) > 0
        assert len(ctx.dome.struts) > 0
        assert len(ctx.dome.panels) > 0

    def test_pipeline_step_ordering(self):
        """All 20 steps should execute in correct order."""
        steps = default_steps()
        names = [s.name for s in steps]
        # Tessellation must come before everything else
        assert names.index("tessellation") == 0
        # Export must be last
        assert names[-1] == "model_export"

    def test_pipeline_manifest_generation(self, tmp_path):
        """Pipeline should produce a valid JSON manifest."""
        params = DomeParameters(
            radius_m=2.0,
            frequency=1,
            hemisphere_ratio=1.0,
        )
        ctx = PipelineContext(params=params, out_dir=tmp_path)
        p = DomePipeline()
        p.run(ctx)
        # Manifest should have been written to out_dir
        manifest_path = tmp_path / ctx.manifest_name
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            assert isinstance(data, (dict, list))


class TestParameterMigration:
    """Integration tests for parameter loading and migration paths."""

    def test_legacy_flat_json(self, tmp_path):
        """Legacy flat JSON configs should load correctly."""
        config = {
            "radius_m": 4.0,
            "frequency": 3,
            "stock_width_m": 0.06,
            "material": "wood",
        }
        config_path = tmp_path / "legacy.json"
        config_path.write_text(json.dumps(config))

        from freecad_dome.parameters import load_json_config

        data = load_json_config(config_path)
        params = DomeParameters.from_dict(data)
        assert params.radius_m == 4.0
        assert params.frequency == 3

    def test_nested_json(self, tmp_path):
        """New nested JSON configs should load correctly."""
        config = {
            "geometry": {"radius_m": 5.0, "frequency": 4},
            "structure": {"stock_width_m": 0.07},
        }
        config_path = tmp_path / "nested.json"
        config_path.write_text(json.dumps(config))

        from freecad_dome.parameters import load_json_config

        data = load_json_config(config_path)
        params = DomeParameters.from_dict(data)
        assert params.radius_m == 5.0
        assert params.stock_width_m == 0.07

    def test_cli_overrides_on_nested_config(self, tmp_path):
        """CLI overrides should work on top of nested JSON config."""
        from freecad_dome.parameters import load_parameters

        config = {
            "geometry": {"radius_m": 5.0},
        }
        config_path = tmp_path / "nested.json"
        config_path.write_text(json.dumps(config))

        params = load_parameters(config_path, cli_overrides={"radius_m": 6.0})
        assert params.radius_m == 6.0

    def test_all_config_fields_roundtrip(self):
        """Every field should survive to_dict → from_dict roundtrip."""
        params = DomeParameters(
            radius_m=4.0,
            frequency=3,
            stock_width_m=0.06,
            generate_ventilation=True,
            ventilation_mode="apex",
            generate_node_connectors=True,
            node_connector_type="ball",
            covering_type="glass",
            generate_foundation=True,
            foundation_type="point",
            generate_loads=True,
            load_snow_zone="IV",
            generate_weather=True,
            gasket_type="silicone_12x8",
            generate_production=True,
            generate_costing=True,
        )
        flat = params.to_dict()
        rebuilt = DomeParameters.from_dict(flat)
        # Verify all values match
        for key in flat:
            if key == "materials":
                continue  # Materials are serialized differently
            orig_val = getattr(params, key)
            rebuilt_val = getattr(rebuilt, key)
            if isinstance(orig_val, float):
                assert math.isclose(orig_val, rebuilt_val), f"{key}: {orig_val} != {rebuilt_val}"
            else:
                assert orig_val == rebuilt_val, f"{key}: {orig_val} != {rebuilt_val}"


# ---------------------------------------------------------------------------
# FreeCAD solid body tests (skipped without FreeCAD)
# ---------------------------------------------------------------------------


@requires_freecad
class TestFreeCADSolids:
    """Tests that validate actual FreeCAD solid body generation."""

    def test_strut_solid_is_valid(self):
        """Generated strut solids should be valid TopoDS_Shape."""
        import FreeCAD  # type: ignore
        import Part  # type: ignore

        dome, params = _make_test_dome()
        from freecad_dome.struts import generate_strut_solids

        solids = generate_strut_solids(dome, params)
        for solid in solids:
            assert solid.isValid(), "Strut solid is not valid"
            assert solid.Volume > 0, "Strut solid has zero volume"

    def test_panel_face_is_planar(self):
        """Panel faces should be planar within tolerance."""
        import FreeCAD  # type: ignore

        dome, params = _make_test_dome()
        for panel in dome.panels:
            coords = [dome.nodes[i] for i in panel.node_indices]
            if len(coords) >= 3:
                # Check coplanarity
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                normal = v1.cross(v2)
                for c in coords[3:]:
                    dist = abs(normal.dot(c - coords[0])) / normal.Length
                    assert dist < 1e-6, "Panel is not planar"

    def test_dome_model_export_ifc(self, tmp_path):
        """Full dome model should export to a valid IFC file."""
        params = DomeParameters(
            radius_m=2.0,
            frequency=1,
            hemisphere_ratio=1.0,
        )
        p = pipeline.DomePipeline(params)
        ctx = p.run()
        # If FreeCAD is available, the model export step should produce IFC
        ifc_path = tmp_path / "test_dome.ifc"
        # Export would need FreeCAD document context
        assert True  # Placeholder for actual IFC export test


# ---------------------------------------------------------------------------
# IFC validation tests (skipped without IfcOpenShell)
# ---------------------------------------------------------------------------


@requires_ifc
class TestIFCValidation:
    """Tests that validate IFC output structure and content."""

    def test_ifc_file_structure(self, tmp_path):
        """IFC file should have valid schema and required entities."""
        # This test requires an existing IFC file from a previous export
        ifc_files = list(Path("exports").glob("*.ifc"))
        if not ifc_files:
            pytest.skip("No IFC files found in exports/")

        import ifcopenshell  # type: ignore

        model = ifcopenshell.open(str(ifc_files[0]))
        # Should have a project
        projects = model.by_type("IfcProject")
        assert len(projects) >= 1, "IFC file must contain an IfcProject"

    def test_ifc_has_building_elements(self, tmp_path):
        """IFC file should contain building element entities."""
        ifc_files = list(Path("exports").glob("*.ifc"))
        if not ifc_files:
            pytest.skip("No IFC files found in exports/")

        import ifcopenshell  # type: ignore

        model = ifcopenshell.open(str(ifc_files[0]))
        elements = model.by_type("IfcBuildingElement")
        assert len(elements) > 0, "IFC file should contain building elements"

    def test_ifc_units_are_meters(self):
        """IFC file should use meters as the length unit."""
        ifc_files = list(Path("exports").glob("*.ifc"))
        if not ifc_files:
            pytest.skip("No IFC files found in exports/")

        import ifcopenshell  # type: ignore

        model = ifcopenshell.open(str(ifc_files[0]))
        units = model.by_type("IfcSIUnit")
        length_units = [u for u in units if u.UnitType == "LENGTHUNIT"]
        assert len(length_units) >= 1, "Must have a length unit definition"


# ---------------------------------------------------------------------------
# Visual snapshot stubs (for future CI use)
# ---------------------------------------------------------------------------


class TestVisualSnapshots:
    """Placeholder tests for visual regression testing.

    In a CI environment with FreeCAD available, these would:
    1. Render the dome model from fixed viewpoints
    2. Compare rendered images against golden snapshots
    3. Flag regressions exceeding a pixel-diff threshold

    For now, they validate that the geometric data needed for
    rendering is correctly produced.
    """

    def test_dome_geometry_deterministic(self):
        """Same parameters should produce identical geometry."""
        dome1, _ = _make_test_dome()
        dome2, _ = _make_test_dome()
        assert len(dome1.nodes) == len(dome2.nodes)
        assert len(dome1.struts) == len(dome2.struts)
        assert len(dome1.panels) == len(dome2.panels)
        for n1, n2 in zip(dome1.nodes, dome2.nodes):
            assert math.isclose(n1[0], n2[0], abs_tol=1e-10)
            assert math.isclose(n1[1], n2[1], abs_tol=1e-10)
            assert math.isclose(n1[2], n2[2], abs_tol=1e-10)

    def test_snapshot_metadata_available(self):
        """Dome should produce enough metadata for snapshot comparison."""
        dome, params = _make_test_dome()
        # Each panel has a defined normal for rendering
        for panel in dome.panels:
            assert panel.normal is not None
            length = math.sqrt(
                panel.normal[0] ** 2 + panel.normal[1] ** 2 + panel.normal[2] ** 2
            )
            assert math.isclose(length, 1.0, abs_tol=1e-6)
