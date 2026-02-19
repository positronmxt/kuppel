import inspect
import json
import math
import pytest

from freecad_dome import icosahedron, tessellation, panels, parameters, struts, base_wall, spreadsheets
from freecad_dome import ventilation, pipeline
from freecad_dome import node_connectors
from freecad_dome import covering
from freecad_dome import foundation
from freecad_dome import loads
from freecad_dome import structural_check
from freecad_dome import cnc_export
from freecad_dome import production
from freecad_dome import weather
from freecad_dome import costing
from freecad_dome import skylight
from freecad_dome import riser_wall
from freecad_dome import multi_dome
from freecad_dome.parameters import DomeParameters, parse_cli_overrides
from freecad_dome.parameters import (
    GeometryConfig, StructureConfig, CoveringConfig, OpeningsConfig,
    FoundationConfig, ExportConfig, CostingConfig, MultiDomeConfig,
)
from freecad_dome.tessellation import Panel as PanelDef, TessellatedDome


def test_truncation_increases_faces():
    mesh = icosahedron.build_icosahedron(1.0)
    truncated, _ = icosahedron.truncate_mesh(mesh, 0.18)
    assert len(truncated.nodes) > len(mesh.nodes)
    assert len(truncated.faces) > len(mesh.faces)


def test_tessellation_frequency_growth():
    params_low = DomeParameters(radius_m=3.0, frequency=1, hemisphere_ratio=1.0)
    params_high = DomeParameters(radius_m=3.0, frequency=3, hemisphere_ratio=1.0)

    base_mesh = icosahedron.build_icosahedron(params_low.radius_m)
    dome_low = tessellation.tessellate(base_mesh, params_low)
    dome_high = tessellation.tessellate(base_mesh, params_high)

    assert len(dome_high.nodes) > len(dome_low.nodes)
    assert len(dome_high.struts) > len(dome_low.struts)
    assert len(dome_high.panels) > len(dome_low.panels)
    assert all(s.length > 0 for s in dome_high.struts)
    assert all(len(p.node_indices) >= 3 for p in dome_high.panels)


def test_hemisphere_clipping_reduces_nodes():
    params_full = DomeParameters(radius_m=3.0, frequency=2, hemisphere_ratio=1.0)
    params_half = DomeParameters(radius_m=3.0, frequency=2, hemisphere_ratio=0.5)
    mesh = icosahedron.build_icosahedron(params_full.radius_m)

    dome_full = tessellation.tessellate(mesh, params_full)
    dome_half = tessellation.tessellate(mesh, params_half)

    assert len(dome_half.nodes) < len(dome_full.nodes)
    assert len(dome_half.struts) <= len(dome_full.struts)
    belt_height = params_half.radius_m * (1 - 2 * params_half.hemisphere_ratio)
    assert any(
        math.isclose(node[2], belt_height, abs_tol=1e-6) for node in dome_half.nodes
    ), "Clipping should introduce nodes on the belt plane"


def test_strut_normals_point_inward():
    params = DomeParameters(radius_m=3.0, frequency=2, hemisphere_ratio=0.625)
    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)

    for strut in dome.struts:
        mx, my, mz = strut.midpoint
        nx, ny, nz = strut.primary_normal
        dot = mx * nx + my * ny + mz * nz
        assert dot <= 1e-6


def test_panel_builder_supports_polygons():
    params = DomeParameters(radius_m=2.0, generate_panel_frames=True)
    nodes = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    panel = PanelDef(index=0, node_indices=(0, 1, 2, 3), normal=(0.0, 0.0, 1.0))
    dome = TessellatedDome(nodes=nodes, struts=[], panels=[panel])

    builder = panels.PanelBuilder(params)
    created = builder.create_panels(dome)

    assert len(created) == 1
    assert math.isclose(created[0].area, 1.0)
    assert math.isclose(created[0].perimeter, 4.0)
    assert created[0].frame_name is None


def test_cli_panel_frame_overrides():
    overrides, _ = parameters.parse_cli_overrides(
        [
            "--panel-frames",
            "--panel-frame-inset",
            "0.02",
            "--panel-frame-profile",
            "0.03",
            "0.04",
        ]
    )
    assert overrides["generate_panel_frames"] is True
    assert math.isclose(overrides["panel_frame_inset_m"], 0.02)
    assert math.isclose(overrides["panel_frame_profile_width_m"], 0.03)
    assert math.isclose(overrides["panel_frame_profile_height_m"], 0.04)

    overrides, _ = parameters.parse_cli_overrides(["--no-panel-frames"])
    assert overrides["generate_panel_frames"] is False


def test_glass_seat_offset_uses_adjacent_struts():
    params = DomeParameters(radius_m=1.0, frequency=1)
    params.stock_width_m = 0.10

    # A single panel with a single adjacent strut. The strut axis is along +X.
    # The panel normal is +Y (treated as matching the strut primary normal).
    # The outer ridge falls at -Y by width/2 due to fallback perpendicular selection.
    nodes = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    panel = PanelDef(index=0, node_indices=(0, 1, 2), normal=(0.0, 1.0, 0.0))
    strut = tessellation.Strut(
        index=0,
        start_index=0,
        end_index=1,
        start=nodes[0],
        end=nodes[1],
        panel_indices=(0,),
        primary_normal=panel.normal,
        secondary_normal=None,
        length=1.0,
    )
    dome = TessellatedDome(nodes=nodes, struts=[strut], panels=[panel])

    builder = panels.PanelBuilder(params)
    plane = builder._panel_plane_data(dome, panel)
    assert plane is not None
    offset = builder._glass_seat_offset_m(dome, panel, plane)
    assert math.isclose(offset, -params.stock_width_m * 0.5, abs_tol=1e-9)


def test_split_struts_per_panel_splits_internal_non_belt_struts():
    # FreeCAD is not available in unit tests, but create_struts() should still
    # emit a manifest consistent with the intended split behavior.
    params = DomeParameters(radius_m=3.0, frequency=2, hemisphere_ratio=0.625, split_struts_per_panel=True)
    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)

    belt_height = params.radius_m * (1 - 2 * params.hemisphere_ratio)
    eps = max(1e-6, params.radius_m * 1e-5)

    def is_belt_strut(s: tessellation.Strut) -> bool:
        return (
            abs(float(s.start[2]) - belt_height) <= eps
            and abs(float(s.end[2]) - belt_height) <= eps
        )

    splittable = [s for s in dome.struts if len(s.panel_indices) == 2 and not is_belt_strut(s)]

    builder = struts.StrutBuilder(params)
    instances = builder.create_struts(dome)

    assert len(instances) == len(dome.struts) + len(splittable)
    assert any("_P" in inst.name for inst in instances)


def test_auto_door_angle_is_normalized_and_deterministic():
    params = DomeParameters(radius_m=2.5, frequency=2, truncation_ratio=0.25, hemisphere_ratio=0.65)
    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)

    a1 = base_wall.suggest_door_angle_deg(params, dome)
    a2 = base_wall.suggest_door_angle_deg(params, dome)

    assert a1 is not None
    assert a2 is not None
    assert 0.0 <= float(a1) < 360.0
    assert math.isclose(float(a1), float(a2), abs_tol=1e-12)


def test_entry_porch_bom_is_deterministic_and_sane():
    params = DomeParameters(
        generate_entry_porch=True,
        porch_depth_m=0.5,
        porch_width_m=1.2,
        porch_height_m=2.1,
        porch_member_size_m=0.045,
        porch_glass_thickness_m=0.006,
        door_width_m=0.9,
        door_height_m=2.1,
    )
    members1, glass1 = spreadsheets._compute_entry_porch_bom(params)
    members2, glass2 = spreadsheets._compute_entry_porch_bom(params)

    assert members1 == members2
    assert glass1 == glass2
    assert sum(int(r[2]) for r in members1) == 15
    assert any(r[1] == "Front post" for r in members1)
    assert any(r[1] == "Door stile" for r in members1)
    assert any(r[1] == "Front panel" for r in glass1)
    assert any(r[1] == "Door leaf" for r in glass1)


def test_auto_door_angle_points_to_top_pentagon_edge_midpoint_when_available():
    params = DomeParameters(radius_m=2.5, frequency=2, truncation_ratio=0.25, hemisphere_ratio=0.65)
    params.use_truncation = True
    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)

    R = float(params.radius_m)
    hemi = float(params.hemisphere_ratio)
    belt_height = R * (1.0 - 2.0 * hemi)
    eps = max(1e-6, R * 1e-5)

    pents = [p for p in dome.panels if len(getattr(p, "node_indices", ()) or ()) == 5]
    assert pents, "Expected at least one pentagon with truncation enabled"

    # Find the top (highest centroid-z) pentagon.
    top = None
    top_z = None
    for p in pents:
        zs = 0.0
        cnt = 0
        for ni in getattr(p, "node_indices", ()) or ():
            _x, _y, z = dome.nodes[int(ni)]
            zs += float(z)
            cnt += 1
        cz = zs / cnt
        if top_z is None or cz > top_z:
            top_z = cz
            top = p
    assert top is not None

    # Compute all apex pentagon edge-midpoint directions and pick the one closest
    # to the preferred azimuth (params.door_angle_deg).
    node_ids = list(getattr(top, "node_indices", ()) or ())
    preferred = float(getattr(params, "door_angle_deg", 0.0) or 0.0) % 360.0

    def ang_dist(a: float, b: float) -> float:
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    candidates = []
    for i in range(len(node_ids)):
        a = int(node_ids[i])
        b = int(node_ids[(i + 1) % len(node_ids)])
        ax, ay, _az = dome.nodes[a]
        bx, by, _bz = dome.nodes[b]
        mx = (float(ax) + float(bx)) * 0.5
        my = (float(ay) + float(by)) * 0.5
        if abs(mx) <= 1e-12 and abs(my) <= 1e-12:
            continue
        candidates.append(float(math.degrees(math.atan2(my, mx)) % 360.0))

    assert candidates
    candidates.sort(key=lambda a: (ang_dist(a, preferred), a))
    expected = float(candidates[0])

    got = base_wall.suggest_door_angle_deg(params, dome)
    assert got is not None
    assert math.isclose(float(got), float(expected), abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Edge case / regression tests
# ---------------------------------------------------------------------------


def test_frequency_one_produces_icosahedron_struts():
    """frequency=1 should produce the base icosahedral topology (after truncation)."""
    params = DomeParameters(radius_m=2.0, frequency=1, hemisphere_ratio=1.0)
    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)
    # Truncation converts 12 vertices into pentagons, producing more edges
    # than the base icosahedron (30).  Just verify we get something reasonable.
    assert len(dome.struts) > 0
    assert len(dome.panels) > 0
    assert all(s.length > 0 for s in dome.struts)
    assert all(len(p.node_indices) >= 3 for p in dome.panels)


def test_hemisphere_ratio_boundary_values():
    """hemisphere_ratio near extremes should still produce a valid dome."""
    params_high = DomeParameters(radius_m=2.0, frequency=2, hemisphere_ratio=0.99)
    params_low = DomeParameters(radius_m=2.0, frequency=2, hemisphere_ratio=0.1)
    mesh = icosahedron.build_icosahedron(params_high.radius_m)

    dome_high = tessellation.tessellate(mesh, params_high)
    dome_low = tessellation.tessellate(mesh, params_low)

    # Near-full sphere should have many more panels than a thin slice.
    assert len(dome_high.panels) > len(dome_low.panels)
    assert len(dome_low.panels) >= 1
    assert all(len(p.node_indices) >= 3 for p in dome_low.panels)


def test_strut_direction_property():
    """Strut.direction should return the (end - start) vector."""
    s = tessellation.Strut(
        index=0,
        start_index=0,
        end_index=1,
        start=(0.0, 0.0, 0.0),
        end=(3.0, 4.0, 0.0),
        panel_indices=(0,),
        primary_normal=(0.0, 0.0, 1.0),
        secondary_normal=None,
        length=5.0,
    )
    dx, dy, dz = s.direction
    assert math.isclose(dx, 3.0, abs_tol=1e-12)
    assert math.isclose(dy, 4.0, abs_tol=1e-12)
    assert math.isclose(dz, 0.0, abs_tol=1e-12)
    # Magnitude should equal the strut length.
    mag = math.sqrt(dx**2 + dy**2 + dz**2)
    assert math.isclose(mag, s.length, abs_tol=1e-12)


def test_strut_midpoint_property():
    """Strut.midpoint should return the geometric center of start/end."""
    s = tessellation.Strut(
        index=0,
        start_index=0,
        end_index=1,
        start=(1.0, 2.0, 3.0),
        end=(5.0, 6.0, 7.0),
        panel_indices=(0,),
        primary_normal=(0.0, 0.0, 1.0),
        secondary_normal=None,
        length=1.0,
    )
    mx, my, mz = s.midpoint
    assert math.isclose(mx, 3.0)
    assert math.isclose(my, 4.0)
    assert math.isclose(mz, 5.0)


def test_validate_rejects_zero_frequency():
    """DomeParameters.validate() must reject frequency < 1."""
    with pytest.raises(ValueError, match="(?i)frequency"):
        DomeParameters(frequency=0).validate()


def test_validate_rejects_negative_radius():
    """DomeParameters.validate() must reject radius_m <= 0."""
    with pytest.raises(ValueError, match="(?i)radius"):
        DomeParameters(radius_m=0.0).validate()


def test_validate_rejects_hemisphere_out_of_range():
    """hemisphere_ratio must be in (0, 1]."""
    with pytest.raises(ValueError, match="(?i)hemisphere"):
        DomeParameters(hemisphere_ratio=0.0).validate()
    with pytest.raises(ValueError, match="(?i)hemisphere"):
        DomeParameters(hemisphere_ratio=1.5).validate()


def test_parameters_roundtrip_dict():
    """to_dict → from_dict should preserve all key values."""
    original = DomeParameters(
        radius_m=4.5,
        frequency=3,
        truncation_ratio=0.2,
        hemisphere_ratio=0.7,
        stock_width_m=0.06,
    )
    rebuilt = DomeParameters.from_dict(original.to_dict())
    assert math.isclose(rebuilt.radius_m, original.radius_m)
    assert rebuilt.frequency == original.frequency
    assert math.isclose(rebuilt.truncation_ratio, original.truncation_ratio)
    assert math.isclose(rebuilt.hemisphere_ratio, original.hemisphere_ratio)
    assert math.isclose(rebuilt.stock_width_m, original.stock_width_m)


def test_tessellated_dome_structure_validation():
    """validate_structure() should detect mismatched node references."""
    params = DomeParameters(radius_m=1.0)
    nodes = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    # Panel references node index 5 which doesn't exist.
    bad_panel = PanelDef(index=0, node_indices=(0, 1, 5), normal=(0.0, 0.0, 1.0))
    dome = TessellatedDome(nodes=nodes, struts=[], panels=[bad_panel])
    with pytest.raises((IndexError, Exception)):
        tessellation.validate_structure(dome, params)


# ---------------------------------------------------------------------------
# Ventilation tests
# ---------------------------------------------------------------------------


def _make_test_dome(params=None):
    """Build a real tessellated dome for ventilation tests."""
    if params is None:
        params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    mesh = icosahedron.build_icosahedron(params.radius_m)
    return tessellation.tessellate(mesh, params), params


def test_floor_area_calculation():
    """floor_area_m2 should return π·(r²-belt_z²) for the circular footprint."""
    params = DomeParameters(radius_m=3.0, hemisphere_ratio=0.625)
    area = ventilation.floor_area_m2(params)
    belt_z = 3.0 * (1.0 - 2.0 * 0.625)  # -0.75
    expected = math.pi * (9.0 - belt_z ** 2)  # π * (9 - 0.5625)
    assert math.isclose(area, expected, rel_tol=1e-9)


def test_floor_area_full_hemisphere():
    """Full hemisphere (ratio=1.0) should give floor area = π·r²."""
    params = DomeParameters(radius_m=2.0, hemisphere_ratio=1.0)
    area = ventilation.floor_area_m2(params)
    # belt_z = 2*(1-2*1) = -2, disc = 4-4 = 0 → area 0
    # Wait, hemisphere_ratio=1.0 means belt_z = r*(1-2) = -r
    # disc = r²-r² = 0 → this gives 0 which is mathematically correct:
    # the dome goes all the way down to -r where the base is a single point.
    assert math.isclose(area, 0.0, abs_tol=1e-9)


def test_floor_area_half_sphere():
    """hemisphere_ratio=0.5 means belt at equator; floor area = π·r²."""
    params = DomeParameters(radius_m=3.0, hemisphere_ratio=0.5)
    area = ventilation.floor_area_m2(params)
    assert math.isclose(area, math.pi * 9.0, rel_tol=1e-9)


def test_ventilation_apex_selects_top_panel():
    """Apex vent should pick the panel with the highest centroid z."""
    dome, params = _make_test_dome()
    params.ventilation_apex_count = 1
    vents = ventilation.select_apex_vents(dome, params, max_panels=1)
    assert len(vents) == 1
    vp = vents[0]
    assert vp.vent_type == "apex"

    # The selected panel's centroid z should be higher than 95 % of panels.
    all_z = [
        sum(dome.nodes[ni][2] for ni in p.node_indices) / len(p.node_indices)
        for p in dome.panels
    ]
    all_z.sort()
    threshold = all_z[int(len(all_z) * 0.95)]
    assert vp.centroid[2] >= threshold


def test_ventilation_apex_multiple():
    """Requesting N apex vents should return N panels near the top."""
    dome, params = _make_test_dome()
    vents = ventilation.select_apex_vents(dome, params, max_panels=3)
    assert len(vents) == 3
    zs = [v.centroid[2] for v in vents]
    # All should be in the upper quartile.
    max_z = max(
        sum(dome.nodes[ni][2] for ni in p.node_indices) / len(p.node_indices)
        for p in dome.panels
    )
    for z in zs:
        assert z > max_z * 0.6  # well within upper half


def test_ventilation_ring_selects_mid_height():
    """Ring vents should cluster around the target height ratio."""
    dome, params = _make_test_dome()
    params.ventilation_ring_height_ratio = 0.5
    vents = ventilation.select_ring_vents(dome, params, max_panels=6)
    assert len(vents) >= 3  # should find at least a few

    belt_z = params.radius_m * (1.0 - 2.0 * params.hemisphere_ratio)
    dome_height = params.radius_m - belt_z
    target_z = belt_z + dome_height * 0.5

    for v in vents:
        # Each vent centroid should be within 40% of dome height from target.
        assert abs(v.centroid[2] - target_z) < dome_height * 0.4


def test_ventilation_ring_azimuth_spread():
    """Ring vents should be distributed around the perimeter."""
    dome, params = _make_test_dome()
    vents = ventilation.select_ring_vents(dome, params, max_panels=4)
    if len(vents) < 2:
        pytest.skip("Not enough ring vent candidates")

    azimuths = sorted(
        math.atan2(v.centroid[1], v.centroid[0]) % (2.0 * math.pi)
        for v in vents
    )
    # Check gaps between consecutive azimuths are not too uneven.
    gaps = [azimuths[i + 1] - azimuths[i] for i in range(len(azimuths) - 1)]
    if azimuths:
        gaps.append(2.0 * math.pi - azimuths[-1] + azimuths[0])
    max_gap = max(gaps)
    # For 4 vents, ideal gap = π/2. Allow up to 2× ideal.
    assert max_gap < 2.0 * (2.0 * math.pi / len(vents))


def test_ventilation_plan_auto_mode():
    """Auto mode should combine apex + ring vents."""
    dome, params = _make_test_dome()
    params.generate_ventilation = True
    params.ventilation_mode = "auto"
    params.ventilation_apex_count = 1
    params.ventilation_ring_count = 6
    params.ventilation_target_ratio = 0.20
    plan = ventilation.plan_ventilation(dome, params)

    assert plan.floor_area_m2 > 0
    assert plan.total_vent_area_m2 > 0
    assert plan.vent_ratio > 0
    # Should have both apex and ring vents.
    types = {v.vent_type for v in plan.vents}
    assert "apex" in types
    assert "ring" in types


def test_ventilation_plan_manual_mode():
    """Manual mode should only open the requested panel indices."""
    dome, params = _make_test_dome()
    params.ventilation_mode = "manual"
    params.ventilation_panel_indices = [0, 5, 10]
    plan = ventilation.plan_ventilation(dome, params)

    assert len(plan.vents) == 3
    indices = {v.panel_index for v in plan.vents}
    assert indices == {0, 5, 10}
    for v in plan.vents:
        assert v.vent_type == "manual"


def test_ventilation_plan_to_dict():
    """VentilationPlan.to_dict() should produce a JSON-serializable dict."""
    dome, params = _make_test_dome()
    params.ventilation_mode = "apex"
    params.ventilation_apex_count = 1
    plan = ventilation.plan_ventilation(dome, params)
    d = plan.to_dict()

    assert "vent_count" in d
    assert "floor_area_m2" in d
    assert "total_vent_area_m2" in d
    assert "vent_ratio" in d
    assert "meets_target" in d
    assert isinstance(d["panels"], list)
    assert len(d["panels"]) == d["vent_count"]

    # Should be JSON-serializable.
    import json
    json.dumps(d)  # should not raise


def test_ventilation_hinge_edge_apex():
    """Apex vent hinge edge should be the lowest edge of the panel."""
    dome, params = _make_test_dome()
    vents = ventilation.select_apex_vents(dome, params, max_panels=1)
    assert len(vents) == 1
    vp = vents[0]
    panel = dome.panels[vp.panel_index]

    # The hinge edge's average z should be <= all other edges' average z.
    hinge_avg_z = (dome.nodes[vp.hinge_edge[0]][2] + dome.nodes[vp.hinge_edge[1]][2]) / 2
    for i in range(len(panel.node_indices)):
        a = panel.node_indices[i]
        b = panel.node_indices[(i + 1) % len(panel.node_indices)]
        edge_avg_z = (dome.nodes[a][2] + dome.nodes[b][2]) / 2
        assert hinge_avg_z <= edge_avg_z + 1e-9


def test_ventilation_hinge_edge_ring():
    """Ring vent hinge edge should be the highest edge (hopper style)."""
    dome, params = _make_test_dome()
    vents = ventilation.select_ring_vents(dome, params, max_panels=1)
    if not vents:
        pytest.skip("No ring vent candidates")
    vp = vents[0]
    panel = dome.panels[vp.panel_index]

    hinge_avg_z = (dome.nodes[vp.hinge_edge[0]][2] + dome.nodes[vp.hinge_edge[1]][2]) / 2
    for i in range(len(panel.node_indices)):
        a = panel.node_indices[i]
        b = panel.node_indices[(i + 1) % len(panel.node_indices)]
        edge_avg_z = (dome.nodes[a][2] + dome.nodes[b][2]) / 2
        assert hinge_avg_z >= edge_avg_z - 1e-9


def test_ventilation_vent_panel_area_positive():
    """Every VentPanel should have a positive area."""
    dome, params = _make_test_dome()
    params.ventilation_mode = "auto"
    plan = ventilation.plan_ventilation(dome, params)
    for v in plan.vents:
        assert v.area_m2 > 0


def test_ventilation_open_direction_outward():
    """Open direction vector should point away from the dome center."""
    dome, params = _make_test_dome()
    vents = ventilation.select_apex_vents(dome, params, max_panels=1)
    assert len(vents) == 1
    vp = vents[0]
    # Dot product of centroid and open_direction should be positive (outward).
    dot = sum(c * d for c, d in zip(vp.centroid, vp.open_direction))
    assert dot > 0


def test_ventilation_invalid_manual_index_skipped():
    """Out-of-range panel indices should be silently skipped."""
    dome, params = _make_test_dome()
    params.ventilation_mode = "manual"
    params.ventilation_panel_indices = [0, 99999]
    plan = ventilation.plan_ventilation(dome, params)
    assert len(plan.vents) == 1
    assert plan.vents[0].panel_index == 0


def test_ventilation_summary_string():
    """VentilationPlan.summary() should return a readable string."""
    dome, params = _make_test_dome()
    params.ventilation_mode = "apex"
    params.ventilation_apex_count = 1
    plan = ventilation.plan_ventilation(dome, params)
    s = plan.summary()
    assert "vent panel" in s.lower() or "1 vent" in s.lower()
    assert "%" in s or "m²" in s


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


def test_pipeline_default_step_count():
    """Default pipeline should have 28 steps."""
    steps = pipeline.default_steps()
    assert len(steps) == 28


def test_pipeline_default_step_names():
    """Pipeline step names should match the expected order."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    assert names == [
        "tessellation",
        "auto_door_angle",
        "entry_porch",
        "base_wall",
        "riser_wall",
        "strut_generation",
        "node_connectors",
        "strut_bolt_holes",
        "panel_generation",
        "glass_panels",
        "door_opening",
        "ventilation",
        "skylights",
        "covering_report",
        "foundation",
        "load_calculation",
        "structural_check",
        "production_drawings",
        "cnc_export",
        "techdraw",
        "assembly_guide",
        "multi_dome",
        "weather_protection",
        "cost_estimation",
        "spreadsheets",
        "panel_accuracy_report",
        "manifest_export",
        "model_export",
    ]


def test_pipeline_insert_before():
    """insert_before should place a step at the correct position."""
    p = pipeline.DomePipeline()

    class CustomStep(pipeline.PipelineStep):
        name = "custom"
        def execute(self, ctx):
            pass

    p.insert_before("base_wall", CustomStep())
    names = [s.name for s in p.steps]
    idx_custom = names.index("custom")
    idx_wall = names.index("base_wall")
    assert idx_custom == idx_wall - 1


def test_pipeline_insert_after():
    """insert_after should place a step after the reference."""
    p = pipeline.DomePipeline()

    class CustomStep(pipeline.PipelineStep):
        name = "custom"
        def execute(self, ctx):
            pass

    p.insert_after("tessellation", CustomStep())
    names = [s.name for s in p.steps]
    assert names[1] == "custom"
    assert names[0] == "tessellation"


def test_pipeline_remove():
    """remove should drop the named step."""
    p = pipeline.DomePipeline()
    p.remove("model_export")
    names = [s.name for s in p.steps]
    assert "model_export" not in names
    assert len(p.steps) == 27


def test_pipeline_replace():
    """replace should swap an existing step for a new one."""
    p = pipeline.DomePipeline()

    class NoOpStep(pipeline.PipelineStep):
        name = "tessellation"
        def execute(self, ctx):
            pass

    p.replace("tessellation", NoOpStep())
    assert isinstance(p.steps[0], NoOpStep)


def test_pipeline_tessellation_only(tmp_path):
    """Running a pipeline with only TessellationStep should fill ctx.dome."""
    params = DomeParameters(radius_m=2.0, frequency=2, hemisphere_ratio=0.625)
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    p = pipeline.DomePipeline(steps=[pipeline.TessellationStep()])
    p.run(ctx)

    assert ctx.dome is not None
    assert len(ctx.dome.nodes) > 0
    assert len(ctx.dome.panels) > 0


def test_pipeline_ventilation_step_skipped_when_disabled(tmp_path):
    """VentilationStep.should_run should return False when generate_ventilation is off."""
    params = DomeParameters(generate_ventilation=False)
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    step = pipeline.VentilationStep()
    assert step.should_run(ctx) is False


def test_pipeline_ventilation_step_runs_when_enabled(tmp_path):
    """VentilationStep should produce a ventilation plan on ctx."""
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        generate_ventilation=True, ventilation_mode="apex",
        ventilation_apex_count=1,
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    # First tessellate to populate ctx.dome.
    pipeline.TessellationStep().execute(ctx)
    # Then run ventilation.
    pipeline.VentilationStep().execute(ctx)

    assert hasattr(ctx, "ventilation_plan")
    plan = ctx.ventilation_plan
    assert len(plan.vents) == 1
    assert (tmp_path / "ventilation_plan.json").exists()


def test_validate_ventilation_mode():
    """Validation should reject unknown ventilation modes."""
    p = DomeParameters(ventilation_mode="invalid_mode", generate_ventilation=True)
    with pytest.raises(ValueError, match="(?i)ventilation"):
        p.validate()


def test_validate_ventilation_target_ratio_range():
    """Validation should reject ventilation target ratio outside [0, 1]."""
    with pytest.raises(ValueError, match="(?i)ventilation"):
        DomeParameters(ventilation_target_ratio=1.5, generate_ventilation=True).validate()
    with pytest.raises(ValueError, match="(?i)ventilation"):
        DomeParameters(ventilation_target_ratio=-0.1, generate_ventilation=True).validate()


# ---------------------------------------------------------------------------
# Node connector tests
# ---------------------------------------------------------------------------


def test_incident_map_covers_all_nodes():
    """build_incident_map should have an entry for every strut endpoint."""
    dome, _params = _make_test_dome()
    inc = node_connectors.build_incident_map(dome)

    # Every strut's start and end node must be in the map.
    for s in dome.struts:
        assert s.start_index in inc
        assert s.end_index in inc


def test_incident_map_valence():
    """Each node should have ≥ 3 incident struts (geodesic property)."""
    dome, _params = _make_test_dome()
    inc = node_connectors.build_incident_map(dome)

    for node_idx, items in inc.items():
        # Belt nodes may have lower valence, but dome nodes should have ≥ 3.
        assert len(items) >= 2, f"Node {node_idx} has only {len(items)} incident struts"


def test_connector_builder_creates_connectors():
    """NodeConnectorBuilder should create one connector per node."""
    dome, params = _make_test_dome()
    params.generate_node_connectors = True
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)

    # Should have one connector for every node that has incident struts.
    inc = node_connectors.build_incident_map(dome)
    assert len(connectors) == len(inc)


def test_connector_valence_matches_incident():
    """Each connector's valence should match incident strut count."""
    dome, params = _make_test_dome()
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)
    inc = node_connectors.build_incident_map(dome)

    for nc in connectors:
        expected = len(inc[nc.node_index])
        assert nc.valence == expected
        assert len(nc.bolt_positions) == expected


def test_connector_bolt_positions_offset():
    """Bolt positions should be at the configured offset from node center."""
    dome, params = _make_test_dome()
    offset = 0.030
    params.node_connector_bolt_offset_m = offset
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)

    for nc in connectors:
        for bp in nc.bolt_positions:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(bp.center, nc.position)))
            assert math.isclose(dist, offset, rel_tol=1e-6), (
                f"Bolt at node {nc.node_index} is {dist:.6f}m from center, expected {offset}"
            )


def test_connector_plate_radius_includes_margin():
    """Plate radius should be at least bolt_offset + bolt_radius + margin."""
    dome, params = _make_test_dome()
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)

    min_r = params.node_connector_bolt_offset_m + params.node_connector_bolt_diameter_m * 0.5
    for nc in connectors:
        assert nc.plate_radius_m > min_r


def test_connector_normal_is_outward():
    """Connector normal (plate normal) should point outward from dome center."""
    dome, params = _make_test_dome()
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)

    for nc in connectors:
        if nc.is_belt_node:
            # Belt nodes use (0, 0, 1) as normal — pointing up, which is correct
            # for the belt plane even though the node z is below the equator.
            assert nc.normal == (0.0, 0.0, 1.0)
        else:
            # Non-belt nodes: dot of position and normal should be positive (outward).
            dot = sum(p * n for p, n in zip(nc.position, nc.normal))
            assert dot > 0, f"Node {nc.node_index} normal not outward"


def test_connector_to_dict_serializable():
    """NodeConnector.to_dict() should produce JSON-serializable output."""
    import json
    dome, params = _make_test_dome()
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)

    for nc in connectors[:5]:
        d = nc.to_dict()
        json.dumps(d)  # must not raise
        assert d["node_index"] == nc.node_index
        assert d["valence"] == nc.valence
        assert len(d["bolts"]) == nc.valence


def test_connector_min_strut_angle():
    """min_strut_angle_deg should be positive and < 180 for geodesic nodes."""
    dome, params = _make_test_dome()
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)

    for nc in connectors:
        if nc.valence >= 2 and not nc.is_belt_node:
            angle = nc.min_strut_angle_deg()
            assert 0 < angle < 180, f"Node {nc.node_index}: angle={angle}"


def test_connector_angular_order():
    """angular_order should list strut indices in circular order."""
    dome, params = _make_test_dome()
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)

    for nc in connectors:
        assert len(nc.angular_order) == nc.valence
        # All entries should be valid strut indices.
        for si in nc.angular_order:
            assert 0 <= si < len(dome.struts)


def test_connector_bom_rows():
    """connector_bom_rows should produce BOM with plates and fasteners."""
    dome, params = _make_test_dome()
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)
    rows = node_connectors.connector_bom_rows(connectors, params)

    # Should have at least: plate groups + bolt + nut + washer rows.
    assert len(rows) >= 4
    items = [r["item"] for r in rows]
    assert any("plate" in i.lower() or "gusset" in i.lower() for i in items)
    assert any("bolt" in i.lower() for i in items)
    assert any("nut" in i.lower() for i in items)
    assert any("washer" in i.lower() for i in items)

    # Total bolt quantity should equal sum of all valences.
    bolt_row = next(r for r in rows if "bolt" in r["item"].lower())
    expected_bolts = sum(nc.valence for nc in connectors)
    assert bolt_row["quantity"] == expected_bolts


def test_connector_write_report(tmp_path):
    """write_connector_report should create a JSON file."""
    dome, params = _make_test_dome()
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)

    report_path = tmp_path / "connectors.json"
    node_connectors.write_connector_report(connectors, report_path)

    assert report_path.exists()
    import json
    data = json.loads(report_path.read_text())
    assert data["total_connectors"] == len(connectors)
    assert "valence_histogram" in data
    assert len(data["connectors"]) == len(connectors)


def test_pipeline_node_connector_step_count():
    """Default pipeline should now have 28 steps (including NodeConnectorStep + StrutBoltHoleStep)."""
    steps = pipeline.default_steps()
    assert len(steps) == 28


def test_pipeline_node_connector_step_position():
    """NodeConnectorStep should be after StrutGenerationStep."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    assert "node_connectors" in names
    strut_idx = names.index("strut_generation")
    conn_idx = names.index("node_connectors")
    assert conn_idx == strut_idx + 1


def test_pipeline_connector_step_skipped_when_disabled(tmp_path):
    """NodeConnectorStep.should_run should return False when disabled."""
    params = DomeParameters(generate_node_connectors=False)
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    step = pipeline.NodeConnectorStep()
    assert step.should_run(ctx) is False


def test_pipeline_connector_step_runs(tmp_path):
    """NodeConnectorStep should produce connectors on ctx."""
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        generate_node_connectors=True,
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    pipeline.TessellationStep().execute(ctx)
    pipeline.NodeConnectorStep().execute(ctx)

    assert hasattr(ctx, "node_connectors")
    assert len(ctx.node_connectors) > 0
    assert (tmp_path / "node_connector_report.json").exists()


def test_validate_node_connector_type():
    """Validation should reject unknown connector types."""
    with pytest.raises(ValueError, match="(?i)node_connector"):
        DomeParameters(node_connector_type="invalid").validate()


def test_validate_node_connector_dims():
    """Validation should reject non-positive connector dimensions."""
    with pytest.raises(ValueError, match="(?i)node_connector_thickness"):
        DomeParameters(node_connector_thickness_m=0).validate()
    with pytest.raises(ValueError, match="(?i)node_connector_bolt_diameter"):
        DomeParameters(node_connector_bolt_diameter_m=-1).validate()
    with pytest.raises(ValueError, match="(?i)node_connector_bolt_offset"):
        DomeParameters(node_connector_bolt_offset_m=0).validate()


def test_validate_strut_profile():
    """Validation should reject unknown strut profiles."""
    with pytest.raises(ValueError, match="(?i)strut_profile"):
        DomeParameters(strut_profile="hexagonal").validate()


def test_validate_strut_profile_accepts_all():
    """All three strut profiles should pass validation."""
    for profile in ("rectangular", "round", "trapezoidal"):
        DomeParameters(strut_profile=profile).validate()


def test_validate_cap_blend_mode():
    """Validation should reject unknown cap blend modes."""
    with pytest.raises(ValueError, match="(?i)cap_blend_mode"):
        DomeParameters(cap_blend_mode="bezier").validate()


def test_validate_cap_blend_mode_accepts_all():
    """All three cap blend modes should pass validation."""
    for mode in ("sharp", "chamfer", "fillet"):
        DomeParameters(cap_blend_mode=mode).validate()


def test_validate_connector_type_lapjoint():
    """Lap joint should be a valid connector type."""
    DomeParameters(node_connector_type="lapjoint").validate()


def test_validate_connector_type_ball_pipe():
    """Ball and pipe should be valid connector types."""
    for ct in ("plate", "ball", "pipe", "lapjoint"):
        DomeParameters(node_connector_type=ct).validate()


def test_connector_lapjoint_metadata():
    """Lap-joint connectors should have correct type and bolt positions."""
    params = DomeParameters(
        frequency=2,
        generate_node_connectors=True,
        node_connector_type="lapjoint",
    )
    dome, params = _make_test_dome(params)
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)
    assert len(connectors) > 0
    for nc in connectors:
        assert nc.connector_type == "lapjoint"
        assert nc.valence >= 2
        assert len(nc.bolt_positions) == nc.valence


def test_connector_ball_metadata():
    """Ball connectors should have correct type."""
    params = DomeParameters(
        frequency=2,
        generate_node_connectors=True,
        node_connector_type="ball",
    )
    dome, params = _make_test_dome(params)
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)
    assert len(connectors) > 0
    for nc in connectors:
        assert nc.connector_type == "ball"


def test_connector_pipe_metadata():
    """Pipe connectors should have correct type."""
    params = DomeParameters(
        frequency=2,
        generate_node_connectors=True,
        node_connector_type="pipe",
    )
    dome, params = _make_test_dome(params)
    builder = node_connectors.NodeConnectorBuilder(params)
    connectors = builder.create_connectors(dome)
    assert len(connectors) > 0
    for nc in connectors:
        assert nc.connector_type == "pipe"


def test_endpoint_inset_excludes_plate_for_lapjoint():
    """Lap joint should not add plate thickness to strut inset."""
    from freecad_dome.struts import StrutBuilder

    params_plate = DomeParameters(
        generate_node_connectors=True,
        node_connector_type="plate",
        connector_strut_inset=True,
        node_connector_thickness_m=0.006,
        kerf_m=0.002,
        node_fit_extension_m=0.0,  # disable extension for this test
    )
    params_lap = DomeParameters(
        generate_node_connectors=True,
        node_connector_type="lapjoint",
        connector_strut_inset=True,
        node_connector_thickness_m=0.006,
        kerf_m=0.002,
        node_fit_extension_m=0.0,
    )
    builder_plate = StrutBuilder(params_plate)
    builder_lap = StrutBuilder(params_lap)
    inset_plate = builder_plate._endpoint_inset_m()
    inset_lap = builder_lap._endpoint_inset_m()
    # Plate adds half-thickness; lapjoint does not.
    assert inset_plate > inset_lap
    assert inset_lap == pytest.approx(0.002)  # kerf only
    assert inset_plate == pytest.approx(0.002 + 0.003)  # kerf + half-thickness


def test_node_fit_extension_reduces_inset():
    """node_fit_extension_m should reduce the effective endpoint inset."""
    from freecad_dome.struts import StrutBuilder

    # Without extension: inset = kerf = 0.002
    params_no_ext = DomeParameters(kerf_m=0.002, node_fit_extension_m=0.0)
    builder_no_ext = StrutBuilder(params_no_ext)
    inset_no_ext = builder_no_ext._endpoint_inset_m()
    assert inset_no_ext == pytest.approx(0.002)

    # With extension = 0.005: inset = 0.002 - 0.005 = -0.003 (strut extends past node)
    params_ext = DomeParameters(kerf_m=0.002, node_fit_extension_m=0.005)
    builder_ext = StrutBuilder(params_ext)
    inset_ext = builder_ext._endpoint_inset_m()
    assert inset_ext == pytest.approx(-0.003)
    assert inset_ext < inset_no_ext


def test_node_fit_extension_default():
    """Default node_fit_extension_m should be 0.005."""
    params = DomeParameters()
    assert params.node_fit_extension_m == pytest.approx(0.005)


def test_node_fit_plane_mode_miter_valid():
    """'miter' should be a valid node_fit_plane_mode."""
    params = DomeParameters(node_fit_plane_mode="miter")
    params.validate()  # should not raise


def test_node_fit_plane_mode_invalid():
    """Invalid node_fit_plane_mode should raise ValueError."""
    params = DomeParameters(node_fit_plane_mode="invalid_mode")
    with pytest.raises(ValueError, match="node_fit_plane_mode"):
        params.validate()


def test_node_fit_extension_negative_rejected():
    """Negative node_fit_extension_m should raise ValueError."""
    params = DomeParameters(node_fit_extension_m=-0.001)
    with pytest.raises(ValueError, match="node_fit_extension_m"):
        params.validate()


def test_node_fit_mode_valid():
    """All three node_fit_mode values should be accepted."""
    for mode in ("planar", "tapered", "voronoi"):
        p = DomeParameters(node_fit_mode=mode)
        p.validate()


def test_node_fit_mode_invalid():
    """Invalid node_fit_mode should raise ValueError."""
    p = DomeParameters(node_fit_mode="magic")
    with pytest.raises(ValueError, match="node_fit_mode"):
        p.validate()


def test_node_fit_taper_ratio_bounds():
    """node_fit_taper_ratio must be between 0 and 1 (exclusive)."""
    for bad in (0.0, 1.0, -0.1, 1.5):
        p = DomeParameters(node_fit_taper_ratio=bad)
        with pytest.raises(ValueError, match="node_fit_taper_ratio"):
            p.validate()
    # Valid values should pass.
    DomeParameters(node_fit_taper_ratio=0.3).validate()
    DomeParameters(node_fit_taper_ratio=0.9).validate()


def test_voronoi_cut_planes_all_pairwise():
    """Voronoi mode should produce bisector planes for all strut pairs at a node."""
    from freecad_dome.struts import StrutBuilder

    # Use hemisphere_ratio=1.0 (full sphere, no belt) to avoid belt-node special cases.
    params = DomeParameters(node_fit_mode="voronoi", hemisphere_ratio=1.0)
    builder = StrutBuilder(params)

    mesh = icosahedron.build_icosahedron(params.radius_m)
    if params.use_truncation:
        mesh, _ = icosahedron.truncate_mesh(mesh, params.truncation_ratio)
    dome = tessellation.tessellate(mesh, params)

    builder._endpoint_cut_planes = builder._compute_endpoint_cut_planes(dome)

    # For every strut endpoint, count the number of planes computed.
    # In Voronoi mode, the number of planes = (valence - 1) where valence is
    # the number of struts meeting at that node.
    # Build node-to-strut mapping.
    from collections import Counter
    node_valence: Counter[int] = Counter()
    for s in dome.struts:
        node_valence[s.start_index] += 1
        node_valence[s.end_index] += 1

    for s in dome.struts:
        for end_label, node_idx in (("start", s.start_index), ("end", s.end_index)):
            planes = builder._endpoint_cut_planes.get((s.index, end_label), [])
            expected = node_valence[node_idx] - 1
            assert len(planes) == expected, (
                f"Strut {s.index} {end_label}: expected {expected} Voronoi planes, got {len(planes)}"
            )


def test_tapered_mode_defaults():
    """Tapered mode defaults should be correct."""
    params = DomeParameters()
    assert params.node_fit_mode == "planar"
    assert params.node_fit_taper_ratio == pytest.approx(0.6)


def test_structure_config_new_fields_defaults():
    """New E-series and G-series fields should have correct defaults."""
    params = DomeParameters()
    assert params.min_strut_length_factor == 0.5
    assert params.prism_only_length_factor == 3.0
    assert params.cap_length_factor == 2.0
    assert params.max_cap_ratio == 0.45
    assert params.split_keep_offset_factor == 0.35
    assert params.min_wedge_angle_deg == 15.0
    assert params.bevel_fillet_radius_m == 0.0
    assert params.cap_blend_mode == "sharp"
    assert params.strut_profile == "rectangular"
    assert params.connector_strut_inset is True
    assert params.node_connector_lap_extension_m == 0.03
    assert params.node_fit_extension_m == 0.005
    assert params.node_fit_mode == "planar"
    assert params.node_fit_taper_ratio == 0.6


def test_vec3_module_functions():
    """Shared vec3 module should provide correct results."""
    from freecad_dome import vec3 as v3

    assert v3.norm((3, 4, 0)) == pytest.approx(5.0)
    n = v3.normalize((3, 4, 0))
    assert v3.norm(n) == pytest.approx(1.0)
    assert v3.dot((1, 0, 0), (0, 1, 0)) == pytest.approx(0.0)
    c = v3.cross((1, 0, 0), (0, 1, 0))
    assert c == pytest.approx((0, 0, 1))
    assert v3.sub((3, 4, 5), (1, 1, 1)) == pytest.approx((2, 3, 4))
    assert v3.add((1, 2, 3), (4, 5, 6)) == pytest.approx((5, 7, 9))
    assert v3.scale((1, 2, 3), 2) == pytest.approx((2, 4, 6))
    import math
    assert v3.angle_between((1, 0, 0), (0, 1, 0)) == pytest.approx(math.pi / 2)
    l = v3.lerp((0, 0, 0), (10, 10, 10), 0.5)
    assert l == pytest.approx((5, 5, 5))


def test_node_fit_data_structure():
    """NodeFitData should contain entries for every node with struts."""
    from freecad_dome.node_fit import compute_node_fit_data

    params = DomeParameters(frequency=2)
    dome, params = _make_test_dome(params)
    nfd = compute_node_fit_data(dome, params.hemisphere_ratio, params.radius_m)
    assert len(nfd.nodes) > 0
    for node_idx, nfi in nfd.nodes.items():
        assert nfi.node_index == node_idx
        assert len(nfi.position) == 3
        assert len(nfi.radial) == 3
        assert len(nfi.incident) >= 2
        assert len(nfi.angular_order) == len(nfi.incident)
        assert isinstance(nfi.tangent_directions, dict)


# ---------------------------------------------------------------------------
# Covering material tests
# ---------------------------------------------------------------------------

def test_covering_catalogue_has_entries():
    """COVERINGS catalogue should have at least glass + polycarbonate entries."""
    assert len(covering.COVERINGS) >= 8
    assert "glass" in covering.COVERINGS
    assert "polycarbonate_twin_8" in covering.COVERINGS


def test_covering_attachment_profiles():
    """ATTACHMENT_PROFILES should have at least 4 entries."""
    assert len(covering.ATTACHMENT_PROFILES) >= 4
    assert "H_profile_alu" in covering.ATTACHMENT_PROFILES
    profile = covering.ATTACHMENT_PROFILES["H_profile_alu"]
    assert profile.weight_kg_m > 0
    assert profile.material == "aluminum"


def test_covering_spec_weight():
    """Weight per m² should equal density × thickness."""
    spec = covering.COVERINGS["polycarbonate_twin_8"]
    expected = spec.density_kg_m3 * spec.thickness_m
    assert abs(spec.weight_kg_m2 - expected) < 1e-6


def test_covering_spec_to_dict():
    """to_dict should include key fields."""
    spec = covering.COVERINGS["glass"]
    d = spec.to_dict()
    assert d["key"] == "glass"
    assert d["category"] == "glass"
    assert "weight_kg_m2" in d
    assert "u_value_w_m2k" in d


def test_covering_for_params_glass_default():
    """Default params should yield glass covering."""
    params = DomeParameters()
    spec = covering.covering_for_params(params)
    assert spec.category == "glass"
    assert spec.thickness_m == pytest.approx(0.004)


def test_covering_for_params_polycarbonate():
    """Setting covering_type should pick the right material."""
    params = DomeParameters(covering_type="polycarbonate_twin_10")
    spec = covering.covering_for_params(params)
    assert spec.key == "polycarbonate_twin_10"
    assert spec.category == "polycarbonate"
    assert spec.thickness_m == pytest.approx(0.010)


def test_covering_for_params_custom_thickness():
    """covering_thickness_m should override catalogue thickness."""
    params = DomeParameters(
        covering_type="polycarbonate_twin_8",
        covering_thickness_m=0.012,
    )
    spec = covering.covering_for_params(params)
    assert spec.thickness_m == pytest.approx(0.012)


def test_covering_thermal_gap_polycarbonate():
    """Polycarbonate gap should exceed base gap due to thermal expansion."""
    spec = covering.COVERINGS["polycarbonate_twin_8"]
    base_gap = 0.005
    span = 0.6
    delta_t = 40.0
    total = spec.effective_edge_gap_m(span, base_gap, delta_t)
    thermal = total - base_gap
    assert thermal > 0
    # ~0.065 * 0.6 * 40 / 1000 ≈ 0.00156 m
    assert thermal == pytest.approx(0.065 * 0.6 * 40 / 1000, abs=1e-6)
    assert total > base_gap


def test_covering_thermal_gap_glass_small():
    """Glass thermal expansion should be negligible."""
    spec = covering.COVERINGS["glass"]
    base_gap = 0.005
    total = spec.effective_edge_gap_m(0.6, base_gap, 40.0)
    thermal = total - base_gap
    # 0.009 * 0.6 * 40 / 1000 = 0.000216 m — very small
    assert thermal < 0.001


def test_covering_effective_gap_from_params():
    """Module-level effective_edge_gap_m should use params correctly."""
    params = DomeParameters(
        covering_type="polycarbonate_twin_8",
        covering_gap_m=0.005,
        covering_delta_t_k=50.0,
    )
    gap = covering.effective_edge_gap_m(params, span_m=0.5)
    assert gap > 0.005  # must exceed base gap


def test_covering_bom_rows():
    """BOM should contain at least one covering entry."""
    params = DomeParameters(covering_type="polycarbonate_twin_8")
    areas = [0.25, 0.30, 0.20]
    rows = covering.covering_bom_rows(areas, params)
    assert len(rows) >= 1
    assert rows[0]["type"] == "covering"
    assert rows[0]["total_area_m2"] == pytest.approx(0.75, abs=0.01)
    assert rows[0]["weight_kg_total"] > 0


def test_covering_bom_with_profile():
    """BOM should include attachment profile when configured."""
    params = DomeParameters(
        covering_type="polycarbonate_twin_8",
        covering_profile_type="H_profile_alu",
    )
    areas = [0.25, 0.30, 0.20]
    rows = covering.covering_bom_rows(areas, params)
    assert len(rows) == 2
    assert rows[1]["type"] == "attachment_profile"
    assert rows[1]["total_length_m"] > 0


def test_covering_cut_list():
    """Cut list should have one entry per panel."""
    params = DomeParameters(
        covering_type="polycarbonate_twin_8",
        covering_gap_m=0.005,
    )
    panel_data = [
        {"panel_index": 0, "area_m2": 0.25, "edge_lengths_m": [0.5, 0.5, 0.5]},
        {"panel_index": 1, "area_m2": 0.30, "edge_lengths_m": [0.6, 0.5, 0.55]},
    ]
    cuts = covering.covering_cut_list(panel_data, params)
    assert len(cuts) == 2
    assert cuts[0]["covering_type"] == "polycarbonate_twin_8"
    assert cuts[0]["effective_gap_m"] > 0.005  # includes thermal
    assert cuts[1]["max_span_m"] == pytest.approx(0.6, abs=0.01)


def test_covering_write_report(tmp_path):
    """write_covering_report should produce a valid JSON file."""
    import json
    params = DomeParameters(covering_type="polycarbonate_twin_8")
    areas = [0.20, 0.25, 0.30]
    path = tmp_path / "covering_report.json"
    covering.write_covering_report(areas, params, path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["panel_count"] == 3
    assert data["total_area_m2"] == pytest.approx(0.75, abs=0.01)
    assert data["total_weight_kg"] > 0
    assert "bom" in data


def test_covering_unknown_type_fallback():
    """Unknown covering_type should fall back to glass."""
    params = DomeParameters(covering_type="unobtanium")
    spec = covering.covering_for_params(params)
    assert spec.key == "glass"


def test_pipeline_covering_step_skipped_when_no_glass():
    """CoveringReportStep should skip when no covering is configured."""
    params = DomeParameters(glass_thickness_m=0.0, covering_thickness_m=0.0)
    ctx = pipeline.PipelineContext(params=params, out_dir="/tmp")
    step = pipeline.CoveringReportStep()
    assert step.should_run(ctx) is False


def test_pipeline_covering_step_runs_with_glass():
    """CoveringReportStep.should_run should be True when glass is set."""
    params = DomeParameters(glass_thickness_m=0.004)
    ctx = pipeline.PipelineContext(params=params, out_dir="/tmp")
    step = pipeline.CoveringReportStep()
    assert step.should_run(ctx) is True


def test_pipeline_covering_step_runs_with_polycarbonate():
    """CoveringReportStep.should_run should be True for polycarbonate."""
    params = DomeParameters(covering_type="polycarbonate_twin_8")
    ctx = pipeline.PipelineContext(params=params, out_dir="/tmp")
    step = pipeline.CoveringReportStep()
    assert step.should_run(ctx) is True


def test_pipeline_covering_step_execute(tmp_path):
    """CoveringReportStep should produce a covering_report.json."""
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        covering_type="polycarbonate_twin_8",
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    pipeline.TessellationStep().execute(ctx)
    pipeline.CoveringReportStep().execute(ctx)
    report = tmp_path / "covering_report.json"
    assert report.exists()
    import json
    data = json.loads(report.read_text())
    assert data["panel_count"] > 0
    assert data["total_area_m2"] > 0


def test_pipeline_covering_step_position():
    """CoveringReportStep should be between skylights and foundation."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    sky_idx = names.index("skylights")
    cov_idx = names.index("covering_report")
    found_idx = names.index("foundation")
    assert cov_idx == sky_idx + 1
    assert cov_idx == found_idx - 1


def test_validate_covering_thickness_nonneg():
    """Validation should reject negative covering dimensions."""
    with pytest.raises(ValueError, match="(?i)covering_thickness"):
        DomeParameters(covering_thickness_m=-0.001).validate()


def test_validate_covering_gap_nonneg():
    """Negative covering gap should fail validation."""
    with pytest.raises(ValueError, match="(?i)covering_gap"):
        DomeParameters(covering_gap_m=-1.0).validate()


def test_validate_covering_delta_t_nonneg():
    """Negative delta_t should fail validation."""
    with pytest.raises(ValueError, match="(?i)covering_delta_t"):
        DomeParameters(covering_delta_t_k=-5.0).validate()


# ---------------------------------------------------------------------------
# Foundation tests
# ---------------------------------------------------------------------------

def test_foundation_belt_nodes_detected():
    """Belt nodes should be detected for a hemisphere dome."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    plan = foundation.foundation_for_params(dome, params)
    assert plan.anchor_count > 0
    assert plan.belt_radius_m > 0
    # All anchors should be at the belt height (within tolerance).
    for anchor in plan.anchors:
        assert anchor.z_m == pytest.approx(plan.belt_height_m, abs=0.01)


def test_foundation_anchors_sorted_by_azimuth():
    """Anchors should be ordered by atan2 azimuth (may wrap at ±180°)."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    plan = foundation.foundation_for_params(dome, params)
    # The underlying sort uses atan2 which returns (-π, π].
    # Verify that the raw atan2 values are sorted.
    atan2_vals = [math.atan2(a.y_m, a.x_m) for a in plan.anchors]
    assert atan2_vals == sorted(atan2_vals)


def test_foundation_strip_concrete_volume():
    """Strip foundation should have positive concrete volume."""
    dome, _ = _make_test_dome()
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        foundation_type="strip",
    )
    plan = foundation.foundation_for_params(dome, params)
    assert plan.concrete_volume_m3 > 0
    # Volume = circumference * width * (depth + top)
    expected = plan.circumference_m * plan.strip_width_m * (plan.strip_depth_m + plan.strip_top_m)
    assert plan.concrete_volume_m3 == pytest.approx(expected, rel=0.01)


def test_foundation_point_concrete_volume():
    """Point foundation should have positive concrete volume proportional to piers."""
    dome, _ = _make_test_dome()
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        foundation_type="point",
    )
    plan = foundation.foundation_for_params(dome, params)
    assert plan.concrete_volume_m3 > 0


def test_foundation_screw_no_concrete():
    """Screw anchor foundation should have zero concrete volume."""
    dome, _ = _make_test_dome()
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        foundation_type="screw_anchor",
    )
    plan = foundation.foundation_for_params(dome, params)
    assert plan.concrete_volume_m3 == 0.0


def test_foundation_anchor_bolt_defaults():
    """Anchor bolts should use param defaults."""
    dome, _ = _make_test_dome()
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        foundation_bolt_diameter_m=0.020,
    )
    plan = foundation.foundation_for_params(dome, params)
    for anchor in plan.anchors:
        assert anchor.diameter_m == 0.020


def test_foundation_bom_rows():
    """Foundation BOM should include concrete and anchor bolt entries."""
    dome, _ = _make_test_dome()
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        foundation_type="strip",
    )
    plan = foundation.foundation_for_params(dome, params)
    rows = foundation.foundation_bom_rows(plan)
    types = [r["type"] for r in rows]
    assert "concrete" in types
    assert "anchor_bolt" in types


def test_foundation_pour_plan_coordinates():
    """Pour plan should have one entry per anchor."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    plan = foundation.foundation_for_params(dome, params)
    coords = foundation.pour_plan_coordinates(plan)
    assert len(coords) == plan.anchor_count
    for c in coords:
        assert "x_m" in c
        assert "y_m" in c
        assert "azimuth_deg" in c
        assert "bolt_number" in c


def test_foundation_write_report(tmp_path):
    """write_foundation_report should produce a valid JSON file."""
    import json
    dome, _ = _make_test_dome()
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        foundation_type="strip",
    )
    plan = foundation.foundation_for_params(dome, params)
    path = tmp_path / "foundation_report.json"
    foundation.write_foundation_report(plan, params, path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["foundation"]["foundation_type"] == "strip"
    assert data["foundation"]["anchor_count"] > 0
    assert "bom" in data
    assert "pour_plan" in data
    assert len(data["pour_plan"]["coordinates"]) > 0


def test_foundation_plan_to_dict():
    """FoundationPlan.to_dict should include type-specific fields."""
    plan = foundation.FoundationPlan(
        foundation_type="strip",
        belt_radius_m=2.5,
        belt_height_m=-0.75,
    )
    d = plan.to_dict()
    assert d["foundation_type"] == "strip"
    assert "strip_width_m" in d
    assert "concrete_volume_m3" in d


def test_foundation_anchor_spacing():
    """Anchor spacing should equal circumference / count."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    plan = foundation.foundation_for_params(dome, params)
    expected = plan.circumference_m / plan.anchor_count
    assert plan.anchor_spacing_m() == pytest.approx(expected, rel=0.01)


def test_pipeline_foundation_step_skipped_when_disabled():
    """FoundationStep should skip when generate_foundation is False."""
    params = DomeParameters(generate_foundation=False)
    ctx = pipeline.PipelineContext(params=params, out_dir="/tmp")
    step = pipeline.FoundationStep()
    assert step.should_run(ctx) is False


def test_pipeline_foundation_step_execute(tmp_path):
    """FoundationStep should produce foundation_report.json."""
    import json
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        generate_foundation=True,
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    pipeline.TessellationStep().execute(ctx)
    pipeline.FoundationStep().execute(ctx)
    report = tmp_path / "foundation_report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["foundation"]["anchor_count"] > 0


def test_pipeline_foundation_step_position():
    """FoundationStep should be between covering_report and load_calculation."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    cov_idx = names.index("covering_report")
    found_idx = names.index("foundation")
    load_idx = names.index("load_calculation")
    assert found_idx == cov_idx + 1
    assert found_idx == load_idx - 1


def test_validate_foundation_type():
    """Validation should reject unknown foundation types."""
    with pytest.raises(ValueError, match="(?i)foundation_type"):
        DomeParameters(foundation_type="invalid").validate()


def test_validate_foundation_bolt_diameter():
    """Foundation bolt diameter must be positive."""
    with pytest.raises(ValueError, match="(?i)foundation_bolt_diameter"):
        DomeParameters(foundation_bolt_diameter_m=0).validate()


def test_validate_foundation_strip_width():
    """Strip width must be positive."""
    with pytest.raises(ValueError, match="(?i)foundation_strip_width"):
        DomeParameters(foundation_strip_width_m=0).validate()


# ---------------------------------------------------------------------------
# Load calculation tests
# ---------------------------------------------------------------------------

def test_loads_compute_returns_result():
    """compute_loads should return a LoadResult with three cases."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    result = loads.compute_loads(dome, params)
    assert len(result.cases) == 3
    case_names = [c.name for c in result.cases]
    assert "dead" in case_names
    assert "snow" in case_names
    assert "wind" in case_names


def test_loads_dead_weight_positive():
    """Total dead weight should be positive (downward)."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    result = loads.compute_loads(dome, params)
    assert result.total_dead_weight_kn > 0


def test_loads_dead_weight_direction():
    """Dead load should produce downward (negative Z) forces on each node."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    result = loads.compute_loads(dome, params)
    dead_case = [c for c in result.cases if c.name == "dead"][0]
    for nl in dead_case.node_loads:
        assert nl.fz_kn <= 0  # all downward


def test_loads_snow_shape_factor():
    """Snow shape factor should be 0.8 at apex and 0 at 60°+ zenith."""
    assert loads._snow_dome_shape_factor(0) == pytest.approx(0.8)
    assert loads._snow_dome_shape_factor(60) == pytest.approx(0.0)
    assert loads._snow_dome_shape_factor(90) == pytest.approx(0.0)
    # Mid-range should be between 0 and 0.8.
    mid = loads._snow_dome_shape_factor(30)
    assert 0 < mid < 0.8


def test_loads_snow_vertical_only():
    """Snow load should produce only vertical (Z) forces."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    result = loads.compute_loads(dome, params)
    snow_case = [c for c in result.cases if c.name == "snow"][0]
    for nl in snow_case.node_loads:
        assert nl.fx_kn == 0.0
        assert nl.fy_kn == 0.0
        assert nl.fz_kn <= 0  # downward


def test_loads_wind_has_horizontal():
    """Wind load should produce some horizontal forces."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    result = loads.compute_loads(dome, params)
    assert result.total_wind_kn > 0


def test_loads_combinations():
    """Load result should include standard Eurocode combinations."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    result = loads.compute_loads(dome, params)
    assert len(result.combinations) >= 3
    combo_names = [c.name for c in result.combinations]
    assert "ULS_1_snow_dominant" in combo_names
    assert "ULS_2_wind_dominant" in combo_names


def test_loads_node_load_magnitude():
    """NodeLoad magnitude should be consistent with components."""
    nl = loads.NodeLoad(node_index=0, fx_kn=3.0, fy_kn=4.0, fz_kn=0.0)
    assert nl.magnitude_kn == pytest.approx(5.0)


def test_loads_to_dict():
    """LoadResult.to_dict should serialize properly."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    result = loads.compute_loads(dome, params)
    d = result.to_dict()
    assert "total_dead_weight_kn" in d
    assert "total_snow_kn" in d
    assert "cases" in d
    assert len(d["cases"]) == 3


def test_loads_write_report(tmp_path):
    """write_load_report should produce a valid JSON file."""
    import json
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    result = loads.compute_loads(dome, params)
    path = tmp_path / "load_report.json"
    loads.write_load_report(result, params, path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert "summary" in data
    assert "node_loads" in data
    assert "dead" in data["node_loads"]


def test_loads_write_csv(tmp_path):
    """write_load_csv should produce a valid CSV file."""
    dome, _ = _make_test_dome()
    params = DomeParameters(radius_m=3.0, frequency=4, hemisphere_ratio=0.625)
    result = loads.compute_loads(dome, params)
    path = tmp_path / "load_nodes.csv"
    loads.write_load_csv(result, path)
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) > 1  # header + data
    header = lines[0].split(",")
    assert "node_index" in header
    assert "Fz_dead_kN" in header


def test_loads_tributary_areas():
    """Tributary areas should sum to approximately total dome surface area."""
    dome, _ = _make_test_dome()
    tributary = loads._tributary_areas(dome)
    total = sum(tributary.values())
    # For r=3, hemi=0.625, surface area ≈ 2π·r·h where h = r(1+hemi_ratio) ≈ some value
    # Just check it's reasonable (10-100 m²).
    assert 10 < total < 100


def test_pipeline_load_step_skipped_when_disabled():
    """LoadCalculationStep should skip when generate_loads is False."""
    params = DomeParameters(generate_loads=False)
    ctx = pipeline.PipelineContext(params=params, out_dir="/tmp")
    step = pipeline.LoadCalculationStep()
    assert step.should_run(ctx) is False


def test_pipeline_load_step_execute(tmp_path):
    """LoadCalculationStep should produce load_report.json and load_nodes.csv."""
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        generate_loads=True,
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    pipeline.TessellationStep().execute(ctx)
    pipeline.LoadCalculationStep().execute(ctx)
    assert (tmp_path / "load_report.json").exists()
    assert (tmp_path / "load_nodes.csv").exists()


def test_pipeline_load_step_position():
    """LoadCalculationStep should be between foundation and structural_check."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    found_idx = names.index("foundation")
    load_idx = names.index("load_calculation")
    check_idx = names.index("structural_check")
    prod_idx = names.index("production_drawings")
    assert load_idx == found_idx + 1
    assert check_idx == load_idx + 1
    assert prod_idx == check_idx + 1


def test_validate_wind_terrain():
    """Validation should reject unknown wind terrain."""
    with pytest.raises(ValueError, match="(?i)wind_terrain"):
        DomeParameters(load_wind_terrain="X").validate()


def test_validate_snow_zone():
    """Validation should reject unknown snow zone."""
    with pytest.raises(ValueError, match="(?i)snow_zone"):
        DomeParameters(load_snow_zone="X").validate()


def test_validate_wind_speed_nonneg():
    """Wind speed cannot be negative."""
    with pytest.raises(ValueError, match="(?i)wind_speed"):
        DomeParameters(load_wind_speed_ms=-5).validate()


# =========================================================================
# Structural check tests
# =========================================================================


def test_structural_check_returns_result():
    """run_structural_check should return a StructuralCheckResult."""
    dome, params = _make_test_dome()
    load_result = loads.compute_loads(dome, params)
    result = structural_check.run_structural_check(dome, params, load_result)
    assert isinstance(result, structural_check.StructuralCheckResult)
    assert result.total_struts == len(dome.struts)
    assert result.total_struts > 0


def test_structural_check_has_per_strut_results():
    """Each strut should have a StrutCheck result."""
    dome, params = _make_test_dome()
    load_result = loads.compute_loads(dome, params)
    result = structural_check.run_structural_check(dome, params, load_result)
    assert len(result.strut_checks) == len(dome.struts)
    for sc in result.strut_checks:
        assert sc.length_m > 0
        assert sc.euler_buckling_kn > 0
        assert sc.compression_capacity_kn > 0
        assert sc.tension_capacity_kn > 0


def test_structural_check_utilization_nonneg():
    """Utilization ratios should be non-negative."""
    dome, params = _make_test_dome()
    load_result = loads.compute_loads(dome, params)
    result = structural_check.run_structural_check(dome, params, load_result)
    for sc in result.strut_checks:
        assert sc.governing_ratio >= 0
        assert sc.buckling_ratio >= 0
        assert sc.compression_ratio >= 0
        assert sc.tension_ratio >= 0


def test_structural_check_governing_combination():
    """Each checked strut should have a governing combination name."""
    dome, params = _make_test_dome()
    load_result = loads.compute_loads(dome, params)
    result = structural_check.run_structural_check(dome, params, load_result)
    for sc in result.strut_checks:
        if sc.governing_ratio > 0:
            assert sc.governing_combination != ""
            assert sc.governing_check in ("buckling", "compression", "tension")


def test_structural_check_euler_buckling():
    """Euler buckling capacity should scale with I and inversely with L²."""
    dome, params = _make_test_dome()
    load_result = loads.compute_loads(dome, params)
    result = structural_check.run_structural_check(dome, params, load_result)
    sc = result.strut_checks[0]
    # N_cr = π²·E·I / L²
    mat = params.materials[params.material]
    E = float(mat.elastic_modulus)
    I = structural_check._second_moment_of_area(params)
    expected = math.pi ** 2 * E * I / (sc.length_m ** 2) / 1e3  # kN
    assert math.isclose(sc.euler_buckling_kn, expected, rel_tol=1e-6)


def test_structural_check_cross_section_area():
    """Cross-section area for rectangular profile = w × h."""
    params = DomeParameters(stock_width_m=0.05, stock_height_m=0.05)
    A = structural_check._cross_section_area(params)
    assert math.isclose(A, 0.0025, rel_tol=1e-9)


def test_structural_check_cross_section_round():
    """Cross-section area for round profile = π·d²/4."""
    params = DomeParameters(stock_width_m=0.05, stock_height_m=0.05, strut_profile="round")
    A = structural_check._cross_section_area(params)
    expected = math.pi * (0.05 / 2) ** 2
    assert math.isclose(A, expected, rel_tol=1e-9)


def test_structural_check_second_moment_rectangular():
    """I_min for rectangular profile = b·d³/12 (weak axis)."""
    params = DomeParameters(stock_width_m=0.06, stock_height_m=0.04)
    I = structural_check._second_moment_of_area(params)
    # min dim = 0.04, max dim = 0.06  →  I = 0.06 × 0.04³ / 12
    expected = 0.06 * 0.04 ** 3 / 12.0
    assert math.isclose(I, expected, rel_tol=1e-9)


def test_structural_check_to_dict():
    """StructuralCheckResult.to_dict should serialize correctly."""
    dome, params = _make_test_dome()
    load_result = loads.compute_loads(dome, params)
    result = structural_check.run_structural_check(dome, params, load_result)
    d = result.to_dict()
    assert "summary" in d
    assert "struts" in d
    assert d["summary"]["total_struts"] == len(dome.struts)
    assert isinstance(d["summary"]["all_pass"], bool)
    assert isinstance(d["summary"]["max_utilization"], float)


def test_structural_check_write_report(tmp_path):
    """write_check_report should produce a valid JSON file."""
    dome, params = _make_test_dome()
    load_result = loads.compute_loads(dome, params)
    result = structural_check.run_structural_check(dome, params, load_result)
    path = tmp_path / "structural_check.json"
    structural_check.write_check_report(result, path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert "summary" in data
    assert "struts" in data


def test_structural_check_passes_property():
    """StrutCheck.passes should be True when governing_ratio ≤ 1.0."""
    sc = structural_check.StrutCheck(
        strut_index=0, start_node=0, end_node=1, length_m=1.0,
        governing_ratio=0.5,
    )
    assert sc.passes is True
    sc2 = structural_check.StrutCheck(
        strut_index=1, start_node=0, end_node=1, length_m=1.0,
        governing_ratio=1.5,
    )
    assert sc2.passes is False


def test_structural_check_combine_loads():
    """_combine_node_loads should correctly apply combination factors."""
    from freecad_dome.loads import LoadResult, LoadCase, NodeLoad, LoadCombination
    lr = LoadResult(
        cases=[
            LoadCase(name="dead", label="Dead", node_loads=[
                NodeLoad(node_index=0, fz_kn=-10.0),
            ]),
            LoadCase(name="snow", label="Snow", node_loads=[
                NodeLoad(node_index=0, fz_kn=-5.0),
            ]),
        ],
        combinations=[
            LoadCombination(name="test", factors={"dead": 1.35, "snow": 1.50}),
        ],
    )
    combo = lr.combinations[0]
    combined = structural_check._combine_node_loads(lr, combo)
    assert 0 in combined
    fx, fy, fz = combined[0]
    assert math.isclose(fz, 1.35 * (-10.0) + 1.50 * (-5.0), rel_tol=1e-9)


def test_structural_check_pipeline_step_skipped():
    """StructuralCheckStep should skip when generate_structural_check is False."""
    params = DomeParameters(generate_structural_check=False)
    ctx = pipeline.PipelineContext(params=params, out_dir="/tmp")
    step = pipeline.StructuralCheckStep()
    assert step.should_run(ctx) is False


def test_structural_check_pipeline_step_needs_loads():
    """StructuralCheckStep should skip when load_result is missing."""
    params = DomeParameters(generate_structural_check=True, generate_loads=True)
    ctx = pipeline.PipelineContext(params=params, out_dir="/tmp")
    ctx.dome = object()  # dummy
    step = pipeline.StructuralCheckStep()
    assert step.should_run(ctx) is False  # no load_result set


def test_structural_check_pipeline_step_execute(tmp_path):
    """StructuralCheckStep should produce structural_check.json."""
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        generate_loads=True, generate_structural_check=True,
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    pipeline.TessellationStep().execute(ctx)
    pipeline.LoadCalculationStep().execute(ctx)
    pipeline.StructuralCheckStep().execute(ctx)
    assert (tmp_path / "structural_check.json").exists()
    data = json.loads((tmp_path / "structural_check.json").read_text())
    assert data["summary"]["total_struts"] > 0


def test_structural_check_material_strengths():
    """Default materials should have strength properties populated."""
    params = DomeParameters()
    for name, mat in params.materials.items():
        assert mat.density is not None, f"{name} missing density"
        assert mat.elastic_modulus is not None, f"{name} missing elastic_modulus"
        assert mat.compressive_strength_mpa is not None, f"{name} missing compressive_strength_mpa"
        assert mat.tensile_strength_mpa is not None, f"{name} missing tensile_strength_mpa"
        assert mat.bending_strength_mpa is not None, f"{name} missing bending_strength_mpa"


def test_structural_check_all_materials():
    """Structural check should work with all default materials."""
    for mat_name in ("wood", "aluminum", "steel"):
        dome, _ = _make_test_dome(DomeParameters(
            radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
            material=mat_name,
        ))
        params = DomeParameters(material=mat_name)
        load_result = loads.compute_loads(dome, params)
        result = structural_check.run_structural_check(dome, params, load_result)
        assert result.total_struts > 0
        assert result.max_utilization >= 0


# =========================================================================
# Production drawings tests
# =========================================================================

def test_production_strut_cuts():
    """compute_strut_cuts should produce a cut spec for every strut."""
    from freecad_dome.production import compute_strut_cuts
    dome, params = _make_test_dome()
    cuts = compute_strut_cuts(dome, params)
    assert len(cuts) == len(dome.struts)
    for c in cuts:
        assert c.raw_length_mm > 0
        assert c.net_length_mm > 0
        assert c.net_length_mm <= c.raw_length_mm
        assert c.stock_width_mm == params.stock_width_m * 1000
        assert c.stock_height_mm == params.stock_height_m * 1000


def test_production_strut_cuts_groups():
    """Strut cuts should be assigned group labels."""
    from freecad_dome.production import compute_strut_cuts
    dome, params = _make_test_dome()
    cuts = compute_strut_cuts(dome, params)
    groups = set(c.group for c in cuts)
    assert len(groups) >= 1
    assert all(g != "?" for g in groups)


def test_production_strut_cuts_angles():
    """Miter and bevel angles should be non-negative."""
    from freecad_dome.production import compute_strut_cuts
    dome, params = _make_test_dome()
    cuts = compute_strut_cuts(dome, params)
    for c in cuts:
        assert c.start_miter_deg >= 0
        assert c.end_miter_deg >= 0
        assert c.start_bevel_deg >= 0
        assert c.end_bevel_deg >= 0


def test_production_saw_table():
    """Saw table should have one row per unique strut type."""
    from freecad_dome.production import compute_strut_cuts, compute_saw_table
    dome, params = _make_test_dome()
    cuts = compute_strut_cuts(dome, params)
    saw = compute_saw_table(cuts)
    assert len(saw) >= 1
    total_count = sum(r.count for r in saw)
    assert total_count == len(cuts)
    # Each group should appear exactly once
    groups = [r.group for r in saw]
    assert len(groups) == len(set(groups))


def test_production_node_plates():
    """Node plates should be generated for every node with struts."""
    from freecad_dome.production import compute_node_plates
    dome, params = _make_test_dome()
    plates = compute_node_plates(dome, params)
    assert len(plates) > 0
    for p in plates:
        assert p.valence > 0
        assert p.radius_mm > 0
        assert len(p.bolt_hole_positions) == p.valence
        assert len(p.outline_points) == 32


def test_production_node_plate_bolt_diameter():
    """Bolt diameter should match parameter."""
    from freecad_dome.production import compute_node_plates
    dome, params = _make_test_dome()
    plates = compute_node_plates(dome, params)
    expected = params.node_connector_bolt_diameter_m * 1000
    for p in plates:
        assert p.bolt_diameter_mm == expected


def test_production_assembly_stages():
    """Assembly stages should cover all nodes and struts."""
    from freecad_dome.production import compute_assembly_stages
    dome, params = _make_test_dome()
    stages = compute_assembly_stages(dome, params)
    assert len(stages) >= 1
    # Stage numbers should be sequential
    for i, s in enumerate(stages, start=1):
        assert s.stage_number == i
    # All stages should have descriptions
    for s in stages:
        assert len(s.description) > 0
        assert len(s.name) > 0


def test_production_assembly_first_last_names():
    """First stage should say 'Base ring', last should say 'Apex'."""
    from freecad_dome.production import compute_assembly_stages
    dome, params = _make_test_dome()
    stages = compute_assembly_stages(dome, params)
    if len(stages) >= 2:
        assert "Base ring" in stages[0].name or "Aluskiht" in stages[0].name
        assert "Apex" in stages[-1].name or "Tipusõlm" in stages[-1].name


def test_production_pack():
    """production_for_params should return a complete pack."""
    from freecad_dome.production import production_for_params
    dome, params = _make_test_dome()
    pack = production_for_params(dome, params)
    assert pack.total_struts == len(dome.struts)
    assert pack.total_raw_length_m > 0
    assert pack.unique_types >= 1
    assert len(pack.strut_cuts) == len(dome.struts)
    assert len(pack.saw_table) >= 1
    assert len(pack.node_plates) > 0
    assert len(pack.assembly_stages) >= 1


def test_production_write_report(tmp_path):
    """write_production_report should create a JSON file."""
    from freecad_dome.production import production_for_params, write_production_report
    dome, params = _make_test_dome()
    pack = production_for_params(dome, params)
    path = tmp_path / "production_report.json"
    write_production_report(pack, params, path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert "saw_table" in data
    assert "strut_cuts" in data
    assert "assembly_stages" in data
    assert "node_plates" in data
    assert data["summary"]["total_struts"] == len(dome.struts)


def test_production_write_saw_csv(tmp_path):
    """write_saw_table_csv should create a CSV file."""
    from freecad_dome.production import compute_strut_cuts, compute_saw_table, write_saw_table_csv
    dome, params = _make_test_dome()
    cuts = compute_strut_cuts(dome, params)
    saw = compute_saw_table(cuts)
    path = tmp_path / "saw_table.csv"
    write_saw_table_csv(saw, path)
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == len(saw) + 1  # header + rows


def test_production_write_node_dxf(tmp_path):
    """write_node_plate_dxf should create a DXF file."""
    from freecad_dome.production import compute_node_plates, write_node_plate_dxf
    dome, params = _make_test_dome()
    plates = compute_node_plates(dome, params)
    plate = plates[0]
    path = tmp_path / "plate.dxf"
    write_node_plate_dxf(plate, path)
    assert path.exists()
    content = path.read_text()
    assert "PLATE_OUTLINE" in content
    assert "BOLT_HOLES" in content
    assert "EOF" in content


def test_production_write_node_svg(tmp_path):
    """write_node_plate_svg should create an SVG file."""
    from freecad_dome.production import compute_node_plates, write_node_plate_svg
    dome, params = _make_test_dome()
    plates = compute_node_plates(dome, params)
    plate = plates[0]
    path = tmp_path / "plate.svg"
    write_node_plate_svg(plate, path)
    assert path.exists()
    content = path.read_text()
    assert "<svg" in content
    assert "polygon" in content
    assert "circle" in content


def test_pipeline_production_step_skipped_when_disabled():
    """ProductionDrawingsStep should skip when generate_production=False."""
    params = DomeParameters()
    ctx = pipeline.PipelineContext(params=params)
    step = pipeline.ProductionDrawingsStep()
    assert not step.should_run(ctx)


def test_pipeline_production_step_runs_when_enabled(tmp_path):
    """ProductionDrawingsStep should run when enabled."""
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        generate_production=True,
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    pipeline.TessellationStep().execute(ctx)
    pipeline.ProductionDrawingsStep().execute(ctx)
    assert (tmp_path / "production_report.json").exists()
    assert (tmp_path / "saw_table.csv").exists()


def test_pipeline_production_step_position():
    """ProductionDrawingsStep should be between structural_check and cnc_export."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    check_idx = names.index("structural_check")
    prod_idx = names.index("production_drawings")
    cnc_idx = names.index("cnc_export")
    weather_idx = names.index("weather_protection")
    assert prod_idx == check_idx + 1
    assert cnc_idx == prod_idx + 1
    assert weather_idx == cnc_idx + 4  # TechDrawStep + AssemblyGuideStep + MultiDomeStep sit between cnc and weather


def test_production_cli_flag():
    """--production flag should set generate_production=True."""
    overrides, _ = parse_cli_overrides(["--production"])
    assert overrides.get("generate_production") is True


# =========================================================================
# CNC export tests
# =========================================================================


def test_cnc_classify_strut_types():
    """classify_strut_types should return at least one type."""
    from freecad_dome.cnc_export import classify_strut_types
    dome, params = _make_test_dome()
    types = classify_strut_types(dome, params)
    assert len(types) > 0
    # Total quantity should equal total struts.
    total_q = sum(t.quantity for t in types)
    assert total_q == len(dome.struts)


def test_cnc_strut_type_fields():
    """Each StrutType should have valid fields."""
    from freecad_dome.cnc_export import classify_strut_types
    dome, params = _make_test_dome()
    types = classify_strut_types(dome, params)
    for st in types:
        assert st.type_id != ""
        assert st.length_mm > 0
        assert st.raw_length_mm >= st.length_mm
        assert st.stock_width_mm > 0
        assert st.stock_height_mm > 0
        assert st.quantity >= 1
        assert len(st.strut_indices) == st.quantity


def test_cnc_strut_type_filename():
    """StrutType.filename should follow the naming scheme."""
    from freecad_dome.cnc_export import classify_strut_types
    dome, params = _make_test_dome()
    types = classify_strut_types(dome, params)
    for st in types:
        assert st.filename.startswith(f"Strut_{st.type_id}_L")
        assert st.filename.endswith("mm.step")


def test_cnc_write_cutting_table(tmp_path):
    """write_cutting_table should produce a valid CSV file."""
    from freecad_dome.cnc_export import classify_strut_types, write_cutting_table
    dome, params = _make_test_dome()
    types = classify_strut_types(dome, params)
    csv_path = tmp_path / "cutting_table.csv"
    write_cutting_table(types, csv_path)
    assert csv_path.exists()
    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == len(types) + 1  # header + data
    header = lines[0].split(",")
    assert "Type" in header
    assert "Qty" in header
    assert "NetLength_mm" in header
    assert "STEP_File" in header


def test_cnc_write_manifest(tmp_path):
    """write_cnc_manifest should produce a valid JSON file."""
    from freecad_dome.cnc_export import CncExportResult, write_cnc_manifest
    result = CncExportResult(total_struts=10, unique_types=3)
    path = tmp_path / "cnc_manifest.json"
    write_cnc_manifest(result, path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["total_struts"] == 10
    assert data["unique_types"] == 3


def test_cnc_export_for_dome(tmp_path):
    """cnc_export_for_dome should create cutting_table.csv and cnc_manifest.json."""
    from freecad_dome.cnc_export import cnc_export_for_dome
    dome, params = _make_test_dome()
    cnc_dir = tmp_path / "cnc_export"
    result = cnc_export_for_dome(dome, params, cnc_dir)
    assert result.total_struts == len(dome.struts)
    assert result.unique_types > 0
    assert (cnc_dir / "cutting_table.csv").exists()
    assert (cnc_dir / "cnc_manifest.json").exists()
    # Without FreeCAD doc, no STEP files should be written.
    assert result.step_files_written == 0


def test_cnc_strut_type_to_dict():
    """StrutType.to_dict should include all required fields."""
    from freecad_dome.cnc_export import classify_strut_types
    dome, params = _make_test_dome()
    types = classify_strut_types(dome, params)
    d = types[0].to_dict()
    assert "type_id" in d
    assert "quantity" in d
    assert "filename" in d
    assert "length_mm" in d
    assert "strut_indices" in d


def test_cnc_export_result_to_dict():
    """CncExportResult.to_dict should serialize properly."""
    from freecad_dome.cnc_export import cnc_export_for_dome
    dome, params = _make_test_dome()
    result = cnc_export_for_dome(dome, params, "/tmp/cnc_test")
    d = result.to_dict()
    assert "total_struts" in d
    assert "unique_types" in d
    assert "strut_types" in d
    assert len(d["strut_types"]) == result.unique_types


def test_cnc_pipeline_step_skipped():
    """CncExportStep should skip when generate_cnc_export is False."""
    params = DomeParameters(generate_cnc_export=False)
    ctx = pipeline.PipelineContext(params=params, out_dir="/tmp")
    step = pipeline.CncExportStep()
    assert step.should_run(ctx) is False


def test_cnc_pipeline_step_execute(tmp_path):
    """CncExportStep should produce cnc_export/ directory."""
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        generate_cnc_export=True,
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    pipeline.TessellationStep().execute(ctx)
    pipeline.CncExportStep().execute(ctx)
    cnc_dir = tmp_path / "cnc_export"
    assert cnc_dir.exists()
    assert (cnc_dir / "cutting_table.csv").exists()
    assert (cnc_dir / "cnc_manifest.json").exists()


def test_cnc_types_unique():
    """Each strut should belong to exactly one type."""
    from freecad_dome.cnc_export import classify_strut_types
    dome, params = _make_test_dome()
    types = classify_strut_types(dome, params)
    all_indices = []
    for t in types:
        all_indices.extend(t.strut_indices)
    # Check no duplicates.
    assert len(all_indices) == len(set(all_indices))
    # Check all struts accounted for.
    assert set(all_indices) == {s.index for s in dome.struts}


# =========================================================================
# Weather protection tests
# =========================================================================

def test_weather_gasket_profiles_catalogue():
    """GASKET_PROFILES should have at least 5 entries."""
    from freecad_dome.weather import GASKET_PROFILES
    assert len(GASKET_PROFILES) >= 5
    for key, profile in GASKET_PROFILES.items():
        assert profile.width_mm > 0
        assert profile.height_mm > 0
        assert 0 < profile.compression_ratio < 1


def test_weather_gasket_selection_default():
    """Default gasket should be EPDM D-profile."""
    from freecad_dome.weather import _select_gasket
    params = DomeParameters()
    gasket = _select_gasket(params)
    assert gasket.material == "EPDM"


def test_weather_gasket_selection_custom():
    """Custom gasket_type should select from catalogue."""
    from freecad_dome.weather import _select_gasket
    params = DomeParameters(gasket_type="silicone_12x8")
    gasket = _select_gasket(params)
    assert gasket.material == "silicone"


def test_weather_gasket_bom():
    """gasket_bom_rows should return one row per strut."""
    from freecad_dome.weather import gasket_bom_rows
    dome, params = _make_test_dome()
    rows = gasket_bom_rows(dome, params)
    assert len(rows) == len(dome.struts)
    for row in rows:
        assert row["length_m"] > 0
        assert row["width_mm"] > 0


def test_weather_pack_total_gasket():
    """Total gasket length should be positive."""
    from freecad_dome.weather import weather_for_params
    dome, params = _make_test_dome()
    pack = weather_for_params(dome, params)
    assert pack.total_gasket_length_m > 0
    assert pack.estimated_gasket_cost_eur > 0


def test_weather_pack_gasket_per_strut():
    """Gasket per strut should cover all struts."""
    from freecad_dome.weather import weather_for_params
    dome, params = _make_test_dome()
    pack = weather_for_params(dome, params)
    assert len(pack.gasket_per_strut_m) == len(dome.struts)


def test_weather_pack_gasket_per_panel():
    """Gasket per panel should cover all panels."""
    from freecad_dome.weather import weather_for_params
    dome, params = _make_test_dome()
    pack = weather_for_params(dome, params)
    assert len(pack.gasket_per_panel_m) == len(dome.panels)


def test_weather_drainage_only_polycarbonate():
    """Drainage specs should only be generated for polycarbonate covering."""
    from freecad_dome.weather import weather_for_params
    dome, params = _make_test_dome()
    # Default is glass — no drainage
    pack = weather_for_params(dome, params)
    assert len(pack.drainage_specs) == 0
    assert pack.total_drain_holes == 0


def test_weather_drainage_polycarbonate():
    """Drainage specs should be generated for polycarbonate panels."""
    from freecad_dome.weather import weather_for_params
    dome, _ = _make_test_dome()
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        covering_type="pc_multiwall_10",
    )
    pack = weather_for_params(dome, params)
    assert len(pack.drainage_specs) == len(dome.panels)
    assert pack.total_drain_holes > 0
    for d in pack.drainage_specs:
        assert d.drain_hole_count >= 1
        assert d.drain_hole_diameter_mm > 0


def test_weather_eave_details():
    """Eave details should be generated for belt nodes."""
    from freecad_dome.weather import weather_for_params
    dome, params = _make_test_dome()
    pack = weather_for_params(dome, params)
    assert len(pack.eave_details) > 0
    for e in pack.eave_details:
        assert e.drip_edge_length_mm > 0
        assert e.overhang_mm > 0


def test_weather_eave_azimuth_sorted():
    """Eave details should be sorted by azimuth."""
    from freecad_dome.weather import weather_for_params
    dome, params = _make_test_dome()
    pack = weather_for_params(dome, params)
    if len(pack.eave_details) >= 2:
        azimuths = [e.azimuth_deg for e in pack.eave_details]
        assert azimuths == sorted(azimuths)


def test_weather_write_report(tmp_path):
    """write_weather_report should create a JSON file."""
    from freecad_dome.weather import weather_for_params, write_weather_report
    dome, params = _make_test_dome()
    pack = weather_for_params(dome, params)
    path = tmp_path / "weather_report.json"
    write_weather_report(pack, params, path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert "gasket" in data
    assert "drainage" in data
    assert "eave_details" in data
    assert "summary" in data


def test_pipeline_weather_step_skipped_when_disabled():
    """WeatherProtectionStep should skip when generate_weather=False."""
    params = DomeParameters()
    ctx = pipeline.PipelineContext(params=params)
    step = pipeline.WeatherProtectionStep()
    assert not step.should_run(ctx)


def test_pipeline_weather_step_runs_when_enabled(tmp_path):
    """WeatherProtectionStep should run when enabled."""
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        generate_weather=True,
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    pipeline.TessellationStep().execute(ctx)
    pipeline.WeatherProtectionStep().execute(ctx)
    assert (tmp_path / "weather_report.json").exists()


def test_pipeline_weather_step_position():
    """WeatherProtectionStep should be between cnc_export and cost_estimation."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    cnc_idx = names.index("cnc_export")
    weather_idx = names.index("weather_protection")
    cost_idx = names.index("cost_estimation")
    assert weather_idx == cnc_idx + 4  # TechDrawStep + AssemblyGuideStep + MultiDomeStep sit between cnc and weather
    assert weather_idx == cost_idx - 1


def test_weather_cli_flags():
    """--weather and --gasket-type flags should work."""
    overrides, _ = parse_cli_overrides(["--weather", "--gasket-type", "silicone_12x8"])
    assert overrides.get("generate_weather") is True
    assert overrides.get("gasket_type") == "silicone_12x8"


def test_validate_gasket_type():
    """Validation should reject unknown gasket types."""
    with pytest.raises(ValueError, match="(?i)gasket"):
        DomeParameters(gasket_type="unknown_gasket").validate()


# =========================================================================
# Cost estimation tests
# =========================================================================

def test_cost_default_catalogue():
    """DEFAULT_PRICE_CATALOGUE should have entries."""
    from freecad_dome.costing import DEFAULT_PRICE_CATALOGUE
    assert len(DEFAULT_PRICE_CATALOGUE) >= 10
    for key, entry in DEFAULT_PRICE_CATALOGUE.items():
        assert entry.price_eur > 0
        assert entry.unit in ("m", "m2", "pcs", "kg")


def test_cost_estimate_bom_items():
    """cost_estimate_for_params should produce BOM items."""
    from freecad_dome.costing import cost_estimate_for_params
    dome, params = _make_test_dome()
    est = cost_estimate_for_params(dome, params)
    assert len(est.bom) >= 4  # timber + covering + hardware + gasket
    categories = set(item.category for item in est.bom)
    assert "timber" in categories
    assert "covering" in categories
    assert "hardware" in categories


def test_cost_estimate_positive_totals():
    """All cost totals should be positive."""
    from freecad_dome.costing import cost_estimate_for_params
    dome, params = _make_test_dome()
    est = cost_estimate_for_params(dome, params)
    assert est.total_eur > 0
    assert est.total_material_eur > 0
    assert est.total_hardware_eur > 0


def test_cost_estimate_timber_total():
    """Timber total should account for waste."""
    from freecad_dome.costing import cost_estimate_for_params
    dome, params = _make_test_dome()
    est = cost_estimate_for_params(dome, params)
    raw_timber = sum(s.length for s in dome.struts)
    # With 10% waste
    assert est.timber_total_m > raw_timber
    assert est.timber_total_m < raw_timber * 1.15


def test_cost_estimate_covering_sheets():
    """Covering sheets should match panel count."""
    from freecad_dome.costing import cost_estimate_for_params
    dome, params = _make_test_dome()
    est = cost_estimate_for_params(dome, params)
    assert len(est.covering_sheets) == len(dome.panels)
    for cs in est.covering_sheets:
        assert cs.area_m2 > 0
        assert cs.sheet_area_m2 >= cs.area_m2
        assert cs.waste_pct > 0


def test_cost_estimate_bolt_count():
    """Bolt count should be 2 per strut."""
    from freecad_dome.costing import cost_estimate_for_params
    dome, params = _make_test_dome()
    est = cost_estimate_for_params(dome, params)
    assert est.bolt_count == len(dome.struts) * 2


def test_cost_estimate_waste_coefficient():
    """Waste coefficient should be >= 1.0."""
    from freecad_dome.costing import cost_estimate_for_params
    dome, params = _make_test_dome()
    est = cost_estimate_for_params(dome, params)
    assert est.waste_coefficient >= 1.0


def test_cost_write_report(tmp_path):
    """write_cost_report should create a JSON file."""
    from freecad_dome.costing import cost_estimate_for_params, write_cost_report
    dome, params = _make_test_dome()
    est = cost_estimate_for_params(dome, params)
    path = tmp_path / "cost_report.json"
    write_cost_report(est, params, path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert "summary" in data
    assert "bom" in data
    assert data["summary"]["total_eur"] > 0


def test_cost_write_csv(tmp_path):
    """write_cost_csv should create a CSV file."""
    from freecad_dome.costing import cost_estimate_for_params, write_cost_csv
    dome, params = _make_test_dome()
    est = cost_estimate_for_params(dome, params)
    path = tmp_path / "cost_bom.csv"
    write_cost_csv(est, path)
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    # header + items + empty + 3 total rows
    assert len(lines) >= len(est.bom) + 1


def test_pipeline_cost_step_skipped_when_disabled():
    """CostEstimationStep should skip when generate_costing=False."""
    params = DomeParameters()
    ctx = pipeline.PipelineContext(params=params)
    step = pipeline.CostEstimationStep()
    assert not step.should_run(ctx)


def test_pipeline_cost_step_runs_when_enabled(tmp_path):
    """CostEstimationStep should run when enabled."""
    params = DomeParameters(
        radius_m=3.0, frequency=4, hemisphere_ratio=0.625,
        generate_costing=True,
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    pipeline.TessellationStep().execute(ctx)
    pipeline.CostEstimationStep().execute(ctx)
    assert (tmp_path / "cost_report.json").exists()
    assert (tmp_path / "cost_bom.csv").exists()


def test_pipeline_cost_step_position():
    """CostEstimationStep should be between weather_protection and spreadsheets."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    weather_idx = names.index("weather_protection")
    cost_idx = names.index("cost_estimation")
    ss_idx = names.index("spreadsheets")
    assert cost_idx == weather_idx + 1
    assert cost_idx == ss_idx - 1


def test_cost_cli_flag():
    """--costing flag should set generate_costing=True."""
    overrides, _ = parse_cli_overrides(["--costing"])
    assert overrides.get("generate_costing") is True


# =========================================================================
# II1 — CostingConfig expansion tests
# =========================================================================

def test_costing_config_defaults():
    """CostingConfig should have sensible defaults for all new fields."""
    from freecad_dome.parameters import CostingConfig
    cc = CostingConfig()
    assert cc.currency == "EUR"
    assert cc.waste_timber_pct == 10.0
    assert cc.waste_covering_pct == 8.0
    assert cc.overhead_pct == 0.0
    assert cc.timber_price_per_m == 0.0
    assert cc.labor_install_eur_h == 0.0
    assert cc.price_catalogue_path == ""


def test_costing_config_via_dome_params():
    """CostingConfig fields should be accessible via flat DomeParameters attrs."""
    params = DomeParameters(
        currency="USD",
        waste_timber_pct=15.0,
        overhead_pct=5.0,
        timber_price_per_m=6.50,
        labor_install_eur_h=35.0,
        estimated_install_hours=80.0,
    )
    assert params.currency == "USD"
    assert params.costing.currency == "USD"
    assert params.waste_timber_pct == 15.0
    assert params.overhead_pct == 5.0
    assert params.timber_price_per_m == 6.50
    assert params.labor_install_eur_h == 35.0
    assert params.estimated_install_hours == 80.0


def test_cost_estimate_custom_waste():
    """Custom waste percentages should be honored."""
    from freecad_dome.costing import cost_estimate_for_params
    dome, params = _make_test_dome()
    params.waste_timber_pct = 20.0

    est = cost_estimate_for_params(dome, params)
    raw_timber = sum(s.length for s in dome.struts)
    expected = raw_timber * 1.20
    assert abs(est.timber_total_m - expected) < expected * 0.01


def test_cost_estimate_override_timber_price():
    """Explicit timber_price_per_m should override catalogue default."""
    from freecad_dome.costing import cost_estimate_for_params
    dome, params = _make_test_dome()
    params.timber_price_per_m = 99.0

    est = cost_estimate_for_params(dome, params)
    timber_item = [b for b in est.bom if b.category == "timber"][0]
    assert timber_item.unit_price_eur == 99.0


def test_cost_estimate_labour_included():
    """Labour costs should appear when rates and hours are set."""
    from freecad_dome.costing import cost_estimate_for_params
    dome, params = _make_test_dome()
    params.labor_install_eur_h = 40.0
    params.estimated_install_hours = 100.0
    params.labor_cnc_eur_h = 60.0
    params.estimated_cnc_hours = 20.0

    est = cost_estimate_for_params(dome, params)
    assert est.total_labour_eur == 40.0 * 100.0 + 60.0 * 20.0
    labour_items = [b for b in est.bom if b.category == "labour"]
    assert len(labour_items) == 2


def test_cost_estimate_overhead():
    """Overhead should add markup to the total."""
    from freecad_dome.costing import cost_estimate_for_params
    dome, params = _make_test_dome()
    params.overhead_pct = 10.0

    est = cost_estimate_for_params(dome, params)
    subtotal = sum(b.total_eur for b in est.bom)
    expected_overhead = subtotal * 0.10
    assert abs(est.total_overhead_eur - expected_overhead) < 0.02
    assert abs(est.total_eur - (subtotal + expected_overhead)) < 0.02


def test_cost_estimate_currency_usd():
    """USD currency should apply exchange rate."""
    from freecad_dome.costing import cost_estimate_for_params, _CURRENCY_RATES
    dome, params = _make_test_dome()
    params.currency = "USD"

    est = cost_estimate_for_params(dome, params)
    assert est.currency == "USD"
    # Prices scaled by USD rate — total should differ from EUR
    est_eur = cost_estimate_for_params(dome, DomeParameters())
    ratio = _CURRENCY_RATES["USD"]
    # Timber items should reflect the rate
    t_usd = [b for b in est.bom if b.category == "timber"]
    t_eur = [b for b in est_eur.bom if b.category == "timber"]
    if t_usd and t_eur:
        assert abs(t_usd[0].unit_price_eur / t_eur[0].unit_price_eur - ratio) < 0.01


def test_cost_supplier_field():
    """PriceCatalogueEntry and BomItem should carry supplier info."""
    from freecad_dome.costing import (
        PriceCatalogueEntry, cost_estimate_for_params
    )
    custom_cat = {
        "timber_45x70_pine": PriceCatalogueEntry(
            name="Pine 45x70", unit="m", price_eur=5.0,
            category="timber", supplier="TestSupplier OÜ"
        ),
    }
    dome, params = _make_test_dome()
    # Force stock size to match the 45x70 entry
    params.stock_width_m = 0.045
    params.stock_height_m = 0.070
    est = cost_estimate_for_params(dome, params, catalogue=custom_cat)
    timber_items = [b for b in est.bom if b.category == "timber"]
    assert len(timber_items) >= 1
    assert timber_items[0].supplier == "TestSupplier OÜ"


def test_load_price_catalogue_from_json(tmp_path):
    """load_price_catalogue should merge external JSON onto defaults."""
    from freecad_dome.costing import load_price_catalogue, DEFAULT_PRICE_CATALOGUE
    cat_json = {
        "timber_45x70_pine": {
            "name": "Updated Pine",
            "unit": "m",
            "price_eur": 7.77,
            "category": "timber",
            "supplier": "Ehituspuud AS"
        },
        "custom_part_99": {
            "name": "Special bracket",
            "unit": "pcs",
            "price_eur": 2.50,
            "category": "hardware",
            "supplier": "BoltCo"
        },
    }
    p = tmp_path / "prices.json"
    p.write_text(json.dumps(cat_json))

    result = load_price_catalogue(str(p))
    # Overridden entry
    assert result["timber_45x70_pine"].price_eur == 7.77
    assert result["timber_45x70_pine"].supplier == "Ehituspuud AS"
    # New entry
    assert "custom_part_99" in result
    assert result["custom_part_99"].price_eur == 2.50
    # Original entries still present
    assert "glass_4mm" in result


def test_load_price_catalogue_missing_file(tmp_path):
    """Missing catalogue file should fall back to defaults."""
    from freecad_dome.costing import load_price_catalogue, DEFAULT_PRICE_CATALOGUE
    result = load_price_catalogue(str(tmp_path / "nonexistent.json"))
    assert len(result) == len(DEFAULT_PRICE_CATALOGUE)


def test_cost_estimate_with_external_catalogue(tmp_path):
    """Pipeline should use external catalogue when configured."""
    from freecad_dome.costing import cost_estimate_for_params
    cat_json = {
        "timber_45x70_pine": {
            "name": "Cheap Pine", "unit": "m",
            "price_eur": 1.00, "category": "timber"
        },
    }
    p = tmp_path / "prices.json"
    p.write_text(json.dumps(cat_json))

    dome, params = _make_test_dome()
    params.stock_width_m = 0.045
    params.stock_height_m = 0.070
    params.price_catalogue_path = str(p)

    est = cost_estimate_for_params(dome, params)
    timber_items = [b for b in est.bom if b.category == "timber"]
    assert len(timber_items) >= 1
    assert timber_items[0].unit_price_eur == 1.0


def test_cost_report_includes_new_fields(tmp_path):
    """JSON report should contain new II1 fields."""
    from freecad_dome.costing import cost_estimate_for_params, write_cost_report
    dome, params = _make_test_dome()
    params.overhead_pct = 5.0
    est = cost_estimate_for_params(dome, params)
    path = tmp_path / "cost_report.json"
    write_cost_report(est, params, path)
    data = json.loads(path.read_text())
    assert "currency" in data["summary"]
    assert "total_labour_eur" in data["summary"]
    assert "total_overhead_eur" in data["summary"]
    # BOM items should have supplier field
    for item in data["bom"]:
        assert "supplier" in item


def test_cost_csv_includes_supplier(tmp_path):
    """CSV should include Supplier column."""
    from freecad_dome.costing import cost_estimate_for_params, write_cost_csv
    dome, params = _make_test_dome()
    est = cost_estimate_for_params(dome, params)
    path = tmp_path / "cost_bom.csv"
    write_cost_csv(est, path)
    header = path.read_text().split("\n")[0]
    assert "Supplier" in header


# =========================================================================
# II2 — Hidden parameters in GUI tests
# =========================================================================

def test_advanced_strut_params_via_dome_params():
    """Advanced strut params should be settable and readable via DomeParameters."""
    params = DomeParameters(
        strut_profile="round",
        cap_blend_mode="chamfer",
        bevel_fillet_radius_m=0.005,
        min_wedge_angle_deg=20.0,
        cap_length_factor=3.0,
        max_cap_ratio=0.35,
        generate_belt_cap=True,
    )
    assert params.strut_profile == "round"
    assert params.cap_blend_mode == "chamfer"
    assert params.bevel_fillet_radius_m == 0.005
    assert params.min_wedge_angle_deg == 20.0
    assert params.cap_length_factor == 3.0
    assert params.max_cap_ratio == 0.35
    assert params.generate_belt_cap is True


def test_advanced_strut_profile_validation():
    """Invalid strut_profile should fail validation."""
    params = DomeParameters(strut_profile="hexagonal")
    with pytest.raises(ValueError, match="strut_profile"):
        params.validate()


def test_advanced_cap_blend_mode_validation():
    """Invalid cap_blend_mode should fail validation."""
    params = DomeParameters(cap_blend_mode="smooth")
    with pytest.raises(ValueError, match="cap_blend_mode"):
        params.validate()


def test_advanced_params_in_to_dict():
    """Advanced params should appear in to_dict output."""
    params = DomeParameters(
        strut_profile="trapezoidal",
        cap_blend_mode="fillet",
        bevel_fillet_radius_m=0.003,
    )
    d = params.to_dict()
    assert d["strut_profile"] == "trapezoidal"
    assert d["cap_blend_mode"] == "fillet"
    assert d["bevel_fillet_radius_m"] == 0.003


def test_advanced_params_roundtrip():
    """Advanced params should survive to_dict → from_dict roundtrip."""
    params = DomeParameters(
        strut_profile="round",
        cap_blend_mode="chamfer",
        bevel_fillet_radius_m=0.008,
        min_wedge_angle_deg=25.0,
        cap_length_factor=4.0,
        max_cap_ratio=0.40,
        generate_belt_cap=True,
    )
    d = params.to_dict()
    params2 = DomeParameters.from_dict(d)
    assert params2.strut_profile == "round"
    assert params2.cap_blend_mode == "chamfer"
    assert params2.bevel_fillet_radius_m == 0.008
    assert params2.min_wedge_angle_deg == 25.0
    assert params2.cap_length_factor == 4.0
    assert params2.max_cap_ratio == 0.40
    assert params2.generate_belt_cap is True


def test_preset_save_load_roundtrip(tmp_path):
    """Simulated preset save and load should preserve parameters."""
    params = DomeParameters(
        radius_m=4.0,
        frequency=6,
        strut_profile="trapezoidal",
        cap_blend_mode="fillet",
        overhead_pct=12.0,
        currency="GBP",
    )
    d = params.to_dict()
    d.pop("materials", None)
    preset_path = tmp_path / "my_preset.json"
    preset_path.write_text(json.dumps(d, indent=2, default=str))

    raw = json.loads(preset_path.read_text())
    params2 = DomeParameters.from_dict(raw)
    assert params2.radius_m == 4.0
    assert params2.frequency == 6
    assert params2.strut_profile == "trapezoidal"
    assert params2.cap_blend_mode == "fillet"
    assert params2.overhead_pct == 12.0
    assert params2.currency == "GBP"


def test_generate_belt_cap_default_false():
    """generate_belt_cap should default to False."""
    params = DomeParameters()
    assert params.generate_belt_cap is False


def test_gui_dialog_importable():
    """gui_dialog module should import without Qt."""
    import freecad_dome.gui_dialog as gd
    assert hasattr(gd, "prompt_parameters_dialog")


# ---------------------------------------------------------------------------
# P2 #10 — Configuration hierarchy tests
# ---------------------------------------------------------------------------


def test_sub_config_classes_exist():
    """All 7 sub-config dataclasses should be importable."""
    for cls in (
        GeometryConfig, StructureConfig, CoveringConfig,
        OpeningsConfig, FoundationConfig, ExportConfig, CostingConfig,
    ):
        assert hasattr(cls, "__dataclass_fields__")


def test_flat_attribute_access():
    """Flat attribute access should delegate to sub-configs."""
    params = DomeParameters(radius_m=5.0, frequency=3)
    assert params.radius_m == 5.0
    assert params.frequency == 3
    # Also accessible via sub-config
    assert params.geometry.radius_m == 5.0
    assert params.geometry.frequency == 3


def test_flat_attribute_write():
    """Writing a flat attribute should update the sub-config."""
    params = DomeParameters()
    params.radius_m = 7.0
    assert params.radius_m == 7.0
    assert params.geometry.radius_m == 7.0


def test_sub_config_direct_construction():
    """DomeParameters should accept sub-config instances directly."""
    geo = GeometryConfig(radius_m=4.0, frequency=2)
    params = DomeParameters(geometry=geo)
    assert params.radius_m == 4.0
    assert params.frequency == 2
    assert params.geometry is geo


def test_sub_config_mixed_construction_error():
    """Mixing sub-config instance and flat kwargs for the same group should raise."""
    geo = GeometryConfig(radius_m=4.0)
    with pytest.raises(TypeError, match="Cannot mix"):
        DomeParameters(geometry=geo, frequency=3)


def test_unknown_kwarg_raises():
    """Unknown keyword argument should raise TypeError."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        DomeParameters(unknown_param=42)


def test_to_dict_flat():
    """to_dict() should return a flat dictionary with all fields."""
    params = DomeParameters(radius_m=5.0, stock_width_m=0.06)
    d = params.to_dict()
    assert d["radius_m"] == 5.0
    assert d["stock_width_m"] == 0.06
    # Should not have sub-config keys at top level
    assert "geometry" not in d
    assert "structure" not in d


def test_to_nested_dict():
    """to_nested_dict() should return hierarchically grouped dict."""
    params = DomeParameters(radius_m=5.0, stock_width_m=0.06)
    d = params.to_nested_dict()
    assert "geometry" in d
    assert "structure" in d
    assert d["geometry"]["radius_m"] == 5.0
    assert d["structure"]["stock_width_m"] == 0.06


def test_from_dict_flat():
    """from_dict should accept flat format (backward compat)."""
    params = DomeParameters.from_dict({"radius_m": 4.0, "frequency": 2})
    assert params.radius_m == 4.0
    assert params.frequency == 2


def test_from_dict_nested():
    """from_dict should accept nested format (new hierarchical JSON)."""
    params = DomeParameters.from_dict({
        "geometry": {"radius_m": 6.0, "frequency": 5},
        "structure": {"stock_width_m": 0.08},
    })
    assert params.radius_m == 6.0
    assert params.frequency == 5
    assert params.stock_width_m == 0.08
    # Defaults should be preserved for unspecified fields
    assert params.hemisphere_ratio == 0.625


def test_from_dict_mixed_flat_nested():
    """from_dict should handle mix of nested and flat keys."""
    params = DomeParameters.from_dict({
        "geometry": {"radius_m": 4.0},
        "stock_width_m": 0.07,
    })
    assert params.radius_m == 4.0
    assert params.stock_width_m == 0.07


def test_roundtrip_to_dict_from_dict():
    """to_dict → from_dict roundtrip should preserve all values."""
    original = DomeParameters(
        radius_m=4.5, frequency=3, stock_width_m=0.06,
        gasket_type="silicone_12x8",
    )
    rebuilt = DomeParameters.from_dict(original.to_dict())
    assert math.isclose(rebuilt.radius_m, original.radius_m)
    assert rebuilt.frequency == original.frequency
    assert math.isclose(rebuilt.stock_width_m, original.stock_width_m)
    assert rebuilt.gasket_type == original.gasket_type


def test_roundtrip_nested_dict():
    """to_nested_dict → from_dict roundtrip should work."""
    original = DomeParameters(radius_m=3.5, foundation_type="point")
    rebuilt = DomeParameters.from_dict(original.to_nested_dict())
    assert math.isclose(rebuilt.radius_m, 3.5)
    assert rebuilt.foundation_type == "point"


def test_equality():
    """Two DomeParameters with same values should be equal."""
    a = DomeParameters(radius_m=5.0)
    b = DomeParameters(radius_m=5.0)
    assert a == b
    b.radius_m = 6.0
    assert a != b


def test_repr_contains_fields():
    """repr should contain field name=value pairs."""
    params = DomeParameters(radius_m=5.0)
    r = repr(params)
    assert "radius_m=5.0" in r
    assert "DomeParameters(" in r


def test_all_fields_in_field_map():
    """Every sub-config field should appear in _FIELD_MAP."""
    from freecad_dome.parameters import _FIELD_MAP, _SUB_CONFIGS, dc_fields
    for _, cfg_cls in _SUB_CONFIGS:
        for f in dc_fields(cfg_cls):
            assert f.name in _FIELD_MAP, f"Field {f.name} missing from _FIELD_MAP"


def test_panels_only_implies_no_struts():
    """panels_only=True should set generate_struts=False via __init__."""
    params = DomeParameters(panels_only=True)
    assert params.generate_struts is False
    assert params.structure.generate_struts is False


def test_sub_config_field_count():
    """Verify total field count across all sub-configs."""
    from freecad_dome.parameters import _SUB_CONFIGS, dc_fields
    total = sum(len(dc_fields(cls)) for _, cls in _SUB_CONFIGS)
    assert total == 145


# ---------------------------------------------------------------------------
# I3: TechDraw pipeline step
# ---------------------------------------------------------------------------


def test_techdraw_params_defaults():
    """ExportConfig TechDraw params should have correct defaults."""
    p = DomeParameters()
    assert p.generate_techdraw is False
    assert p.techdraw_page_format == "A3"
    assert math.isclose(p.techdraw_scale, 0.02)
    assert p.techdraw_views == "all"
    assert p.techdraw_project_name == ""
    assert p.techdraw_version == ""


def test_techdraw_params_via_dome_params():
    """TechDraw fields should be settable via DomeParameters flat init."""
    p = DomeParameters(
        generate_techdraw=True,
        techdraw_page_format="A2",
        techdraw_scale=0.01,
        techdraw_views="nodes",
        techdraw_project_name="Test Project",
        techdraw_version="v1.0",
    )
    assert p.generate_techdraw is True
    assert p.techdraw_page_format == "A2"
    assert math.isclose(p.techdraw_scale, 0.01)
    assert p.techdraw_views == "nodes"
    assert p.techdraw_project_name == "Test Project"
    assert p.techdraw_version == "v1.0"


def test_techdraw_params_roundtrip():
    """TechDraw fields should survive to_dict / from_dict."""
    p = DomeParameters(
        generate_techdraw=True,
        techdraw_page_format="A4",
        techdraw_scale=0.05,
        techdraw_views="overview",
        techdraw_project_name="Round Trip",
        techdraw_version="v2",
    )
    d = p.to_dict()
    p2 = DomeParameters.from_dict(d)
    assert p2.generate_techdraw is True
    assert p2.techdraw_page_format == "A4"
    assert math.isclose(p2.techdraw_scale, 0.05)
    assert p2.techdraw_views == "overview"
    assert p2.techdraw_project_name == "Round Trip"
    assert p2.techdraw_version == "v2"


def test_techdraw_result_dataclass():
    """TechDrawResult should be importable and have correct defaults."""
    from freecad_dome.techdraw import TechDrawResult
    r = TechDrawResult()
    assert r.pages_created == 0
    assert r.pdf_exported == 0
    assert r.dxf_exported == 0
    assert r.page_format == "A3"
    assert r.views_mode == "all"
    assert r.manifest_path == ""


def test_techdraw_generate_headless(tmp_path):
    """generate_techdraw_for_dome should produce manifest in headless mode."""
    from freecad_dome.techdraw import generate_techdraw_for_dome
    p = DomeParameters(
        generate_techdraw=True,
        techdraw_page_format="A3",
        techdraw_scale=0.02,
        techdraw_views="all",
        techdraw_project_name="Headless Test",
        techdraw_version="v0.1",
    )
    result = generate_techdraw_for_dome(p, tmp_path)
    assert result.page_format == "A3"
    assert result.views_mode == "all"
    assert result.manifest_path != ""
    # manifest file should exist
    import json
    from pathlib import Path as _Path
    manifest = json.loads(_Path(result.manifest_path).read_text())
    assert manifest["page_format"] == "A3"
    assert manifest["scale_text"] == "1:50"
    assert manifest["project_name"] == "Headless Test"
    assert manifest["version"] == "v0.1"
    assert len(manifest["sheets"]) > 0


def test_techdraw_plan_sheets_all():
    """_plan_sheets('all') should return overview + parts + nodes."""
    from freecad_dome.techdraw import _plan_sheets
    sheets = _plan_sheets("all")
    names = [s["name"] for s in sheets]
    assert "TD_Overview" in names
    assert "TD_Section" in names
    assert "TD_Struts" in names
    assert "TD_PanelFrames" in names
    assert "TD_NodeDetails" in names


def test_techdraw_plan_sheets_overview_only():
    """_plan_sheets('overview') should return only overview sheets."""
    from freecad_dome.techdraw import _plan_sheets
    sheets = _plan_sheets("overview")
    names = [s["name"] for s in sheets]
    assert "TD_Overview" in names
    assert "TD_Struts" not in names
    assert "TD_NodeDetails" not in names


def test_techdraw_plan_sheets_nodes_only():
    """_plan_sheets('nodes') should return only node detail sheets."""
    from freecad_dome.techdraw import _plan_sheets
    sheets = _plan_sheets("nodes")
    names = [s["name"] for s in sheets]
    assert "TD_NodeDetails" in names
    assert "TD_Overview" not in names
    assert "TD_Struts" not in names


def test_techdraw_page_formats():
    """_PAGE_FORMATS should list A2, A3, A4 sizes."""
    from freecad_dome.techdraw import _PAGE_FORMATS
    assert "A2" in _PAGE_FORMATS
    assert "A3" in _PAGE_FORMATS
    assert "A4" in _PAGE_FORMATS
    assert _PAGE_FORMATS["A3"] == (420.0, 297.0)


def test_techdraw_template_path_for_format():
    """template_path_for_format should return a path string."""
    from freecad_dome.techdraw import template_path_for_format
    path = template_path_for_format("A2")
    assert isinstance(path, str)
    assert len(path) > 0


def test_techdraw_pipeline_step_position():
    """TechDrawStep should be after CncExportStep and before AssemblyGuideStep."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    assert "techdraw" in names
    cnc_idx = names.index("cnc_export")
    td_idx = names.index("techdraw")
    ag_idx = names.index("assembly_guide")
    assert td_idx == cnc_idx + 1
    assert td_idx == ag_idx - 1


def test_techdraw_step_skipped_when_disabled(tmp_path):
    """TechDrawStep.should_run should return False when disabled."""
    step = pipeline.TechDrawStep()
    ctx = pipeline.PipelineContext(
        params=DomeParameters(generate_techdraw=False),
        out_dir=tmp_path,
    )
    assert step.should_run(ctx) is False


def test_techdraw_step_enabled_when_active(tmp_path):
    """TechDrawStep.should_run should return True when enabled."""
    step = pipeline.TechDrawStep()
    ctx = pipeline.PipelineContext(
        params=DomeParameters(generate_techdraw=True),
        out_dir=tmp_path,
    )
    assert step.should_run(ctx) is True


# ---------------------------------------------------------------------------
# I4: Assembly guide (montaažijuhised)
# ---------------------------------------------------------------------------


def test_assembly_params_defaults():
    """ExportConfig assembly params should have correct defaults."""
    p = DomeParameters()
    assert p.generate_assembly_guide is False
    assert math.isclose(p.assembly_time_per_strut_min, 15.0)
    assert math.isclose(p.assembly_time_per_node_min, 10.0)
    assert math.isclose(p.assembly_time_per_panel_min, 20.0)
    assert p.assembly_workers == 2


def test_assembly_params_via_dome_params():
    """Assembly fields should be settable via DomeParameters flat init."""
    p = DomeParameters(
        generate_assembly_guide=True,
        assembly_time_per_strut_min=20.0,
        assembly_time_per_node_min=12.0,
        assembly_time_per_panel_min=25.0,
        assembly_workers=4,
    )
    assert p.generate_assembly_guide is True
    assert math.isclose(p.assembly_time_per_strut_min, 20.0)
    assert math.isclose(p.assembly_time_per_node_min, 12.0)
    assert math.isclose(p.assembly_time_per_panel_min, 25.0)
    assert p.assembly_workers == 4


def test_assembly_params_roundtrip():
    """Assembly fields should survive to_dict / from_dict."""
    p = DomeParameters(
        generate_assembly_guide=True,
        assembly_time_per_strut_min=18.0,
        assembly_workers=3,
    )
    d = p.to_dict()
    p2 = DomeParameters.from_dict(d)
    assert p2.generate_assembly_guide is True
    assert math.isclose(p2.assembly_time_per_strut_min, 18.0)
    assert p2.assembly_workers == 3


def test_assembly_guide_for_dome():
    """assembly_guide_for_dome should produce stages with BOM and time."""
    from freecad_dome.assembly import assembly_guide_for_dome
    params = DomeParameters(radius_m=5.0, frequency=2)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    guide = assembly_guide_for_dome(dome, params)
    assert guide.total_stages > 0
    assert guide.total_estimated_minutes > 0
    assert guide.total_estimated_hours > 0
    assert guide.workers == 2
    # Each stage should have BOM entries
    for stage in guide.stages:
        assert stage.stage_number > 0
        assert isinstance(stage.bom, list)
        assert stage.estimated_minutes >= 0
        assert stage.cumulative_minutes >= stage.estimated_minutes


def test_assembly_guide_stages_bottom_up():
    """Assembly stages should be ordered from lowest to highest Z."""
    from freecad_dome.assembly import assembly_guide_for_dome
    params = DomeParameters(radius_m=5.0, frequency=2)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    guide = assembly_guide_for_dome(dome, params)
    # z_min of each stage should be non-decreasing
    z_mins = [s.z_min for s in guide.stages]
    for i in range(1, len(z_mins)):
        assert z_mins[i] >= z_mins[i - 1] - 0.01  # small tolerance


def test_assembly_guide_bom_has_struts():
    """At least one stage should have strut BOM entries."""
    from freecad_dome.assembly import assembly_guide_for_dome
    params = DomeParameters(radius_m=5.0, frequency=2)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    guide = assembly_guide_for_dome(dome, params)
    strut_bom_found = False
    for stage in guide.stages:
        for b in stage.bom:
            if b.category == "strut":
                strut_bom_found = True
                break
    assert strut_bom_found


def test_assembly_guide_panels_assigned():
    """Panels should be assigned to stages (no duplicates)."""
    from freecad_dome.assembly import assembly_guide_for_dome
    params = DomeParameters(radius_m=5.0, frequency=2)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    guide = assembly_guide_for_dome(dome, params)
    all_panels = []
    for stage in guide.stages:
        all_panels.extend(stage.panel_indices)
    # No duplicates
    assert len(all_panels) == len(set(all_panels))


def test_assembly_time_varies_with_params():
    """Different time params should give different totals."""
    from freecad_dome.assembly import assembly_guide_for_dome
    params_fast = DomeParameters(radius_m=5.0, frequency=2,
                                 assembly_time_per_strut_min=5.0,
                                 assembly_time_per_node_min=3.0,
                                 assembly_time_per_panel_min=5.0)
    params_slow = DomeParameters(radius_m=5.0, frequency=2,
                                 assembly_time_per_strut_min=30.0,
                                 assembly_time_per_node_min=20.0,
                                 assembly_time_per_panel_min=40.0)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(5.0), params_fast
    )
    g_fast = assembly_guide_for_dome(dome, params_fast)
    g_slow = assembly_guide_for_dome(dome, params_slow)
    assert g_slow.total_estimated_minutes > g_fast.total_estimated_minutes


def test_assembly_report_json(tmp_path):
    """write_assembly_report should produce valid JSON."""
    from freecad_dome.assembly import assembly_guide_for_dome, write_assembly_report
    import json
    params = DomeParameters(radius_m=5.0, frequency=2)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    guide = assembly_guide_for_dome(dome, params)
    report_path = tmp_path / "assembly" / "assembly_guide.json"
    write_assembly_report(guide, params, report_path)
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert "summary" in data
    assert "stages" in data
    assert data["summary"]["total_stages"] == guide.total_stages
    assert data["summary"]["workers"] == 2


def test_assembly_svg_output(tmp_path):
    """write_assembly_svg should produce SVG files."""
    from freecad_dome.assembly import assembly_guide_for_dome, write_assembly_svg
    params = DomeParameters(radius_m=5.0, frequency=2)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    guide = assembly_guide_for_dome(dome, params)
    svg_dir = tmp_path / "assembly"
    paths = write_assembly_svg(guide, dome, svg_dir)
    assert len(paths) == guide.total_stages
    for p in paths:
        from pathlib import Path as _P
        assert _P(p).exists()
        content = _P(p).read_text()
        assert '<svg' in content
        assert 'Etapp' in content


def test_assembly_svg_has_bom_legend(tmp_path):
    """SVG diagrams should include BOM legend text."""
    from freecad_dome.assembly import assembly_guide_for_dome, write_assembly_svg
    params = DomeParameters(radius_m=5.0, frequency=2)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    guide = assembly_guide_for_dome(dome, params)
    svg_dir = tmp_path / "assembly"
    paths = write_assembly_svg(guide, dome, svg_dir)
    # First SVG should contain BOM text
    if paths:
        from pathlib import Path as _P
        content = _P(paths[0]).read_text()
        assert 'Osaloend' in content or 'BOM' in content


def test_assembly_pipeline_step_position():
    """AssemblyGuideStep should be after TechDrawStep and before MultiDomeStep."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    assert "assembly_guide" in names
    td_idx = names.index("techdraw")
    ag_idx = names.index("assembly_guide")
    md_idx = names.index("multi_dome")
    assert ag_idx == td_idx + 1
    assert ag_idx == md_idx - 1


def test_assembly_step_skipped_when_disabled(tmp_path):
    """AssemblyGuideStep.should_run should return False when disabled."""
    step = pipeline.AssemblyGuideStep()
    ctx = pipeline.PipelineContext(
        params=DomeParameters(generate_assembly_guide=False),
        out_dir=tmp_path,
    )
    assert step.should_run(ctx) is False


def test_assembly_step_needs_dome(tmp_path):
    """AssemblyGuideStep.should_run needs dome to be present."""
    step = pipeline.AssemblyGuideStep()
    ctx = pipeline.PipelineContext(
        params=DomeParameters(generate_assembly_guide=True),
        out_dir=tmp_path,
    )
    ctx.dome = None
    assert step.should_run(ctx) is False


def test_assembly_step_enabled_with_dome(tmp_path):
    """AssemblyGuideStep.should_run returns True when enabled with dome."""
    step = pipeline.AssemblyGuideStep()
    params = DomeParameters(radius_m=5.0, frequency=2, generate_assembly_guide=True)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    ctx.dome = dome
    assert step.should_run(ctx) is True


# ---------------------------------------------------------------------------
# III1: Skylight / window tests
# ---------------------------------------------------------------------------


def test_skylight_params_defaults():
    """Skylight parameters should have sensible defaults."""
    p = DomeParameters()
    assert p.generate_skylights is False
    assert p.skylight_count == 1
    assert p.skylight_position == "apex"
    assert p.skylight_glass_thickness_m == 0.006
    assert p.skylight_frame_width_m == 0.05
    assert p.skylight_hinge_side == "top"
    assert p.skylight_material == "glass"
    assert p.skylight_panel_indices == []


def test_skylight_params_roundtrip():
    """Skylight params should survive to_dict / from_dict roundtrip."""
    p = DomeParameters(
        generate_skylights=True,
        skylight_count=3,
        skylight_position="ring",
        skylight_glass_thickness_m=0.008,
        skylight_frame_width_m=0.06,
        skylight_hinge_side="bottom",
        skylight_material="polycarbonate",
    )
    d = p.to_dict()
    assert d["generate_skylights"] is True
    assert d["skylight_count"] == 3
    assert d["skylight_material"] == "polycarbonate"
    p2 = DomeParameters.from_dict(d)
    assert p2.generate_skylights is True
    assert p2.skylight_count == 3
    assert p2.skylight_position == "ring"
    assert p2.skylight_glass_thickness_m == 0.008
    assert p2.skylight_material == "polycarbonate"


def test_skylight_plan_apex():
    """plan_skylights with apex mode should select top-most panels."""
    from freecad_dome.skylight import plan_skylights
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_skylights=True, skylight_count=1, skylight_position="apex",
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_skylights(dome, params)
    assert len(plan.skylights) == 1
    # Apex skylight should be near the top — highest centroid z
    sl = plan.skylights[0]
    assert sl.centroid[2] > params.radius_m * 0.5
    assert sl.glass_thickness_m == 0.006
    assert sl.material == "glass"
    assert sl.placement == "apex"


def test_skylight_plan_ring():
    """plan_skylights with ring mode should select panels in a ring."""
    from freecad_dome.skylight import plan_skylights
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_skylights=True, skylight_count=3, skylight_position="ring",
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_skylights(dome, params)
    assert len(plan.skylights) == 3
    for sl in plan.skylights:
        assert sl.placement == "ring"
        assert sl.area_m2 > 0


def test_skylight_plan_manual():
    """plan_skylights with manual mode should use explicit indices."""
    from freecad_dome.skylight import plan_skylights
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_skylights=True,
        skylight_position="manual",
        skylight_panel_indices=[0, 2],
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_skylights(dome, params)
    assert len(plan.skylights) == 2
    indices = {s.panel_index for s in plan.skylights}
    assert indices == {0, 2}
    for sl in plan.skylights:
        assert sl.placement == "manual"


def test_skylight_plan_to_dict():
    """SkylightPlan.to_dict should produce a serializable dictionary."""
    from freecad_dome.skylight import plan_skylights
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_skylights=True, skylight_count=1,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_skylights(dome, params)
    d = plan.to_dict()
    assert "skylight_count" in d
    assert d["skylight_count"] == 1
    assert "panels" in d
    assert len(d["panels"]) == 1
    assert "glass_thickness_m" in d["panels"][0]


def test_skylight_write_report(tmp_path):
    """write_skylight_report should create a JSON file."""
    from freecad_dome.skylight import plan_skylights, write_skylight_report
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_skylights=True, skylight_count=1,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_skylights(dome, params)
    path = write_skylight_report(plan, tmp_path)
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data["skylight_count"] == 1


def test_skylight_hinge_sides():
    """Different hinge_side values should select different edges."""
    from freecad_dome.skylight import plan_skylights
    params_top = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_skylights=True, skylight_count=1,
        skylight_hinge_side="top",
    )
    params_bottom = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_skylights=True, skylight_count=1,
        skylight_hinge_side="bottom",
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(5.0), params_top
    )
    plan_top = plan_skylights(dome, params_top)
    plan_bot = plan_skylights(dome, params_bottom)
    # Same panel but typically different hinge edges
    assert plan_top.skylights[0].panel_index == plan_bot.skylights[0].panel_index
    assert plan_top.skylights[0].hinge_side == "top"
    assert plan_bot.skylights[0].hinge_side == "bottom"


def test_skylight_material_polycarbonate():
    """Skylight material should propagate to SkylightPanel."""
    from freecad_dome.skylight import plan_skylights
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_skylights=True, skylight_count=1,
        skylight_material="polycarbonate",
        skylight_glass_thickness_m=0.010,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_skylights(dome, params)
    assert plan.skylights[0].material == "polycarbonate"
    assert plan.skylights[0].glass_thickness_m == 0.010


def test_skylight_plan_summary():
    """SkylightPlan.summary should return a human-readable string."""
    from freecad_dome.skylight import plan_skylights
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_skylights=True, skylight_count=2,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_skylights(dome, params)
    s = plan.summary()
    assert "2 skylights" in s
    assert "m²" in s


def test_skylight_invalid_manual_index_skipped():
    """Invalid manual panel indices should be silently skipped."""
    from freecad_dome.skylight import plan_skylights
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_skylights=True,
        skylight_position="manual",
        skylight_panel_indices=[0, 9999],
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_skylights(dome, params)
    assert len(plan.skylights) == 1
    assert plan.skylights[0].panel_index == 0


def test_skylight_pipeline_step_position():
    """SkylightStep should be after VentilationStep and before CoveringReportStep."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    assert "skylights" in names
    vent_idx = names.index("ventilation")
    sky_idx = names.index("skylights")
    cov_idx = names.index("covering_report")
    assert sky_idx == vent_idx + 1
    assert sky_idx == cov_idx - 1


def test_skylight_step_skipped_when_disabled(tmp_path):
    """SkylightStep.should_run should return False when disabled."""
    step = pipeline.SkylightStep()
    ctx = pipeline.PipelineContext(
        params=DomeParameters(generate_skylights=False),
        out_dir=tmp_path,
    )
    assert step.should_run(ctx) is False


def test_skylight_step_needs_dome(tmp_path):
    """SkylightStep.should_run needs dome to be present."""
    step = pipeline.SkylightStep()
    ctx = pipeline.PipelineContext(
        params=DomeParameters(generate_skylights=True),
        out_dir=tmp_path,
    )
    ctx.dome = None
    assert step.should_run(ctx) is False


def test_skylight_step_enabled_with_dome(tmp_path):
    """SkylightStep.should_run returns True when enabled with dome."""
    step = pipeline.SkylightStep()
    params = DomeParameters(radius_m=5.0, frequency=2, generate_skylights=True)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    ctx.dome = dome
    assert step.should_run(ctx) is True


def test_skylight_gui_import():
    """GUI dialog module should reference skylight controls."""
    import importlib
    mod = importlib.import_module("freecad_dome.gui_dialog")
    src = inspect.getsource(mod)
    assert "generate_skylights" in src
    assert "skylight_position" in src
    assert "skylight_material" in src


# ---------------------------------------------------------------------------
# III2: Riser wall (pikendusring) tests
# ---------------------------------------------------------------------------


def test_riser_params_defaults():
    """Riser wall parameters should have sensible defaults."""
    p = DomeParameters()
    assert p.generate_riser_wall is False
    assert p.riser_height_m == 1.0
    assert p.riser_thickness_m == 0.15
    assert p.riser_material == "concrete"
    assert p.riser_connection_type == "flange"
    assert p.riser_door_integration is True
    assert p.riser_stud_spacing_m == 0.6
    assert p.riser_segments == 36


def test_riser_params_roundtrip():
    """Riser params should survive to_dict / from_dict roundtrip."""
    p = DomeParameters(
        generate_riser_wall=True,
        riser_height_m=1.5,
        riser_thickness_m=0.20,
        riser_material="wood",
        riser_connection_type="bolted",
        riser_door_integration=False,
        riser_stud_spacing_m=0.4,
        riser_segments=48,
    )
    d = p.to_dict()
    assert d["generate_riser_wall"] is True
    assert d["riser_material"] == "wood"
    p2 = DomeParameters.from_dict(d)
    assert p2.generate_riser_wall is True
    assert p2.riser_height_m == 1.5
    assert p2.riser_thickness_m == 0.20
    assert p2.riser_material == "wood"
    assert p2.riser_connection_type == "bolted"
    assert p2.riser_door_integration is False
    assert p2.riser_stud_spacing_m == 0.4
    assert p2.riser_segments == 48


def test_riser_plan_basic():
    """plan_riser_wall should produce a valid plan."""
    from freecad_dome.riser_wall import plan_riser_wall
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_riser_wall=True, riser_height_m=1.0, riser_thickness_m=0.15,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_riser_wall(dome, params)
    assert plan.riser_height_m == 1.0
    assert plan.riser_thickness_m == 0.15
    assert plan.belt_radius_m > 0
    assert plan.outer_radius_m > plan.inner_radius_m
    assert plan.wall_area_m2 > 0
    assert plan.volume_m3 > 0
    assert plan.riser_bottom_z_m < plan.riser_top_z_m


def test_riser_plan_connections():
    """Riser plan should have connections at belt nodes."""
    from freecad_dome.riser_wall import plan_riser_wall
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_riser_wall=True, riser_height_m=1.2,
        riser_connection_type="bolted",
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_riser_wall(dome, params)
    assert len(plan.connections) > 0
    for conn in plan.connections:
        assert conn.connection_type == "bolted"
        assert 0.0 <= conn.azimuth_deg < 360.0


def test_riser_plan_wood_studs():
    """Wood riser should generate studs."""
    from freecad_dome.riser_wall import plan_riser_wall
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_riser_wall=True, riser_height_m=1.0,
        riser_material="wood", riser_stud_spacing_m=0.6,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_riser_wall(dome, params)
    assert len(plan.studs) > 0
    for stud in plan.studs:
        assert stud.length_m == pytest.approx(1.0, abs=0.01)
        assert stud.z_bottom_m < stud.z_top_m


def test_riser_plan_concrete_no_studs():
    """Concrete riser should have no studs."""
    from freecad_dome.riser_wall import plan_riser_wall
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_riser_wall=True, riser_material="concrete",
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_riser_wall(dome, params)
    assert len(plan.studs) == 0


def test_riser_door_cutout():
    """Riser plan should include door cutout when door is active."""
    from freecad_dome.riser_wall import plan_riser_wall
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_riser_wall=True, riser_height_m=2.5,
        riser_door_integration=True,
        generate_base_wall=True,
        door_width_m=0.9, door_height_m=2.1,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_riser_wall(dome, params)
    assert plan.door_cutout is not None
    assert plan.door_cutout.door_width_m == 0.9
    assert plan.door_cutout.door_height_m == 2.1
    assert plan.door_cutout.fits_in_riser is True


def test_riser_door_cutout_too_tall():
    """Door taller than riser should set fits_in_riser=False."""
    from freecad_dome.riser_wall import plan_riser_wall
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_riser_wall=True, riser_height_m=1.0,
        riser_door_integration=True,
        generate_base_wall=True,
        door_height_m=2.1,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_riser_wall(dome, params)
    assert plan.door_cutout is not None
    assert plan.door_cutout.fits_in_riser is False


def test_riser_no_door_when_disabled():
    """No door cutout when riser_door_integration is False."""
    from freecad_dome.riser_wall import plan_riser_wall
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_riser_wall=True,
        riser_door_integration=False,
        generate_base_wall=True,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_riser_wall(dome, params)
    assert plan.door_cutout is None


def test_riser_plan_to_dict():
    """RiserWallPlan.to_dict should produce a serializable dictionary."""
    from freecad_dome.riser_wall import plan_riser_wall
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_riser_wall=True, riser_material="wood",
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_riser_wall(dome, params)
    d = plan.to_dict()
    assert "riser_height_m" in d
    assert "connection_count" in d
    assert "stud_count" in d
    assert d["stud_count"] > 0
    assert d["material"] == "wood"


def test_riser_write_report(tmp_path):
    """write_riser_report should create a JSON file."""
    from freecad_dome.riser_wall import plan_riser_wall, write_riser_report
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_riser_wall=True,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_riser_wall(dome, params)
    path = write_riser_report(plan, tmp_path)
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data["riser_height_m"] == 1.0
    assert data["material"] == "concrete"


def test_riser_plan_summary():
    """RiserWallPlan.summary should return a human-readable string."""
    from freecad_dome.riser_wall import plan_riser_wall
    params = DomeParameters(
        radius_m=5.0, frequency=2,
        generate_riser_wall=True, riser_height_m=1.5,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    plan = plan_riser_wall(dome, params)
    s = plan.summary()
    assert "Riser wall" in s
    assert "concrete" in s
    assert "1.50" in s


def test_riser_pipeline_step_position():
    """RiserWallStep should be after BaseWallStep and before StrutGenerationStep."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    assert "riser_wall" in names
    bw_idx = names.index("base_wall")
    rw_idx = names.index("riser_wall")
    sg_idx = names.index("strut_generation")
    assert rw_idx == bw_idx + 1
    assert rw_idx == sg_idx - 1


def test_riser_step_skipped_when_disabled(tmp_path):
    """RiserWallStep.should_run should return False when disabled."""
    step = pipeline.RiserWallStep()
    ctx = pipeline.PipelineContext(
        params=DomeParameters(generate_riser_wall=False),
        out_dir=tmp_path,
    )
    assert step.should_run(ctx) is False


def test_riser_step_needs_dome(tmp_path):
    """RiserWallStep.should_run needs dome to be present."""
    step = pipeline.RiserWallStep()
    ctx = pipeline.PipelineContext(
        params=DomeParameters(generate_riser_wall=True),
        out_dir=tmp_path,
    )
    ctx.dome = None
    assert step.should_run(ctx) is False


def test_riser_step_enabled_with_dome(tmp_path):
    """RiserWallStep.should_run returns True when enabled with dome."""
    step = pipeline.RiserWallStep()
    params = DomeParameters(radius_m=5.0, frequency=2, generate_riser_wall=True)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    ctx.dome = dome
    assert step.should_run(ctx) is True


def test_riser_gui_import():
    """GUI dialog module should reference riser wall controls."""
    import importlib
    mod = importlib.import_module("freecad_dome.gui_dialog")
    src = inspect.getsource(mod)
    assert "generate_riser_wall" in src
    assert "riser_material" in src
    assert "riser_connection_type" in src


# ---------------------------------------------------------------------------
# III3: Multi-dome (Mitmikkuppel)
# ---------------------------------------------------------------------------


def test_multi_dome_config_defaults():
    """MultiDomeConfig should have correct defaults."""
    p = DomeParameters()
    assert p.multi_dome_enabled is False
    assert p.dome_instances_json == "[]"
    assert math.isclose(p.corridor_width_m, 1.2)
    assert math.isclose(p.corridor_height_m, 2.1)
    assert math.isclose(p.corridor_wall_thickness_m, 0.15)
    assert p.corridor_material == "wood"
    assert p.corridor_definitions_json == "[]"
    assert p.merge_foundation is True
    assert p.merge_bom is True


def test_multi_dome_config_field_count():
    """MultiDomeConfig should have 9 fields."""
    from freecad_dome.parameters import MultiDomeConfig, dc_fields
    assert len(dc_fields(MultiDomeConfig)) == 9


def test_multi_dome_roundtrip():
    """Multi-dome params should survive to_dict/from_dict roundtrip."""
    p1 = DomeParameters(
        multi_dome_enabled=True,
        dome_instances_json='[{"label": "X", "offset_x_m": 5.0}]',
        corridor_width_m=1.5,
    )
    d = p1.to_dict()
    assert d["multi_dome_enabled"] is True
    assert "offset_x_m" in d["dome_instances_json"]
    p2 = DomeParameters.from_dict(d)
    assert p2.multi_dome_enabled is True
    assert math.isclose(p2.corridor_width_m, 1.5)


def test_parse_dome_instances_primary_only():
    """Without secondary domes the list should contain only the primary."""
    p = DomeParameters(radius_m=5.0, hemisphere_ratio=0.5)
    domes = multi_dome.parse_dome_instances(p)
    assert len(domes) == 1
    assert domes[0].index == 0
    assert domes[0].label == "Peakuppel"
    assert math.isclose(domes[0].radius_m, 5.0)
    assert domes[0].belt_radius_m > 0


def test_parse_dome_instances_with_secondary():
    """dome_instances_json should add secondary domes."""
    p = DomeParameters(
        radius_m=5.0,
        hemisphere_ratio=0.5,
        dome_instances_json=json.dumps([
            {"label": "Anneks", "offset_x_m": 10.0, "offset_y_m": 0.0,
             "overrides": {"radius_m": 3.0}},
        ]),
    )
    domes = multi_dome.parse_dome_instances(p)
    assert len(domes) == 2
    assert domes[1].label == "Anneks"
    assert math.isclose(domes[1].offset_x_m, 10.0)
    assert math.isclose(domes[1].radius_m, 3.0)


def test_parse_corridor_definitions():
    """Corridor definitions should be parsed and geometry computed."""
    p = DomeParameters(
        radius_m=5.0,
        hemisphere_ratio=0.5,
        dome_instances_json=json.dumps([
            {"label": "Anneks", "offset_x_m": 15.0, "offset_y_m": 0.0,
             "overrides": {"radius_m": 3.0}},
        ]),
        corridor_definitions_json=json.dumps([
            {"from_dome": 0, "to_dome": 1},
        ]),
    )
    domes = multi_dome.parse_dome_instances(p)
    corridors = multi_dome.parse_corridor_definitions(p, domes)
    assert len(corridors) == 1
    c = corridors[0]
    assert c.from_dome == 0
    assert c.to_dome == 1
    assert c.length_m > 0
    assert c.floor_area_m2 > 0
    assert c.wall_area_m2 > 0


def test_corridor_invalid_self_reference():
    """Corridor from dome X to dome X should be skipped."""
    p = DomeParameters(
        radius_m=5.0,
        hemisphere_ratio=0.5,
        corridor_definitions_json=json.dumps([
            {"from_dome": 0, "to_dome": 0},
        ]),
    )
    domes = multi_dome.parse_dome_instances(p)
    corridors = multi_dome.parse_corridor_definitions(p, domes)
    assert len(corridors) == 0


def test_corridor_invalid_index():
    """Corridor referencing non-existent dome should be skipped."""
    p = DomeParameters(
        radius_m=5.0,
        hemisphere_ratio=0.5,
        corridor_definitions_json=json.dumps([
            {"from_dome": 0, "to_dome": 5},
        ]),
    )
    domes = multi_dome.parse_dome_instances(p)
    corridors = multi_dome.parse_corridor_definitions(p, domes)
    assert len(corridors) == 0


def test_dome_instance_distance():
    """DomeInstance.distance_to should compute correct distance."""
    d0 = multi_dome.DomeInstance(index=0, label="A", offset_x_m=0.0, offset_y_m=0.0)
    d1 = multi_dome.DomeInstance(index=1, label="B", offset_x_m=3.0, offset_y_m=4.0)
    assert math.isclose(d0.distance_to(d1), 5.0)


def test_dome_instance_edge_distance():
    """Edge distance should subtract belt radii."""
    d0 = multi_dome.DomeInstance(index=0, label="A", belt_radius_m=2.0)
    d1 = multi_dome.DomeInstance(
        index=1, label="B", offset_x_m=10.0, belt_radius_m=3.0,
    )
    assert math.isclose(d0.edge_distance_to(d1), 5.0)


def test_plan_multi_dome_basic(tmp_path):
    """plan_multi_dome should produce a valid plan with 2 domes + 1 corridor."""
    p = DomeParameters(
        radius_m=5.0,
        frequency=2,
        hemisphere_ratio=0.5,
        multi_dome_enabled=True,
        dome_instances_json=json.dumps([
            {"label": "Anneks", "offset_x_m": 15.0, "overrides": {"radius_m": 3.0}},
        ]),
        corridor_definitions_json=json.dumps([
            {"from_dome": 0, "to_dome": 1},
        ]),
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(p.radius_m), p
    )
    plan = multi_dome.plan_multi_dome(dome, p)
    assert plan.dome_count == 2
    assert plan.corridor_count == 1
    assert plan.total_floor_area_m2 > 0
    assert len(plan.merged_foundation_anchors) > 0
    assert len(plan.merged_bom) > 0
    assert "2 domes" in plan.summary()


def test_plan_multi_dome_no_secondary():
    """With only primary dome, plan should still work."""
    p = DomeParameters(
        radius_m=5.0,
        frequency=2,
        hemisphere_ratio=0.5,
        multi_dome_enabled=True,
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(p.radius_m), p
    )
    plan = multi_dome.plan_multi_dome(dome, p)
    assert plan.dome_count == 1
    assert plan.corridor_count == 0


def test_write_multi_dome_report(tmp_path):
    """write_multi_dome_report should create a valid JSON file."""
    p = DomeParameters(
        radius_m=5.0,
        frequency=2,
        hemisphere_ratio=0.5,
        multi_dome_enabled=True,
        dome_instances_json=json.dumps([
            {"label": "Anneks", "offset_x_m": 12.0, "overrides": {"radius_m": 3.0}},
        ]),
        corridor_definitions_json=json.dumps([
            {"from_dome": 0, "to_dome": 1},
        ]),
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(p.radius_m), p
    )
    plan = multi_dome.plan_multi_dome(dome, p)
    out = multi_dome.write_multi_dome_report(plan, tmp_path)
    assert out.exists()
    data = json.loads(out.read_text())
    assert "multi_dome_plan" in data
    assert data["multi_dome_plan"]["dome_count"] == 2
    assert data["multi_dome_plan"]["corridor_count"] == 1


def test_multi_dome_step_disabled():
    """MultiDomeStep.should_run returns False when disabled."""
    step = pipeline.MultiDomeStep()
    ctx = pipeline.PipelineContext(params=DomeParameters())
    assert step.should_run(ctx) is False


def test_multi_dome_step_no_dome(tmp_path):
    """MultiDomeStep.should_run returns False without dome."""
    step = pipeline.MultiDomeStep()
    ctx = pipeline.PipelineContext(
        params=DomeParameters(multi_dome_enabled=True),
        out_dir=tmp_path,
    )
    ctx.dome = None
    assert step.should_run(ctx) is False


def test_multi_dome_step_enabled_with_dome(tmp_path):
    """MultiDomeStep.should_run returns True when enabled with dome."""
    step = pipeline.MultiDomeStep()
    params = DomeParameters(radius_m=5.0, frequency=2, multi_dome_enabled=True)
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(params.radius_m), params
    )
    ctx = pipeline.PipelineContext(params=params, out_dir=tmp_path)
    ctx.dome = dome
    assert step.should_run(ctx) is True


def test_multi_dome_step_position():
    """MultiDomeStep should be after assembly_guide and before weather_protection."""
    steps = pipeline.default_steps()
    names = [s.name for s in steps]
    assert "multi_dome" in names
    ag_idx = names.index("assembly_guide")
    md_idx = names.index("multi_dome")
    wp_idx = names.index("weather_protection")
    assert md_idx == ag_idx + 1
    assert md_idx == wp_idx - 1


def test_multi_dome_merged_bom_corridor_items():
    """Merged BOM should include corridor material items."""
    p = DomeParameters(
        radius_m=5.0,
        hemisphere_ratio=0.5,
        multi_dome_enabled=True,
        dome_instances_json=json.dumps([
            {"label": "Side", "offset_x_m": 15.0, "overrides": {"radius_m": 4.0}},
        ]),
        corridor_definitions_json=json.dumps([
            {"from_dome": 0, "to_dome": 1},
        ]),
    )
    dome = tessellation.tessellate(
        icosahedron.build_icosahedron(p.radius_m), p
    )
    plan = multi_dome.plan_multi_dome(dome, p)
    types = [item.get("type", "") for item in plan.merged_bom]
    assert "corridor_frame" in types
    assert "corridor_floor" in types


def test_multi_dome_gui_import():
    """GUI dialog module should reference multi-dome controls."""
    import importlib
    mod = importlib.import_module("freecad_dome.gui_dialog")
    src = inspect.getsource(mod)
    assert "multi_dome_enabled" in src
    assert "corridor_width" in src
    assert "Mitmikkuppel" in src


def test_corridor_to_dict():
    """Corridor.to_dict should include all fields."""
    c = multi_dome.Corridor(
        from_dome=0, to_dome=1, width_m=1.2, height_m=2.1,
        length_m=5.0, material="wood",
    )
    d = c.to_dict()
    assert d["from_dome"] == 0
    assert d["to_dome"] == 1
    assert math.isclose(d["floor_area_m2"], 6.0)
    assert d["volume_m3"] > 0


def test_multi_dome_plan_to_dict():
    """MultiDomePlan.to_dict should be JSON-serialisable."""
    plan = multi_dome.MultiDomePlan(
        domes=[multi_dome.DomeInstance(index=0, label="Main", belt_radius_m=4.0)],
        corridors=[],
        merged_foundation_anchors=[],
        merged_bom=[],
    )
    d = plan.to_dict()
    # Must be JSON-serialisable
    text = json.dumps(d)
    assert "Main" in text