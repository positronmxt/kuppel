import math

from freecad_dome import icosahedron, tessellation, panels, parameters
from freecad_dome.parameters import DomeParameters
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
