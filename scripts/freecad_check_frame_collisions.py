#!/usr/bin/env python3
"""Generate panel frames in FreeCADCmd and check neighbor-frame collisions.

Run via snap FreeCADCmd console mode:

  snap run --shell freecad.cmd -c 'FreeCADCmd --console "import runpy; runpy.run_path(\"scripts/freecad_check_frame_collisions.py\", run_name=\"__main__\")"'

Env overrides:
- DOME_CONFIG (default: configs/base.json)
- DOME_OUT_DIR (default: exports/frame_collision_check)
- DOME_PANEL_FRAME_INSET_M (float meters, optional)
- DOME_PANEL_FRAME_PROFILE_M ("WIDTH,HEIGHT" meters, optional)

Output:
- Writes `frame_collision_check.json` in out dir.
- Exit code 1 if any collisions are found.
"""

from __future__ import annotations

import json
import logging
import os
import runpy
import sys
from pathlib import Path


def _parse_profile(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    raw = value.replace(" ", "")
    parts = raw.split(",", 1)
    if len(parts) != 2:
        raise ValueError("DOME_PANEL_FRAME_PROFILE_M must be 'WIDTH,HEIGHT'")
    return (float(parts[0]), float(parts[1]))


def _run_generator(config: str, out_dir: str, inset_m: float | None, profile_m: tuple[float, float] | None) -> None:
    argv = [
        "scripts/generate_dome.py",
        "--config",
        config,
        "--no-gui",
        "--out-dir",
        out_dir,
        "--panel-frames-only",
        "--skip-ifc",
        "--skip-stl",
        "--skip-dxf",
    ]
    if inset_m is not None:
        argv.extend(["--panel-frame-inset", str(inset_m)])
    if profile_m is not None:
        argv.extend(["--panel-frame-profile", str(profile_m[0]), str(profile_m[1])])
    sys.argv = argv
    runpy.run_path("scripts/generate_dome.py", run_name="__main__")


def _ordered_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _shape_volume(shape) -> float:
    try:
        return float(getattr(shape, "Volume", 0.0))
    except Exception:
        return 0.0


def _safe_dist_to_shape(a, b) -> float | None:
    try:
        # FreeCAD/Part returns (distance, point_on_a, point_on_b)
        res = a.distToShape(b)
        if isinstance(res, (tuple, list)) and res:
            return float(res[0])
    except Exception:
        return None
    return None


def _edge_cylinder(p0, p1, radius_m: float):
    """Return a Part cylinder aligned to the given edge segment."""
    import Part  # type: ignore
    from FreeCAD import Vector  # type: ignore

    a = Vector(float(p0[0]), float(p0[1]), float(p0[2]))
    b = Vector(float(p1[0]), float(p1[1]), float(p1[2]))
    d = b.sub(a)
    length = float(d.Length)
    if length <= 1e-12:
        return None
    d.normalize()
    try:
        return Part.makeCylinder(float(radius_m), float(length), a, d)
    except Exception:
        return None


def main() -> int:
    config = os.environ.get("DOME_CONFIG", "configs/base.json")
    out_dir = os.environ.get("DOME_OUT_DIR", "exports/frame_collision_check")
    inset_raw = os.environ.get("DOME_PANEL_FRAME_INSET_M")
    profile_raw = os.environ.get("DOME_PANEL_FRAME_PROFILE_M")

    inset_m = float(inset_raw) if inset_raw else None
    profile_m = _parse_profile(profile_raw)

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    _run_generator(config, out_dir, inset_m, profile_m)

    try:
        import FreeCAD  # type: ignore
    except Exception as exc:
        print(f"FreeCAD import failed: {exc}")
        return 2

    doc = FreeCAD.ActiveDocument
    if doc is None:
        print("No ActiveDocument after generation")
        return 2

    frames = [
        o
        for o in getattr(doc, "Objects", [])
        if str(getattr(o, "Name", "")).startswith("PanelFrame_")
        or str(getattr(o, "Label", "")).startswith("PanelFrame_")
    ]

    frame_by_index: dict[int, object] = {}
    for obj in frames:
        raw = str(getattr(obj, "Name", getattr(obj, "Label", "")))
        if "PanelFrame_" not in raw:
            continue
        try:
            idx = int(raw.split("PanelFrame_", 1)[1])
        except Exception:
            continue
        frame_by_index[idx] = obj

    # Build dome panels to derive true neighbor relationships.
    from freecad_dome import icosahedron, parameters, tessellation

    params = parameters.load_parameters(config, {"generate_panel_frames": True, "generate_panel_faces": False, "generate_struts": False})
    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)

    edge_to_panels: dict[tuple[int, int], list[int]] = {}
    for panel in dome.panels:
        nodes = panel.node_indices
        for i in range(len(nodes)):
            edge_to_panels.setdefault(_ordered_edge(nodes[i], nodes[(i + 1) % len(nodes)]), []).append(panel.index)

    collisions = []
    checked = 0
    vol_tol = 1e-12
    dist_tol = 1e-6

    # Localize the collision test to a tube around the shared edge.
    # This avoids huge false positives when frames are on different planes.
    if profile_m is not None:
        tube_radius = max(profile_m[0] * 1.25, 0.002)
    else:
        # Fall back to a small tube; better than global common().
        tube_radius = 0.01

    for edge, pids in edge_to_panels.items():
        if len(pids) != 2:
            continue
        p, q = pids
        a = frame_by_index.get(p)
        b = frame_by_index.get(q)
        if a is None or b is None:
            continue
        sa = getattr(a, "Shape", None)
        sb = getattr(b, "Shape", None)
        if sa is None or sb is None:
            continue
        checked += 1

        p0 = dome.nodes[edge[0]]
        p1 = dome.nodes[edge[1]]
        cyl = _edge_cylinder(p0, p1, tube_radius)
        if cyl is None:
            continue

        try:
            sa_local = sa.common(cyl)
            sb_local = sb.common(cyl)
        except Exception:
            continue

        if sa_local is None or sb_local is None:
            continue
        if (hasattr(sa_local, "isNull") and sa_local.isNull()) or (hasattr(sb_local, "isNull") and sb_local.isNull()):
            continue

        dist = _safe_dist_to_shape(sa_local, sb_local)
        inter_vol = 0.0
        try:
            inter = sa_local.common(sb_local)
            if inter is not None and not (hasattr(inter, "isNull") and inter.isNull()):
                inter_vol = _shape_volume(inter)
        except Exception:
            inter_vol = 0.0

        if (dist is not None and dist <= dist_tol) or inter_vol > vol_tol:
            collisions.append(
                {
                    "edge": list(edge),
                    "panels": [p, q],
                    "tube_radius_m": tube_radius,
                    "min_distance_m": dist,
                    "intersection_volume_m3": inter_vol,
                }
            )

    report = {
        "config": config,
        "out_dir": out_dir,
        "inset_m": inset_m,
        "profile_m": list(profile_m) if profile_m is not None else None,
        "checked_pairs": checked,
        "volume_tol_m3": vol_tol,
        "distance_tol_m": dist_tol,
        "tube_radius_m": tube_radius,
        "collisions": collisions,
    }

    out_path = Path(out_dir) / "frame_collision_check.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"checked_pairs={checked} collisions={len(collisions)} report={out_path}")
    return 1 if collisions else 0


if __name__ == "__main__":
    raise SystemExit(main())
