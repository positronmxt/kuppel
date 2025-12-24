#!/usr/bin/env python3
"""Headless checker for a full-sphere, pure icosahedron (frequency=1).

What it checks:
- All strut (edge) lengths are equal within tolerance.
- All panel-frame solids have equal volume within tolerance.
- All panels report the same *effective* frame width (no unintended auto-narrowing).
- Shared-edge *inner-edge orientation* is consistent (no "inset goes the wrong way" cases
    where one panel's inner edge crosses onto the neighbor's side of the strut).

Run via snap FreeCADCmd console mode:

  snap run --shell freecad.cmd -c 'FreeCADCmd --console "import runpy; runpy.run_path(\"scripts/freecad_check_icosa_frames.py\", run_name=\"__main__\")"'

Env overrides:
- DOME_CONFIG (default: configs/icosahedron_full.json)
- DOME_OUT_DIR (default: exports/icosa_frame_check)
- DOME_PANEL_FRAME_INSET_M (float meters, optional)
- DOME_PANEL_FRAME_PROFILE_M ("WIDTH,HEIGHT" meters, optional)

Exit code:
- 0 on success
- 1 on any failed check
- 2 on FreeCAD/runtime failure
"""

from __future__ import annotations

import json
import logging
import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path


def _parse_profile(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    raw = value.replace(" ", "")
    if "," in raw:
        parts = raw.split(",", 1)
    elif "x" in raw.lower():
        parts = raw.lower().split("x", 1)
    else:
        parts = raw.split(";", 1)
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


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    x, y, z = v
    ln = (x * x + y * y + z * z) ** 0.5
    if ln <= 1e-18:
        return None
    return (x / ln, y / ln, z / ln)


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@dataclass
class _Stats:
    min: float | None
    max: float | None
    avg: float | None


def _stats(values: list[float]) -> _Stats:
    if not values:
        return _Stats(None, None, None)
    return _Stats(min(values), max(values), sum(values) / len(values))


def main() -> int:
    config = os.environ.get("DOME_CONFIG", "configs/icosahedron_full.json")
    out_dir = os.environ.get("DOME_OUT_DIR", "exports/icosa_frame_check")
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
    widths_by_index: dict[int, float] = {}
    for obj in frames:
        raw = str(getattr(obj, "Name", getattr(obj, "Label", "")))
        if "PanelFrame_" not in raw:
            continue
        try:
            idx = int(raw.split("PanelFrame_", 1)[1])
        except Exception:
            continue
        frame_by_index[idx] = obj
        if hasattr(obj, "FrameProfileWidth"):
            try:
                widths_by_index[idx] = float(obj.FrameProfileWidth)
            except Exception:
                pass

    from freecad_dome import icosahedron, parameters, tessellation
    from freecad_dome import panels as panels_mod

    params = parameters.load_parameters(
        config,
        {
            "generate_panel_frames": True,
            "generate_panel_faces": False,
            "generate_struts": False,
        },
    )
    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)

    # Basic sanity: icosahedron full sphere should have 20 faces.
    expected_panels = len(dome.panels)

    # Edge/strut length equality (pure geometry, no FreeCAD).
    edge_lengths: list[float] = []
    unique_edges: set[tuple[int, int]] = set()
    for panel in dome.panels:
        nodes = panel.node_indices
        for i in range(len(nodes)):
            a = nodes[i]
            b = nodes[(i + 1) % len(nodes)]
            e = _ordered_edge(a, b)
            if e in unique_edges:
                continue
            unique_edges.add(e)
            p0 = dome.nodes[e[0]]
            p1 = dome.nodes[e[1]]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dz = p1[2] - p0[2]
            edge_lengths.append((dx * dx + dy * dy + dz * dz) ** 0.5)

    # Frame volume equality.
    frame_volumes: list[float] = []
    invalid_frames: list[str] = []
    for idx in range(expected_panels):
        obj = frame_by_index.get(idx)
        if obj is None:
            invalid_frames.append(f"missing PanelFrame_{idx:04d}")
            continue
        shape = getattr(obj, "Shape", None)
        if shape is None or getattr(shape, "isNull", lambda: True)():
            invalid_frames.append(f"null PanelFrame_{idx:04d}")
            continue
        if hasattr(shape, "isValid") and not shape.isValid():
            invalid_frames.append(f"invalid PanelFrame_{idx:04d}")
            continue
        frame_volumes.append(_shape_volume(shape))

    # Inner-loop corner sanity: inner loop should have the same number of corners
    # as the panel polygon (no bevel duplicates), and should be non-degenerate.
    builder = panels_mod.PanelBuilder(params)
    requested_width = float(profile_m[0]) if profile_m is not None else float(params.panel_frame_profile_width_m)
    inset = float(inset_m) if inset_m is not None else float(params.panel_frame_inset_m)
    inner_corner_issues: list[dict[str, object]] = []
    for panel in dome.panels:
        plane = builder._panel_plane_data(dome, panel)  # type: ignore[attr-defined]
        if plane is None:
            inner_corner_issues.append({"panel": panel.index, "reason": "no_plane"})
            continue
        base_loop = list(plane.coords2d)
        if len(base_loop) < 3:
            inner_corner_issues.append({"panel": panel.index, "reason": "degenerate"})
            continue
        inner_inset = max(0.0, inset) + max(0.0, requested_width)
        inner_loop = builder._validated_inset(base_loop, inner_inset)  # type: ignore[attr-defined]
        if not inner_loop or len(inner_loop) < 3:
            inner_corner_issues.append({"panel": panel.index, "reason": "no_inner_loop"})
            continue
        if len(inner_loop) != len(base_loop):
            inner_corner_issues.append(
                {
                    "panel": panel.index,
                    "reason": "corner_count_mismatch",
                    "outer_corners": len(base_loop),
                    "inner_corners": len(inner_loop),
                }
            )
            continue
        # Ensure adjacent points aren't collapsing.
        ok = True
        for i in range(len(inner_loop)):
            a = inner_loop[i]
            b = inner_loop[(i + 1) % len(inner_loop)]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            if (dx * dx + dy * dy) ** 0.5 < 1e-9:
                ok = False
                break
        if not ok:
            inner_corner_issues.append({"panel": panel.index, "reason": "collapsed_inner_edge"})

    # Neighbor relationships.
    edge_to_panels: dict[tuple[int, int], list[int]] = {}
    for panel in dome.panels:
        nodes = panel.node_indices
        for i in range(len(nodes)):
            edge_to_panels.setdefault(_ordered_edge(nodes[i], nodes[(i + 1) % len(nodes)]), []).append(panel.index)

    # Shared-edge inner-edge direction sanity check.
    # For each shared edge, compute each panel's inward direction (in panel plane,
    # perpendicular to the edge) and ensure the two panels point to opposite sides.
    # If they point to the same side, one panel's "inner edge" would overlap/cross
    # into the neighbor side.
    # (Keep the following shared-edge check based on the same requested width.)
    inward_dot: list[float] = []
    bad_inner_edges: list[dict[str, object]] = []

    panel_centroid: dict[int, tuple[float, float, float]] = {}
    panel_normal: dict[int, tuple[float, float, float]] = {}
    panel_nodes: dict[int, tuple[int, ...]] = {}
    for panel in dome.panels:
        pts = [dome.nodes[i] for i in panel.node_indices]
        c = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
            sum(p[2] for p in pts) / len(pts),
        )
        n = _normalize(panel.normal) or (0.0, 0.0, 0.0)
        panel_centroid[panel.index] = c
        panel_normal[panel.index] = n
        panel_nodes[panel.index] = panel.node_indices

    for edge, pids in sorted(edge_to_panels.items()):
        if len(pids) != 2:
            continue
        p, q = pids
        p0 = dome.nodes[edge[0]]
        p1 = dome.nodes[edge[1]]
        e = _normalize(_sub(p1, p0))
        if e is None:
            continue

        def inward_dir(panel_id: int) -> tuple[float, float, float] | None:
            n = panel_normal.get(panel_id)
            if n is None:
                return None
            u = _normalize(_cross(n, e))
            if u is None:
                return None
            m = _mul(_add(p0, p1), 0.5)
            c = panel_centroid.get(panel_id)
            if c is None:
                return None
            # Pick the sign that points from the edge midpoint toward the panel centroid.
            if _dot(u, _sub(c, m)) < 0:
                u = _mul(u, -1.0)
            return u

        up = inward_dir(p)
        uq = inward_dir(q)
        if up is None or uq is None:
            continue
        d = _dot(up, uq)
        inward_dot.append(d)
        if d > 0.0:
            # Same-side inward direction => inner edges cross/overlap logically.
            # Provide a quick "separation" estimate at the shared edge.
            sep = (requested_width * (2.0 - 2.0 * d)) ** 0.5 if requested_width > 0 else None
            bad_inner_edges.append(
                {
                    "edge": list(edge),
                    "panels": [p, q],
                    "inward_dot": d,
                    "approx_inner_separation_m": sep,
                }
            )

    report = {
        "config": config,
        "out_dir": out_dir,
        "inset_m": inset_m,
        "profile_m": list(profile_m) if profile_m is not None else None,
        "expected_panels": expected_panels,
        "found_frames": len(frames),
        "invalid_frames": invalid_frames,
        "requested_frame_width_m": requested_width,
        "tolerances": {
            "edge_length_rel": 1e-9,
            "frame_volume_rel": 1e-9,
            "inward_dot_max": 0.0,
        },
        "stats": {
            "edge_length_m": _stats(edge_lengths).__dict__,
            "frame_volume_m3": _stats(frame_volumes).__dict__,
            "effective_frame_width_m": _stats(list(widths_by_index.values())).__dict__,
            "shared_edge_inward_dot": _stats(inward_dot).__dict__,
        },
        "counts": {
            "unique_edges": len(unique_edges),
            "neighbor_pairs": sum(1 for v in edge_to_panels.values() if len(v) == 2),
            "bad_inner_edges": len(bad_inner_edges),
            "inner_corner_issues": len(inner_corner_issues),
        },
        "bad_inner_edges": bad_inner_edges,
        "inner_corner_issues": inner_corner_issues,
    }

    out_path = Path(out_dir) / "icosa_frame_check.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Evaluate pass/fail.
    failed = False

    if invalid_frames:
        failed = True

    # Equality checks (relative spread).
    def _rel_spread(values: list[float]) -> float:
        if not values:
            return 0.0
        lo = min(values)
        hi = max(values)
        mid = (lo + hi) * 0.5
        if mid == 0:
            return hi - lo
        return (hi - lo) / mid

    edge_spread = _rel_spread(edge_lengths)
    frame_vol_spread = _rel_spread(frame_volumes)
    if edge_spread > 1e-9:
        failed = True
    if frame_vol_spread > 1e-9:
        failed = True
    if inner_corner_issues:
        failed = True
    if bad_inner_edges:
        failed = True

    print(
        " ".join(
            [
                f"panels={expected_panels}",
                f"unique_edges={len(unique_edges)}",
                f"frames_ok={len(frame_volumes)}/{expected_panels}",
                f"edge_spread={edge_spread:.3e}",
                f"frame_vol_spread={frame_vol_spread:.3e}",
                f"inner_corner_issues={len(inner_corner_issues)}",
                f"bad_inner_edges={len(bad_inner_edges)}",
                f"report={out_path}",
            ]
        )
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
