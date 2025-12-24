#!/usr/bin/env python3
"""Check that (truncation) panel-frame outer edges stay parallel across neighbors.

This is a geometry-level check that does NOT require FreeCAD.

It uses the same panel plane + inset logic as `freecad_dome.panels.PanelBuilder` to
compute an *outer loop* in 2D and then compares directions of the outer-loop
segments corresponding to shared panel edges.

We focus on edges where at least one of the two adjacent panels is non-triangular
(typical of truncation-generated polygons).

Output:
- Writes `exports/frame_parallel_check.json`
- Exit code 1 if violations are found.

Usage:
  python3 scripts/check_truncation_frame_parallel.py --config configs/base.json

Optional:
  --deg 2.0         # angle tolerance in degrees (default 2)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freecad_dome import icosahedron, parameters, tessellation
from freecad_dome.panels import PanelBuilder


def _normalize2(v: Tuple[float, float]) -> Tuple[float, float] | None:
    ln = math.hypot(v[0], v[1])
    if ln < 1e-15:
        return None
    return (v[0] / ln, v[1] / ln)


def _normalize3(v: Tuple[float, float, float]) -> Tuple[float, float, float] | None:
    ln = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if ln < 1e-15:
        return None
    return (v[0] / ln, v[1] / ln, v[2] / ln)


def _dot3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _lift_dir(axis_u: Tuple[float, float, float], axis_v: Tuple[float, float, float], d2: Tuple[float, float]) -> Tuple[float, float, float] | None:
    v = (
        axis_u[0] * d2[0] + axis_v[0] * d2[1],
        axis_u[1] * d2[0] + axis_v[1] * d2[1],
        axis_u[2] * d2[0] + axis_v[2] * d2[1],
    )
    return _normalize3(v)


def _ordered_edge(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--deg", type=float, default=2.0)
    ap.add_argument("--out", default="exports/frame_parallel_check.json")
    args = ap.parse_args()

    cfg = args.config
    params = parameters.load_parameters(cfg, {})

    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)

    builder = PanelBuilder(params, document=None)

    # Precompute plane data + outer loop per panel.
    panel_plane: Dict[int, object] = {}
    panel_outer: Dict[int, List[Tuple[float, float]]] = {}
    panel_nodes_order: Dict[int, Tuple[int, ...]] = {}

    for panel in dome.panels:
        plane = builder._panel_plane_data(dome, panel)  # type: ignore[attr-defined]
        if plane is None:
            continue

        base_loop = plane.coords2d[:]  # type: ignore[attr-defined]
        inset = max(0.0, params.panel_frame_inset_m)
        outer_loop = base_loop
        outer_inset = inset
        if outer_inset > 1e-9:
            outer_loop = []
            for _ in range(10):
                candidate = builder._validated_inset(base_loop, outer_inset)  # type: ignore[attr-defined]
                if candidate and len(candidate) >= 3:
                    outer_loop = candidate
                    break
                outer_inset *= 0.85
        if not outer_loop or len(outer_loop) < 3:
            continue
        if builder._polygon_area_2d(outer_loop) < 0:  # type: ignore[attr-defined]
            outer_loop = list(reversed(outer_loop))

        panel_plane[panel.index] = plane
        panel_outer[panel.index] = outer_loop
        panel_nodes_order[panel.index] = tuple(plane.node_indices)  # type: ignore[attr-defined]

    # Build adjacency map: ordered edge -> [(panel_idx, edge_local_index), ...]
    edge_to_panels: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for panel in dome.panels:
        nodes_order = panel_nodes_order.get(panel.index, panel.node_indices)
        n = len(nodes_order)
        for i in range(n):
            a = nodes_order[i]
            b = nodes_order[(i + 1) % n]
            e = _ordered_edge(a, b)
            edge_to_panels.setdefault(e, []).append((panel.index, i))

    tol_rad = math.radians(float(args.deg))
    cos_tol = math.cos(tol_rad)

    failures = []
    missing = []
    checked = 0

    for (u, v), refs in edge_to_panels.items():
        if len(refs) != 2:
            continue
        (p_idx, p_edge_i), (q_idx, q_edge_i) = refs

        p_panel = dome.panels[p_idx]
        q_panel = dome.panels[q_idx]
        # "truncation panel" heuristic: non-triangular polygon.
        if len(p_panel.node_indices) == 3 and len(q_panel.node_indices) == 3:
            continue

        p_plane = panel_plane.get(p_idx)
        q_plane = panel_plane.get(q_idx)
        p_outer = panel_outer.get(p_idx)
        q_outer = panel_outer.get(q_idx)
        if p_plane is None or q_plane is None or p_outer is None or q_outer is None:
            missing.append({"edge": [u, v], "panels": [p_idx, q_idx], "reason": "missing_plane_or_outer"})
            continue

        # Compute target 2D direction for the shared edge in each panel.
        p_coords = p_plane.coords2d  # type: ignore[attr-defined]
        q_coords = q_plane.coords2d  # type: ignore[attr-defined]

        p_n = len(p_coords)
        q_n = len(q_coords)
        p_a = p_coords[p_edge_i]
        p_b = p_coords[(p_edge_i + 1) % p_n]
        q_a = q_coords[q_edge_i]
        q_b = q_coords[(q_edge_i + 1) % q_n]

        p_dir2 = _normalize2((p_b[0] - p_a[0], p_b[1] - p_a[1]))
        q_dir2 = _normalize2((q_b[0] - q_a[0], q_b[1] - q_a[1]))
        if p_dir2 is None or q_dir2 is None:
            missing.append({"edge": [u, v], "panels": [p_idx, q_idx], "reason": "degenerate_edge"})
            continue

        def best_segment_dir(outer: List[Tuple[float, float]], target: Tuple[float, float]) -> Tuple[float, float] | None:
            best = None
            best_dot = -1.0
            for i in range(len(outer)):
                a2 = outer[i]
                b2 = outer[(i + 1) % len(outer)]
                d = _normalize2((b2[0] - a2[0], b2[1] - a2[1]))
                if d is None:
                    continue
                dot = abs(d[0] * target[0] + d[1] * target[1])
                if dot > best_dot:
                    best_dot = dot
                    best = d
            # Require it to be reasonably parallel to the target.
            if best is None or best_dot < 0.99:
                return None
            return best

        p_seg2 = best_segment_dir(p_outer, p_dir2)
        q_seg2 = best_segment_dir(q_outer, q_dir2)
        if p_seg2 is None or q_seg2 is None:
            missing.append({"edge": [u, v], "panels": [p_idx, q_idx], "reason": "no_matching_outer_segment"})
            continue

        p_axis_u = p_plane.axis_u  # type: ignore[attr-defined]
        p_axis_v = p_plane.axis_v  # type: ignore[attr-defined]
        q_axis_u = q_plane.axis_u  # type: ignore[attr-defined]
        q_axis_v = q_plane.axis_v  # type: ignore[attr-defined]

        p_dir3 = _lift_dir(p_axis_u, p_axis_v, p_seg2)
        q_dir3 = _lift_dir(q_axis_u, q_axis_v, q_seg2)
        if p_dir3 is None or q_dir3 is None:
            missing.append({"edge": [u, v], "panels": [p_idx, q_idx], "reason": "lift_failed"})
            continue

        checked += 1
        dot = abs(_dot3(p_dir3, q_dir3))
        if dot < cos_tol:
            angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            failures.append(
                {
                    "edge": [u, v],
                    "panels": [p_idx, q_idx],
                    "panel_sides": [len(p_panel.node_indices), len(q_panel.node_indices)],
                    "abs_dot": dot,
                    "angle_deg": angle,
                }
            )

    report = {
        "config": cfg,
        "truncation_enabled": bool(params.use_truncation and params.truncation_ratio > 0),
        "tolerance_deg": float(args.deg),
        "checked_shared_edges": checked,
        "failures": failures,
        "missing": missing,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"checked={checked} failures={len(failures)} missing={len(missing)} report={out_path}"
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
