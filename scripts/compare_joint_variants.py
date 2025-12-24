#!/usr/bin/env python3
"""Compare joint/frame variants with node-level metrics (pure-Python).

This script quantifies the core manufacturing problem:
- Each node is shared by multiple panel planes.
- Panel-plane inset produces *different* inner-corner 3D points per incident panel.
- A node-hub strategy can pick a single hub point; the required "pull" distances
  (residuals) show how far panel-derived corners disagree.

We compute, per node:
- spread_max_m: max pairwise distance between incident inner-corner points
- residual_max_m: max distance from hub_point (mean) to incident points

Then, for each requested JointVariant we summarize distributions and whether a
chosen RELIEF depth is at least residual_max_m (a conservative clearance proxy).

Usage:
  python3 scripts/compare_joint_variants.py --config configs/base.json

Env:
- DOME_PANEL_FRAME_INSET_M (optional, overrides config)
- DOME_PANEL_FRAME_PROFILE_M (optional, "WIDTH,HEIGHT" overrides config)

Output:
- Writes a JSON report into exports/joint_variant_metrics/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _slug(value: str) -> str:
    out: list[str] = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_.")
    return s or "config"


def _fmt_float_for_name(value: float) -> str:
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m").replace(".", "p")
    return s or "0"


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _max_pairwise_distance(points: list[tuple[float, float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    m = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = _dist(points[i], points[j])
            if d > m:
                m = d
    return m


def _norm2(v: tuple[float, float]) -> tuple[float, float] | None:
    x, y = v
    ln = math.hypot(x, y)
    if ln <= 1e-18:
        return None
    return (x / ln, y / ln)


def _norm3(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    x, y, z = v
    ln = (x * x + y * y + z * z) ** 0.5
    if ln <= 1e-18:
        return None
    return (x / ln, y / ln, z / ln)


def _add3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul3(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _bisector_dir_2d(
    prev2d: tuple[float, float],
    c2d: tuple[float, float],
    next2d: tuple[float, float],
) -> tuple[float, float] | None:
    d1 = _norm2((prev2d[0] - c2d[0], prev2d[1] - c2d[1]))
    d2 = _norm2((next2d[0] - c2d[0], next2d[1] - c2d[1]))
    if d1 is None or d2 is None:
        return None
    bis = _norm2((d1[0] + d2[0], d1[1] + d2[1]))
    if bis is None:
        bis = _norm2((-d2[1], d2[0]))
    return bis


def _lift_dir_3d(
    bis2: tuple[float, float],
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    bis3 = (
        axis_u[0] * bis2[0] + axis_v[0] * bis2[1],
        axis_u[1] * bis2[0] + axis_v[1] * bis2[1],
        axis_u[2] * bis2[0] + axis_v[2] * bis2[1],
    )
    return _norm3(bis3)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def main() -> int:
    from freecad_dome import icosahedron, parameters, tessellation
    from freecad_dome.joint_variants import CornerTreatment, FrameConstruction, JointVariant
    from freecad_dome.panels import PanelBuilder

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        default=os.environ.get("DOME_CONFIG", "configs/base.json"),
        help="Config JSON path (default: env DOME_CONFIG or configs/base.json)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("DOME_OUT_DIR", "exports/joint_variant_metrics"),
        help="Output directory (default: env DOME_OUT_DIR or exports/joint_variant_metrics)",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Variant spec 'construction,corner,relief_depth_m'. "
            "Example: node_hub,miter,0 or panel_inset,relief,0.01. "
            "May be repeated; if omitted, runs a small default set."
        ),
    )
    args = parser.parse_args()

    config = str(args.config)
    out_dir = Path(str(args.out_dir))

    inset_raw = os.environ.get("DOME_PANEL_FRAME_INSET_M")
    profile_raw = os.environ.get("DOME_PANEL_FRAME_PROFILE_M")
    profile = _parse_profile(profile_raw)

    overrides: dict[str, object] = {
        "generate_panel_frames": True,
        "generate_panel_faces": False,
        "generate_struts": False,
    }
    if inset_raw not in (None, ""):
        overrides["panel_frame_inset_m"] = float(inset_raw)
    if profile is not None:
        overrides["panel_frame_profile_width_m"] = float(profile[0])
        overrides["panel_frame_profile_height_m"] = float(profile[1])

    params = parameters.load_parameters(config, overrides)

    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)
    builder = PanelBuilder(params)

    width = float(params.panel_frame_profile_width_m)
    inset = float(params.panel_frame_inset_m)
    inner_inset = max(0.0, inset) + max(0.0, width)

    inner_points_by_node: dict[int, list[tuple[float, float, float]]] = {}
    incident_by_node: dict[int, list[int]] = {}
    incident_items_by_node: dict[int, list[dict[str, object]]] = {}

    for panel in dome.panels:
        plane = builder._panel_plane_data(dome, panel)  # type: ignore[attr-defined]
        if plane is None:
            continue
        base_loop = list(plane.coords2d)
        if len(base_loop) < 3:
            continue
        inner_loop = builder._validated_inset(base_loop, inner_inset)  # type: ignore[attr-defined]
        if not inner_loop or len(inner_loop) != len(base_loop):
            continue
        for i, node_idx in enumerate(plane.node_indices):
            p3 = builder._lift_point(plane, inner_loop[i])  # type: ignore[attr-defined]
            idx = int(node_idx)
            inner_points_by_node.setdefault(idx, []).append(p3)
            incident_by_node.setdefault(idx, []).append(int(panel.index))

            prev2d = inner_loop[(i - 1) % len(inner_loop)]
            c2d = inner_loop[i]
            next2d = inner_loop[(i + 1) % len(inner_loop)]
            bis2 = _bisector_dir_2d(prev2d, c2d, next2d)
            bis3 = _lift_dir_3d(bis2, plane.axis_u, plane.axis_v) if bis2 is not None else None
            if bis3 is None:
                bis3 = (0.0, 0.0, 0.0)
            incident_items_by_node.setdefault(idx, []).append({"panel": int(panel.index), "p": p3, "bis": bis3})

    nodes_out: list[dict[str, object]] = []
    residual_max_values: list[float] = []
    spread_max_values: list[float] = []

    for node_idx, pts in inner_points_by_node.items():
        if len(pts) == 0:
            continue
        hub = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
            sum(p[2] for p in pts) / len(pts),
        )
        residuals = [_dist(hub, p) for p in pts]
        residual_max = max(residuals) if residuals else 0.0
        spread_max = _max_pairwise_distance(pts)
        residual_max_values.append(float(residual_max))
        spread_max_values.append(float(spread_max))
        nodes_out.append(
            {
                "node": int(node_idx),
                "incident_count": int(len(pts)),
                "incident_panels": incident_by_node.get(node_idx, []),
                "hub_point": [float(hub[0]), float(hub[1]), float(hub[2])],
                "residual_min_m": float(min(residuals)) if residuals else 0.0,
                "residual_max_m": float(residual_max),
                "residual_avg_m": float(sum(residuals) / len(residuals)) if residuals else 0.0,
                "spread_max_m": float(spread_max),
            }
        )

    nodes_out.sort(key=lambda n: float(n["residual_max_m"]), reverse=True)

    # Variants requested
    variants: list[JointVariant] = []
    if args.variant:
        for spec in args.variant:
            parts = [p.strip() for p in str(spec).split(",")]
            if len(parts) not in (2, 3):
                raise ValueError("--variant must be 'construction,corner[,relief_depth_m]'")
            construction = FrameConstruction(parts[0].lower().replace("-", "_"))
            corner = CornerTreatment(parts[1].lower().replace("-", "_"))
            relief = float(parts[2]) if len(parts) == 3 else 0.0
            v = JointVariant(construction=construction, corner=corner, relief_depth_m=relief)
            v.validate()
            variants.append(v)
    else:
        variants = [
            JointVariant(FrameConstruction.PANEL_INSET, CornerTreatment.MITER, 0.0),
            JointVariant(FrameConstruction.PANEL_INSET, CornerTreatment.RELIEF, 0.01),
            JointVariant(FrameConstruction.NODE_HUB, CornerTreatment.MITER, 0.0),
            JointVariant(FrameConstruction.NODE_HUB, CornerTreatment.RELIEF, 0.01),
        ]

    variant_summaries: list[dict[str, object]] = []
    for v in variants:
        rd = float(v.relief_depth_m) if v.corner == CornerTreatment.RELIEF else 0.0

        if v.construction == FrameConstruction.PANEL_INSET:
            # Conservative: if each incident panel/member gets relieved by rd, then any
            # pairwise disagreement up to ~2*rd can be cleared.
            metric_values = [float(x) * 0.5 for x in spread_max_values]
            metric_name = "needed_relief_m"
            covered = sum(1 for x in metric_values if x <= rd + 1e-12)
            required_all = float(max(metric_values)) if metric_values else 0.0
        else:
            # Node-hub: anchor each incident point at hub + bisector*rd and measure remaining
            # mismatch (anchor_error_max) to the panel-derived inner corner points.
            anchor_error_max: list[float] = []
            for node_rec in nodes_out:
                node_id = int(node_rec["node"])  # type: ignore[arg-type]
                hub_list = node_rec["hub_point"]  # type: ignore[index]
                hub = (float(hub_list[0]), float(hub_list[1]), float(hub_list[2]))
                items = incident_items_by_node.get(node_id, [])
                node_max = 0.0
                for it in items:
                    p = it["p"]  # type: ignore[index]
                    bis = it["bis"]  # type: ignore[index]
                    bis_t = (float(bis[0]), float(bis[1]), float(bis[2]))
                    anchor = _add3(hub, _mul3(bis_t, rd)) if rd > 0 else hub
                    d = _dist(anchor, p)  # type: ignore[arg-type]
                    if d > node_max:
                        node_max = d
                anchor_error_max.append(float(node_max))

            metric_values = anchor_error_max
            metric_name = "anchor_error_max_m"
            covered = sum(1 for x in metric_values if x <= rd + 1e-12)
            required_all = float(max(metric_values)) if metric_values else 0.0

        variant_summaries.append(
            {
                "variant": {
                    "construction": str(v.construction.value),
                    "corner": str(v.corner.value),
                    "relief_depth_m": float(v.relief_depth_m),
                },
                "nodes_total": int(len(metric_values)),
                "coverage_nodes": int(covered),
                "coverage_ratio": float(covered / len(metric_values)) if metric_values else 1.0,
                "coverage_metric": metric_name,
                "coverage_metric_stats_m": {
                    "min": float(min(metric_values)) if metric_values else 0.0,
                    "p50": float(_percentile(metric_values, 0.50)),
                    "p90": float(_percentile(metric_values, 0.90)),
                    "p95": float(_percentile(metric_values, 0.95)),
                    "p99": float(_percentile(metric_values, 0.99)),
                    "max": float(max(metric_values)) if metric_values else 0.0,
                },
                "required_relief_depth_for_coverage_m": {
                    "p50": float(_percentile(metric_values, 0.50)),
                    "p90": float(_percentile(metric_values, 0.90)),
                    "p95": float(_percentile(metric_values, 0.95)),
                    "p99": float(_percentile(metric_values, 0.99)),
                    "p100": float(max(metric_values)) if metric_values else 0.0,
                },
                "required_relief_depth_for_full_coverage_m": float(required_all),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Path(config)
    cfg_hash = hashlib.sha1(str(cfg).encode("utf-8")).hexdigest()[:8]
    out_path = out_dir / (
        f"joint_variant_metrics__{_slug(cfg.stem)}__{cfg_hash}__"
        f"inset{_fmt_float_for_name(inset)}__pw{_fmt_float_for_name(width)}__ph{_fmt_float_for_name(float(params.panel_frame_profile_height_m))}.json"
    )

    report = {
        "config": config,
        "inset_m": inset,
        "profile_width_m": width,
        "profile_height_m": float(params.panel_frame_profile_height_m),
        "nodes": int(len(dome.nodes)),
        "panels": int(len(dome.panels)),
        "computed_nodes": int(len(nodes_out)),
        "metrics": {
            "residual_max_m": {
                "min": float(min(residual_max_values)) if residual_max_values else 0.0,
                "p50": float(_percentile(residual_max_values, 0.50)),
                "p90": float(_percentile(residual_max_values, 0.90)),
                "p95": float(_percentile(residual_max_values, 0.95)),
                "max": float(max(residual_max_values)) if residual_max_values else 0.0,
            },
            "spread_max_m": {
                "min": float(min(spread_max_values)) if spread_max_values else 0.0,
                "p50": float(_percentile(spread_max_values, 0.50)),
                "p90": float(_percentile(spread_max_values, 0.90)),
                "p95": float(_percentile(spread_max_values, 0.95)),
                "max": float(max(spread_max_values)) if spread_max_values else 0.0,
            },
        },
        "variants": variant_summaries,
        "worst_nodes": nodes_out[:50],
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"nodes={report['computed_nodes']} variants={len(variants)} report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
