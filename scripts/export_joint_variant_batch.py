#!/usr/bin/env python3
"""Batch-export panel frame cut-planes for multiple joint variants.

Goal
- Pick a target coverage quantile (e.g. p95) for node compatibility clearance.
- Compute a relief depth per construction approach:
  - panel_inset: needed_relief_m = spread_max_m/2 (independent of relief)
  - node_hub: fixed-point on quantile of anchor_error_max_m(relief)
- Run `scripts/export_panel_frame_cutplanes.py` for each variant and write a
  small manifest describing what was exported.

This stays pure-Python (no FreeCAD).

Usage:
  python3 scripts/export_joint_variant_batch.py --config configs/base.json --q 0.95

Env overrides (same as other scripts):
- DOME_PANEL_FRAME_INSET_M
- DOME_PANEL_FRAME_PROFILE_M (WIDTH,HEIGHT)

Output:
- Cut-planes JSONs in exports/panel_frame_cutplanes/ (or --out-dir)
- Manifest JSON in exports/joint_variant_batches/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_batch(
    *,
    config: str,
    out_dir: str = "exports/panel_frame_cutplanes",
    q: float = 0.95,
    tol: float = 1e-5,
    max_iter: int = 40,
    also_node_hub_points: bool = False,
    inset_env: str | None = None,
    profile_env: str | None = None,
) -> Path:
    """Run the batch export and return the manifest path."""
    if inset_env is None:
        inset_env = os.environ.get("DOME_PANEL_FRAME_INSET_M")
    if profile_env is None:
        profile_env = os.environ.get("DOME_PANEL_FRAME_PROFILE_M")

    inset_override = float(inset_env) if inset_env not in (None, "") else None
    profile_override = _parse_profile(profile_env)

    data = _compute_node_data(config, inset_override, profile_override)
    spreads = data["spreads"]

    panel_needed_values = [s * 0.5 for s in spreads]
    panel_relief_q = float(_percentile(panel_needed_values, q))

    node_hub_fp = _node_hub_quantile_fixed_point(
        items_by_node=data["items_by_node"],
        hubs_by_node=data["hubs_by_node"],
        q=q,
        max_iter=int(max_iter),
        tol=float(tol),
    )
    node_hub_relief_q = float(node_hub_fp["relief_depth_m"])

    exports: list[dict[str, object]] = []
    exports.append({"construction": "panel_inset", "corner": "miter", "relief_depth_m": 0.0})
    exports.append({"construction": "node_hub", "corner": "miter", "relief_depth_m": 0.0})
    exports.append({"construction": "panel_inset", "corner": "relief", "relief_depth_m": panel_relief_q})
    exports.append({"construction": "node_hub", "corner": "relief", "relief_depth_m": node_hub_relief_q})

    out_paths: list[dict[str, object]] = []
    for spec in exports:
        report_path = _run_export_cutplanes(
            config=config,
            out_dir=out_dir,
            construction=str(spec["construction"]),
            corner=str(spec["corner"]),
            relief_depth_m=float(spec["relief_depth_m"]),
            inset_env=inset_env,
            profile_env=profile_env,
        )
        out_paths.append({**spec, "cutplanes": report_path})

        if also_node_hub_points and str(spec["construction"]) == "node_hub":
            env = dict(os.environ)
            env["DOME_CONFIG"] = config
            if inset_env is not None:
                env["DOME_PANEL_FRAME_INSET_M"] = inset_env
            if profile_env is not None:
                env["DOME_PANEL_FRAME_PROFILE_M"] = profile_env
            env["DOME_JOINT_CONSTRUCTION"] = str(spec["construction"])
            env["DOME_JOINT_CORNER"] = str(spec["corner"])
            env["DOME_JOINT_RELIEF_DEPTH_M"] = str(float(spec["relief_depth_m"]))
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "prototype_node_hub_points.py"),
                "--config",
                config,
            ]
            proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"hub points export failed: {proc.stderr.strip() or proc.stdout.strip()}")

    cfg_hash = hashlib.sha1(str(Path(config)).encode("utf-8")).hexdigest()[:8]
    batch_dir = REPO_ROOT / "exports" / "joint_variant_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / (
        f"joint_variant_batch__{_slug(Path(config).stem)}__{cfg_hash}__"
        f"q{_fmt_float_for_name(q)}__inset{_fmt_float_for_name(float(data['params']['inset_m']))}__"
        f"pw{_fmt_float_for_name(float(data['params']['profile_width_m']))}__ph{_fmt_float_for_name(float(data['params']['profile_height_m']))}.json"
    )

    manifest = {
        "config": config,
        "q": q,
        "params": data["params"],
        "selected": {
            "panel_inset_relief_depth_m": panel_relief_q,
            "node_hub_relief_depth_m": node_hub_relief_q,
        },
        "exports": out_paths,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


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


def _bisector_dir_2d(prev2d: tuple[float, float], c2d: tuple[float, float], next2d: tuple[float, float]) -> tuple[float, float] | None:
    d1 = _norm2((prev2d[0] - c2d[0], prev2d[1] - c2d[1]))
    d2 = _norm2((next2d[0] - c2d[0], next2d[1] - c2d[1]))
    if d1 is None or d2 is None:
        return None
    bis = _norm2((d1[0] + d2[0], d1[1] + d2[1]))
    if bis is None:
        bis = _norm2((-d2[1], d2[0]))
    return bis


def _lift_dir_3d(bis2: tuple[float, float], axis_u: tuple[float, float, float], axis_v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    bis3 = (
        axis_u[0] * bis2[0] + axis_v[0] * bis2[1],
        axis_u[1] * bis2[0] + axis_v[1] * bis2[1],
        axis_u[2] * bis2[0] + axis_v[2] * bis2[1],
    )
    return _norm3(bis3)


def _compute_node_data(config: str, inset_override: float | None, profile_override: tuple[float, float] | None):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from freecad_dome import icosahedron, parameters, tessellation
    from freecad_dome.panels import PanelBuilder

    overrides: dict[str, object] = {
        "generate_panel_frames": True,
        "generate_panel_faces": False,
        "generate_struts": False,
    }
    if inset_override is not None:
        overrides["panel_frame_inset_m"] = float(inset_override)
    if profile_override is not None:
        overrides["panel_frame_profile_width_m"] = float(profile_override[0])
        overrides["panel_frame_profile_height_m"] = float(profile_override[1])

    params = parameters.load_parameters(config, overrides)
    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)
    builder = PanelBuilder(params)

    width = float(params.panel_frame_profile_width_m)
    inset = float(params.panel_frame_inset_m)
    inner_inset = max(0.0, inset) + max(0.0, width)

    # Per-node: list of incident items {p, bis} and hub point.
    items_by_node: dict[int, list[dict[str, object]]] = {}

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
            idx = int(node_idx)
            p3 = builder._lift_point(plane, inner_loop[i])  # type: ignore[attr-defined]

            prev2d = inner_loop[(i - 1) % len(inner_loop)]
            c2d = inner_loop[i]
            next2d = inner_loop[(i + 1) % len(inner_loop)]
            bis2 = _bisector_dir_2d(prev2d, c2d, next2d)
            bis3 = _lift_dir_3d(bis2, plane.axis_u, plane.axis_v) if bis2 is not None else None
            if bis3 is None:
                bis3 = (0.0, 0.0, 0.0)

            items_by_node.setdefault(idx, []).append({"p": p3, "bis": bis3})

    hubs_by_node: dict[int, tuple[float, float, float]] = {}
    spreads: list[float] = []
    for node_id, items in items_by_node.items():
        pts = [it["p"] for it in items]  # type: ignore[misc]
        hub = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
            sum(p[2] for p in pts) / len(pts),
        )
        hubs_by_node[node_id] = hub
        spreads.append(_max_pairwise_distance(pts))

    return {
        "params": {
            "inset_m": inset,
            "profile_width_m": width,
            "profile_height_m": float(params.panel_frame_profile_height_m),
            "nodes": int(len(dome.nodes)),
            "panels": int(len(dome.panels)),
        },
        "items_by_node": items_by_node,
        "hubs_by_node": hubs_by_node,
        "spreads": spreads,
    }


def _node_hub_quantile_fixed_point(
    *,
    items_by_node: dict[int, list[dict[str, object]]],
    hubs_by_node: dict[int, tuple[float, float, float]],
    q: float,
    max_iter: int,
    tol: float,
) -> dict[str, float]:
    rd = 0.0
    for _ in range(max_iter):
        values: list[float] = []
        for node_id, items in items_by_node.items():
            hub = hubs_by_node[node_id]
            node_max = 0.0
            for it in items:
                p = it["p"]  # type: ignore[index]
                bis = it["bis"]  # type: ignore[index]
                bis_t = (float(bis[0]), float(bis[1]), float(bis[2]))
                anchor = _add3(hub, _mul3(bis_t, rd)) if rd > 0 else hub
                d = _dist(anchor, p)  # type: ignore[arg-type]
                if d > node_max:
                    node_max = d
            values.append(float(node_max))
        target = float(_percentile(values, q))
        if abs(target - rd) <= tol:
            return {"relief_depth_m": float(target), "metric_p": float(target)}
        rd = target
    return {"relief_depth_m": float(rd), "metric_p": float(rd)}


def _run_export_cutplanes(
    *,
    config: str,
    out_dir: str,
    construction: str,
    corner: str,
    relief_depth_m: float,
    inset_env: str | None,
    profile_env: str | None,
) -> str:
    env = dict(os.environ)
    env["DOME_CONFIG"] = config
    if inset_env is not None:
        env["DOME_PANEL_FRAME_INSET_M"] = inset_env
    if profile_env is not None:
        env["DOME_PANEL_FRAME_PROFILE_M"] = profile_env
    env["DOME_JOINT_CONSTRUCTION"] = construction
    env["DOME_JOINT_CORNER"] = corner
    env["DOME_JOINT_RELIEF_DEPTH_M"] = str(float(relief_depth_m))

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_panel_frame_cutplanes.py"),
        "--config",
        config,
        "--out-dir",
        out_dir,
    ]
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"export failed: {proc.stderr.strip() or proc.stdout.strip()}")
    # Parse last 'report=' token.
    out = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if "report=" in out:
        return out.split("report=", 1)[1].strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default=os.environ.get("DOME_CONFIG", "configs/base.json"))
    parser.add_argument("--out-dir", default=os.environ.get("DOME_OUT_DIR", "exports/panel_frame_cutplanes"))
    parser.add_argument("--q", type=float, default=0.95, help="Target coverage quantile (0-1), e.g. 0.95")
    parser.add_argument("--tol", type=float, default=1e-5, help="Fixed-point tolerance (m)")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument(
        "--also-node-hub-points",
        action="store_true",
        help="Also run prototype_node_hub_points.py for node_hub variants",
    )
    args = parser.parse_args()

    manifest_path = run_batch(
        config=str(args.config),
        out_dir=str(args.out_dir),
        q=float(args.q),
        tol=float(args.tol),
        max_iter=int(args.max_iter),
        also_node_hub_points=bool(args.also_node_hub_points),
    )
    print(f"exports=4 manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
