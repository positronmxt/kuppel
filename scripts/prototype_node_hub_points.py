#!/usr/bin/env python3
"""Prototype: compute per-node target points for a node-hub strategy.

This does NOT generate solids yet.

Idea
- For each node, each incident panel produces an inner-corner 3D point (from
  the panel-plane inset logic).
- A node-hub strategy can define a *single* target point for that node.
- Here we compute that target as the average of incident inner-corner points and
  report the residuals (spread).

Env:
- DOME_CONFIG (default: configs/base.json)
- DOME_OUT_DIR (default: exports/node_hub_prototype)
- DOME_PANEL_FRAME_INSET_M (optional)
- DOME_PANEL_FRAME_PROFILE_M (optional, "WIDTH,HEIGHT")

Output:
- Writes `node_hub_points.json`
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _auto_out_path(
    out_dir: Path,
    config_path: str,
    inset_m: float,
    pw_m: float,
    ph_m: float,
    *,
    joint_construction: str,
    joint_corner: str,
    relief_depth_m: float,
) -> Path:
    cfg = Path(config_path)
    cfg_stem = _slug(cfg.stem)
    cfg_hash = hashlib.sha1(str(cfg).encode("utf-8")).hexdigest()[:8]
    jc = _slug(joint_construction)
    cc = _slug(joint_corner)
    name = (
        f"node_hub_points__{cfg_stem}__{cfg_hash}__"
        f"jc{jc}__cc{cc}__rd{_fmt_float_for_name(relief_depth_m)}__"
        f"inset{_fmt_float_for_name(inset_m)}__"
        f"pw{_fmt_float_for_name(pw_m)}__"
        f"ph{_fmt_float_for_name(ph_m)}.json"
    )
    return out_dir / name


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def main() -> int:
    from freecad_dome import icosahedron, parameters, tessellation
    from freecad_dome.panels import PanelBuilder
    from freecad_dome.joint_variants import load_joint_variant_from_env

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        default=os.environ.get("DOME_CONFIG", "configs/base.json"),
        help="Config JSON path (default: env DOME_CONFIG or configs/base.json)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("DOME_OUT_DIR", "exports/node_hub_prototype"),
        help="Output directory (default: env DOME_OUT_DIR or exports/node_hub_prototype)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON file path (overrides --out-dir + auto naming)",
    )
    args = parser.parse_args()

    config = str(args.config)
    out_dir = Path(str(args.out_dir))
    inset_raw = os.environ.get("DOME_PANEL_FRAME_INSET_M")
    profile_raw = os.environ.get("DOME_PANEL_FRAME_PROFILE_M")

    overrides: dict[str, object] = {
        "generate_panel_frames": True,
        "generate_panel_faces": False,
        "generate_struts": False,
    }
    if inset_raw:
        overrides["panel_frame_inset_m"] = float(inset_raw)
    prof = _parse_profile(profile_raw)
    if prof is not None:
        overrides["panel_frame_profile_width_m"] = float(prof[0])
        overrides["panel_frame_profile_height_m"] = float(prof[1])

    params = parameters.load_parameters(config, overrides)
    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)

    builder = PanelBuilder(params)

    joint = load_joint_variant_from_env()

    width = float(params.panel_frame_profile_width_m)
    height = float(params.panel_frame_profile_height_m)
    inset = float(params.panel_frame_inset_m)

    inner_points_by_node: dict[int, list[dict[str, object]]] = {}

    for panel in dome.panels:
        plane = builder._panel_plane_data(dome, panel)  # type: ignore[attr-defined]
        if plane is None:
            continue
        base_loop = list(plane.coords2d)
        if len(base_loop) < 3:
            continue
        inner_inset = max(0.0, inset) + max(0.0, width)
        inner_loop = builder._validated_inset(base_loop, inner_inset)  # type: ignore[attr-defined]
        if not inner_loop or len(inner_loop) != len(base_loop):
            continue
        for i, node_idx in enumerate(plane.node_indices):
            p3 = builder._lift_point(plane, inner_loop[i])  # type: ignore[attr-defined]
            inner_points_by_node.setdefault(int(node_idx), []).append(
                {
                    "panel": int(panel.index),
                    "p": [float(p3[0]), float(p3[1]), float(p3[2])],
                }
            )

    hubs: list[dict[str, object]] = []
    for node_idx, items in inner_points_by_node.items():
        if not items:
            continue
        pts = [tuple(it["p"]) for it in items]  # type: ignore[misc]
        c = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
            sum(p[2] for p in pts) / len(pts),
        )
        residuals = [_dist(c, p) for p in pts]
        hubs.append(
            {
                "node": int(node_idx),
                "hub_point": [float(c[0]), float(c[1]), float(c[2])],
                "incident_panels": [int(it["panel"]) for it in items],
                "residual_min_m": float(min(residuals)) if residuals else 0.0,
                "residual_max_m": float(max(residuals)) if residuals else 0.0,
                "residual_avg_m": float(sum(residuals) / len(residuals)) if residuals else 0.0,
            }
        )

    hubs.sort(key=lambda h: float(h["residual_max_m"]), reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else _auto_out_path(
            out_dir,
            config,
            inset,
            width,
            height,
            joint_construction=str(joint.construction.value),
            joint_corner=str(joint.corner.value),
            relief_depth_m=float(joint.relief_depth_m),
        )
    )
    out_path.write_text(
        json.dumps(
            {
                "config": config,
                "joint_variant": {
                    "construction": str(joint.construction.value),
                    "corner": str(joint.corner.value),
                    "relief_depth_m": float(joint.relief_depth_m),
                },
                "inset_m": inset,
                "profile_width_m": width,
                "profile_height_m": height,
                "nodes": len(dome.nodes),
                "panels": len(dome.panels),
                "hubs": hubs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"nodes={len(hubs)} report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
