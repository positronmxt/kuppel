#!/usr/bin/env python3
"""Export cut-plane definitions for straight panel-frame members.

Goal
- We model each panel frame as straight members along the panel edges.
- Each member end gets a *vertical miter plane* (plane contains panel normal)
  so that, after cutting, members assemble into a closed frame in that panel.

This does NOT solve multi-panel (node) compatibility by itself. It produces the
per-panel member cut data you need for fabrication and later comparison.

Pure-Python (no FreeCAD).

Usage:
  python3 scripts/export_panel_frame_cutplanes.py

Env:
- DOME_CONFIG (default: configs/base.json)
- DOME_OUT_DIR (default: exports/panel_frame_cutplanes)
- DOME_PANEL_FRAME_INSET_M (optional, float meters)
- DOME_PANEL_FRAME_PROFILE_M (optional, "WIDTH,HEIGHT" meters)

Output
- Writes `panel_frame_cutplanes.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
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
        f"panel_frame_cutplanes__{cfg_stem}__{cfg_hash}__"
        f"jc{jc}__cc{cc}__rd{_fmt_float_for_name(relief_depth_m)}__"
        f"inset{_fmt_float_for_name(inset_m)}__"
        f"pw{_fmt_float_for_name(pw_m)}__"
        f"ph{_fmt_float_for_name(ph_m)}.json"
    )
    return out_dir / name


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


def _sub3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mul3(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@dataclass(frozen=True)
class Plane:
    point: tuple[float, float, float]
    normal: tuple[float, float, float]


def _miter_plane_at_corner(
    corner2d: tuple[float, float],
    prev2d: tuple[float, float],
    next2d: tuple[float, float],
    plane_u: tuple[float, float, float],
    plane_v: tuple[float, float, float],
    plane_n: tuple[float, float, float],
    corner3d: tuple[float, float, float],
    *,
    anchor_point3d: tuple[float, float, float] | None = None,
    relief_depth_m: float = 0.0,
) -> Plane | None:
    # Edge directions in 2D (both leaving the corner): prev-corner and next-corner.
    d1 = _norm2((prev2d[0] - corner2d[0], prev2d[1] - corner2d[1]))
    d2 = _norm2((next2d[0] - corner2d[0], next2d[1] - corner2d[1]))
    if d1 is None or d2 is None:
        return None

    # Angle bisector direction in 2D (interior for CCW convex polygons).
    bis2 = _norm2((d1[0] + d2[0], d1[1] + d2[1]))
    if bis2 is None:
        # Nearly straight; use a perpendicular to one edge.
        bis2 = (-d2[1], d2[0])
        bis2 = _norm2(bis2)
    if bis2 is None:
        return None

    # Lift bisector direction into 3D within the panel plane.
    bis3 = _add3(_mul3(plane_u, bis2[0]), _mul3(plane_v, bis2[1]))
    bis3 = _norm3(bis3)
    if bis3 is None:
        return None

    # Miter plane should contain panel normal -> plane normal is perpendicular to both
    # panel normal and bisector direction.
    n = _cross(bis3, plane_n)
    n = _norm3(n)
    if n is None:
        return None

    base_point = anchor_point3d if anchor_point3d is not None else corner3d
    if relief_depth_m > 1e-12:
        base_point = _add3(base_point, _mul3(bis3, float(relief_depth_m)))
    return Plane(point=base_point, normal=n)


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def main() -> int:
    from freecad_dome import icosahedron, parameters, tessellation
    from freecad_dome.panels import PanelBuilder
    from freecad_dome.joint_variants import FrameConstruction, CornerTreatment, load_joint_variant_from_env

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        default=os.environ.get("DOME_CONFIG", "configs/base.json"),
        help="Config JSON path (default: env DOME_CONFIG or configs/base.json)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("DOME_OUT_DIR", "exports/panel_frame_cutplanes"),
        help="Output directory (default: env DOME_OUT_DIR or exports/panel_frame_cutplanes)",
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

    profile = _parse_profile(profile_raw)

    overrides: dict[str, object] = {
        "generate_panel_frames": True,
        "generate_panel_faces": False,
        "generate_struts": False,
    }
    if inset_raw:
        overrides["panel_frame_inset_m"] = float(inset_raw)
    if profile is not None:
        overrides["panel_frame_profile_width_m"] = float(profile[0])
        overrides["panel_frame_profile_height_m"] = float(profile[1])

    params = parameters.load_parameters(config, overrides)

    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)

    builder = PanelBuilder(params)

    joint = load_joint_variant_from_env()

    inset = float(params.panel_frame_inset_m)
    prof_w = float(params.panel_frame_profile_width_m)
    prof_h = float(params.panel_frame_profile_height_m)
    # Member centerline follows the *outer* loop (inset only), because width affects only inner edge.
    outer_inset = max(0.0, inset)

    panels_out: list[dict[str, object]] = []

    hub_by_node: dict[int, tuple[float, float, float]] = {}
    if joint.construction == FrameConstruction.NODE_HUB:
        width = float(params.panel_frame_profile_width_m)
        inner_inset = max(0.0, inset) + max(0.0, width)
        inner_points_by_node: dict[int, list[tuple[float, float, float]]] = {}
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
                inner_points_by_node.setdefault(int(node_idx), []).append(p3)
        for node_idx, pts in inner_points_by_node.items():
            if not pts:
                continue
            c = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
                sum(p[2] for p in pts) / len(pts),
            )
            hub_by_node[int(node_idx)] = c

    for panel in dome.panels:
        plane = builder._panel_plane_data(dome, panel)  # type: ignore[attr-defined]
        if plane is None:
            continue

        base_loop = list(plane.coords2d)
        if len(base_loop) < 3:
            continue

        outer_loop = base_loop
        eff_inset = outer_inset
        if eff_inset > 1e-9:
            outer_loop = []
            eff = eff_inset
            for _ in range(10):
                cand = builder._validated_inset(base_loop, eff)  # type: ignore[attr-defined]
                if cand and len(cand) >= 3:
                    outer_loop = cand
                    eff_inset = eff
                    break
                eff *= 0.85
        if not outer_loop:
            continue

        # Ensure CCW (matches PanelBuilder convention).
        if builder._polygon_area_2d(outer_loop) < 0:  # type: ignore[attr-defined]
            outer_loop = list(reversed(outer_loop))

        corners3d = [builder._lift_point(plane, pt) for pt in outer_loop]  # type: ignore[attr-defined]

        corner_planes: list[dict[str, object]] = []
        members: list[dict[str, object]] = []

        for i in range(len(outer_loop)):
            prev2d = outer_loop[(i - 1) % len(outer_loop)]
            c2d = outer_loop[i]
            next2d = outer_loop[(i + 1) % len(outer_loop)]
            c3d = corners3d[i]
            node_idx = int(plane.node_indices[i])
            hub = hub_by_node.get(node_idx)
            anchor = hub if joint.construction == FrameConstruction.NODE_HUB else None
            pl = _miter_plane_at_corner(
                c2d,
                prev2d,
                next2d,
                plane.axis_u,
                plane.axis_v,
                plane.normal,
                c3d,
                anchor_point3d=anchor,
                relief_depth_m=float(joint.relief_depth_m) if joint.corner == CornerTreatment.RELIEF else 0.0,
            )
            if pl is None:
                continue
            corner_point = c3d
            plane_point = pl.point
            corner_planes.append(
                {
                    "corner": i,
                    "corner_point": [float(corner_point[0]), float(corner_point[1]), float(corner_point[2])],
                    "hub_point": [float(hub[0]), float(hub[1]), float(hub[2])] if hub is not None else None,
                    "point": [float(plane_point[0]), float(plane_point[1]), float(plane_point[2])],
                    "normal": [float(pl.normal[0]), float(pl.normal[1]), float(pl.normal[2])],
                    "construction": str(joint.construction.value),
                    "corner_treatment": str(joint.corner.value),
                    "relief_depth_m": float(joint.relief_depth_m),
                    "anchor_error_m": float(_dist(plane_point, corner_point)),
                }
            )

        # Member definition: edge i from corner i to i+1.
        for i in range(len(corners3d)):
            a = corners3d[i]
            b = corners3d[(i + 1) % len(corners3d)]
            dx, dy, dz = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            members.append(
                {
                    "edge": i,
                    "start_corner": i,
                    "end_corner": (i + 1) % len(corners3d),
                    "start": [float(a[0]), float(a[1]), float(a[2])],
                    "end": [float(b[0]), float(b[1]), float(b[2])],
                    "length_m": float(length),
                    "start_cut_plane_corner": i,
                    "end_cut_plane_corner": (i + 1) % len(corners3d),
                }
            )

        panels_out.append(
            {
                "panel": int(panel.index),
                "corner_count": int(len(corners3d)),
                "effective_inset_m": float(eff_inset),
                "panel_normal": [float(plane.normal[0]), float(plane.normal[1]), float(plane.normal[2])],
                "corner_cut_planes": corner_planes,
                "members": members,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else _auto_out_path(
            out_dir,
            config,
            inset,
            prof_w,
            prof_h,
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
                "profile_m": list(profile) if profile is not None else None,
                "inset_m": inset,
                "effective_profile_width_m": prof_w,
                "effective_profile_height_m": prof_h,
                "panel_count": len(panels_out),
                "panels": panels_out,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"panels={len(panels_out)} report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
