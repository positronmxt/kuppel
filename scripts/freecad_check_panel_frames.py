#!/usr/bin/env python3
"""Run the dome generator inside FreeCADCmd and sanity-check produced panel frames.

Intended to be executed via FreeCADCmd console mode:

  snap run --shell freecad.cmd -c 'FreeCADCmd --console "import runpy; runpy.run_path(\"scripts/freecad_check_panel_frames.py\", run_name=\"__main__\")"'

Config can be overridden with env vars:
- DOME_CONFIG (default: configs/base.json)
- DOME_OUT_DIR (default: exports/panel_frame_check)
Optional panel-frame overrides:
- DOME_PANEL_FRAME_INSET_M (float meters)
- DOME_PANEL_FRAME_PROFILE_M ("WIDTH,HEIGHT" meters)

The check reports the number of generated PanelFrame_* features and flags any
with null/invalid geometry.
"""

from __future__ import annotations

import json
import logging
import os
import runpy
import sys
from pathlib import Path


def _run_generator(
    config: str,
    out_dir: str,
    inset_m: float | None,
    profile_m: tuple[float, float] | None,
) -> None:
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


def _parse_profile(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    raw = value.replace(" ", "")
    if "," in raw:
        parts = raw.split(",", 1)
    else:
        parts = raw.split("x", 1) if "x" in raw.lower() else raw.split(";", 1)
    if len(parts) != 2:
        raise ValueError("DOME_PANEL_FRAME_PROFILE_M must be 'WIDTH,HEIGHT'")
    return (float(parts[0]), float(parts[1]))


def _expected_panel_count(config: str) -> int:
    from freecad_dome import icosahedron, parameters, tessellation

    params = parameters.load_parameters(config, {"generate_panel_frames": True, "generate_panel_faces": False, "generate_struts": False})
    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)
    return len(dome.panels)


def main() -> int:
    config = os.environ.get("DOME_CONFIG", "configs/base.json")
    out_dir = os.environ.get("DOME_OUT_DIR", "exports/panel_frame_check")
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

    expected_panels = _expected_panel_count(config)

    frames = [
        o
        for o in getattr(doc, "Objects", [])
        if str(getattr(o, "Name", "")).startswith("PanelFrame_")
        or str(getattr(o, "Label", "")).startswith("PanelFrame_")
    ]

    frame_indices: set[int] = set()
    for obj in frames:
        raw = str(getattr(obj, "Name", getattr(obj, "Label", "")))
        if "PanelFrame_" not in raw:
            continue
        try:
            suffix = raw.split("PanelFrame_", 1)[1]
            frame_indices.add(int(suffix))
        except Exception:
            continue

    missing = [
        f"PanelFrame_{idx:04d}"
        for idx in range(expected_panels)
        if idx not in frame_indices
    ]

    bad: list[dict[str, object]] = []
    samples: list[dict[str, float | str | None]] = []
    widths: list[float] = []
    insets: list[float] = []

    for obj in frames:
        shape = getattr(obj, "Shape", None)
        label = str(getattr(obj, "Label", getattr(obj, "Name", "")))
        if hasattr(obj, "FrameProfileWidth"):
            try:
                widths.append(float(obj.FrameProfileWidth))
            except Exception:
                pass
        if hasattr(obj, "FrameInset"):
            try:
                insets.append(float(obj.FrameInset))
            except Exception:
                pass
        reason: str | None = None
        if shape is None or getattr(shape, "isNull", lambda: True)():
            reason = "null"
        elif hasattr(shape, "isValid") and not shape.isValid():
            reason = "invalid"

        if reason is not None:
            bb = getattr(shape, "BoundBox", None) if shape is not None else None
            solids = getattr(shape, "Solids", []) if shape is not None else []
            try:
                volume = float(getattr(shape, "Volume", 0.0)) if shape is not None else 0.0
            except Exception:
                volume = None
            bad.append(
                {
                    "name": label,
                    "reason": reason,
                    "shape_type": getattr(shape, "ShapeType", None) if shape is not None else None,
                    "solids": len(solids) if solids is not None else None,
                    "volume": volume,
                    "bb_x": getattr(bb, "XMax", 0.0) - getattr(bb, "XMin", 0.0) if bb else None,
                    "bb_y": getattr(bb, "YMax", 0.0) - getattr(bb, "YMin", 0.0) if bb else None,
                    "bb_z": getattr(bb, "ZMax", 0.0) - getattr(bb, "ZMin", 0.0) if bb else None,
                }
            )
            continue
        if len(samples) < 5:
            bb = getattr(shape, "BoundBox", None)
            samples.append(
                {
                    "name": label,
                    "bb_x": getattr(bb, "XMax", 0.0) - getattr(bb, "XMin", 0.0) if bb else None,
                    "bb_y": getattr(bb, "YMax", 0.0) - getattr(bb, "YMin", 0.0) if bb else None,
                    "bb_z": getattr(bb, "ZMax", 0.0) - getattr(bb, "ZMin", 0.0) if bb else None,
                }
            )

    report = {
        "config": config,
        "out_dir": out_dir,
        "inset_m": inset_m,
        "profile_m": list(profile_m) if profile_m is not None else None,
        "expected_panels": expected_panels,
        "frames": len(frames),
        "missing": missing,
        "effective_width_min_m": min(widths) if widths else None,
        "effective_width_max_m": max(widths) if widths else None,
        "effective_inset_min_m": min(insets) if insets else None,
        "effective_inset_max_m": max(insets) if insets else None,
        "bad": bad,
        "samples": samples,
    }

    report_path = Path(out_dir) / "panel_frame_check.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"expected_panels={expected_panels} frames={len(frames)} bad={len(bad)} report={report_path}"
    )

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
