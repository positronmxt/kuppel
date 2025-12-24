#!/usr/bin/env python3
"""Run the dome generator inside FreeCADCmd and sanity-check the produced strut solids.

Intended to be executed via FreeCADCmd console mode:

  snap run --shell freecad.cmd -c 'FreeCADCmd --console "import runpy; runpy.run_path(\"scripts/freecad_check_outputs.py\", run_name=\"__main__\")"'

Config can be overridden with env vars:
- DOME_CONFIG (default: configs/base.json)
- DOME_OUT_DIR (default: exports/run_check)

The check flags struts whose solid extent along StrutDirection is < 90% of the
manifest length.
"""

from __future__ import annotations

import json
import logging
import os
import runpy
import sys
from pathlib import Path


def _run_generator(config: str, out_dir: str) -> None:
    argv = [
        "scripts/generate_dome.py",
        "--config",
        config,
        "--no-gui",
        "--out-dir",
        out_dir,
        "--skip-ifc",
        "--skip-stl",
        "--skip-dxf",
    ]
    sys.argv = argv
    runpy.run_path("scripts/generate_dome.py", run_name="__main__")


def _extent_along_direction(shape, direction) -> float | None:
    bb = getattr(shape, "BoundBox", None)
    if bb is None:
        return None
    try:
        xs = (bb.XMin, bb.XMax)
        ys = (bb.YMin, bb.YMax)
        zs = (bb.ZMin, bb.ZMax)
    except Exception:
        return None

    # Project the 8 bounding-box corners onto the direction vector.
    projs = []
    for x in xs:
        for y in ys:
            for z in zs:
                projs.append((x * direction.x) + (y * direction.y) + (z * direction.z))
    return max(projs) - min(projs) if projs else None


def main() -> int:
    config = os.environ.get("DOME_CONFIG", "configs/base.json")
    out_dir = os.environ.get("DOME_OUT_DIR", "exports/run_check")

    # Pre-configure logging to avoid extremely verbose INFO logs from the generator.
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    _run_generator(config, out_dir)

    try:
        import FreeCAD  # type: ignore
    except Exception as exc:
        print(f"FreeCAD import failed: {exc}")
        return 2

    doc = FreeCAD.ActiveDocument
    if doc is None:
        print("No ActiveDocument after generation")
        return 2

    manifest_path = Path(out_dir) / "dome_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {m["name"]: float(m["length"]) for m in manifest}

    ifc_objs = [o for o in getattr(doc, "Objects", []) if hasattr(o, "IfcType")]
    has_dir = sum(1 for o in ifc_objs if hasattr(o, "StrutDirection"))
    dir_lengths: list[float] = []
    for o in ifc_objs:
        if not hasattr(o, "StrutDirection"):
            continue
        d = o.StrutDirection
        try:
            ln = (d.x * d.x + d.y * d.y + d.z * d.z) ** 0.5
        except Exception:
            continue
        dir_lengths.append(float(ln))
    dir_zero = sum(1 for ln in dir_lengths if ln < 1e-9)
    dir_min = min(dir_lengths) if dir_lengths else None
    dir_max = max(dir_lengths) if dir_lengths else None
    match = 0
    checked = 0
    undersized: list[tuple[str, float, float]] = []
    samples: list[dict[str, float | str | None]] = []

    for obj in ifc_objs:
        label = str(getattr(obj, "Label", ""))
        exp_len = expected.get(label)
        if exp_len is None:
            continue
        match += 1

        if not hasattr(obj, "StrutDirection"):
            continue
        d = obj.StrutDirection
        ln = (d.x * d.x + d.y * d.y + d.z * d.z) ** 0.5
        if ln < 1e-9:
            continue
        d = FreeCAD.Vector(d.x / ln, d.y / ln, d.z / ln)

        shape = getattr(obj, "Shape", None)
        if shape is None or shape.isNull():
            undersized.append((label, exp_len, 0.0))
            continue

        extent = _extent_along_direction(shape, d)
        if len(samples) < 5:
            bb = getattr(shape, "BoundBox", None)
            samples.append(
                {
                    "name": label,
                    "expected": exp_len,
                    "extent": extent,
                    "bb_x": getattr(bb, "XMax", 0.0) - getattr(bb, "XMin", 0.0) if bb else None,
                    "bb_y": getattr(bb, "YMax", 0.0) - getattr(bb, "YMin", 0.0) if bb else None,
                    "bb_z": getattr(bb, "ZMax", 0.0) - getattr(bb, "ZMin", 0.0) if bb else None,
                }
            )
        if extent is None:
            continue
        checked += 1

        if extent < 0.90 * exp_len:
            undersized.append((label, exp_len, extent))

    report = {
        "config": config,
        "out_dir": out_dir,
        "manifest": len(expected),
        "ifc_objs": len(ifc_objs),
        "has_dir": has_dir,
        "dir_len_min": dir_min,
        "dir_len_max": dir_max,
        "dir_len_zero": dir_zero,
        "matched": match,
        "checked": checked,
        "undersized": [
            {"name": name, "expected": exp_len, "extent": extent}
            for (name, exp_len, extent) in undersized
        ],
        "samples": samples,
    }
    report_path = Path(out_dir) / "solid_check.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"manifest={len(expected)} ifc_objs={len(ifc_objs)} has_dir={has_dir} matched={match} checked={checked} undersized={len(undersized)} report={report_path}"
    )

    return 1 if undersized else 0


if __name__ == "__main__":
    raise SystemExit(main())
