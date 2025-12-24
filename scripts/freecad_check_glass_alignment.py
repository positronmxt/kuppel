#!/usr/bin/env python3
"""Run the dome generator inside FreeCADCmd and validate glass panel alignment.

Checks each `GlassPanel_####` against the mathematical panel plane from the
configured tessellation:
- parallelism (angle between normals)
- offset from the panel plane along the panel normal (sphere center reference)

Intended to be executed via FreeCADCmd console mode, e.g.:

  snap run --shell freecad.cmd -c 'FreeCADCmd --console "import runpy; runpy.run_path(\"scripts/freecad_check_glass_alignment.py\", run_name=\"__main__\")"'

Config can be overridden with env vars:
- DOME_CONFIG (default: configs/base.json)
- DOME_OUT_DIR (default: exports/run_check_glass)

Tolerances (env vars, optional):
- GLASS_ANGLE_TOL_DEG (default: 0.2)
- GLASS_OFFSET_TOL_M (default: 0.001)

Output:
- Writes `glass_alignment_check.json` into DOME_OUT_DIR
- Exit code 1 if any panels fail, else 0
"""

from __future__ import annotations

import json
import logging
import math
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


def _truthy(text: str | None) -> bool:
    if text is None:
        return False
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _deg(rad: float) -> float:
    return rad * (180.0 / math.pi)


def _vec_len(v) -> float:
    return float((v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5)


def _vec_unit(v):
    ln = _vec_len(v)
    if ln <= 1e-18:
        return None
    return v.__class__(v.x / ln, v.y / ln, v.z / ln)


def _face_normal(face):
    # Best-effort stable normal for planar faces.
    try:
        u0, u1, v0, v1 = face.ParameterRange
        u = 0.5 * (u0 + u1)
        v = 0.5 * (v0 + v1)
        n = face.normalAt(u, v)
        return n
    except Exception:
        try:
            return face.normalAt(0, 0)
        except Exception:
            return None


def _collect_glass_faces(shape, expected_n):
    faces = list(getattr(shape, "Faces", []) or [])
    if not faces:
        return []

    cand = []
    for f in faces:
        n = _face_normal(f)
        if n is None:
            continue
        un = _vec_unit(n)
        if un is None:
            continue
        d = float(un.dot(expected_n))
        # Keep faces whose normal is near-parallel to the expected normal.
        if abs(d) < 0.98:
            continue
        try:
            area = float(getattr(f, "Area", 0.0))
        except Exception:
            area = 0.0
        cand.append((area, f, un, d))

    cand.sort(key=lambda x: x[0], reverse=True)
    # We expect exactly 2 large planar faces.
    return cand[:2]


def main() -> int:
    config = os.environ.get("DOME_CONFIG", "configs/base.json")
    out_dir = os.environ.get("DOME_OUT_DIR", "exports/run_check_glass")

    angle_tol_deg = float(os.environ.get("GLASS_ANGLE_TOL_DEG", "0.2"))
    offset_tol_m = float(os.environ.get("GLASS_OFFSET_TOL_M", "0.001"))

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

    # Rebuild the tessellation model from config so we compare to the mathematical planes.
    from freecad_dome import icosahedron, parameters, tessellation
    from freecad_dome.panels import PanelBuilder

    overrides, cli = parameters.parse_cli_overrides(["--config", config])
    params = parameters.load_parameters(Path(config), overrides)

    mesh = icosahedron.build_icosahedron(params.radius_m)
    dome = tessellation.tessellate(mesh, params)

    panel_builder = PanelBuilder(params, document=None)

    # Index objects by name for fast lookup.
    by_name = {str(getattr(o, "Name", "")): o for o in list(getattr(doc, "Objects", []) or [])}

    failures = []
    checked = 0

    for panel in dome.panels:
        glass_name = f"GlassPanel_{panel.index:04d}"
        glass_obj = by_name.get(glass_name)
        if glass_obj is None:
            continue

        plane = panel_builder._panel_plane_data(dome, panel)
        if plane is None:
            continue

        n = FreeCAD.Vector(float(plane.normal[0]), float(plane.normal[1]), float(plane.normal[2]))
        n = _vec_unit(n)
        if n is None:
            continue

        # Expected signed offset along panel normal.
        seat = float(panel_builder._glass_seat_offset_m(dome, panel, plane))
        thickness = float(getattr(params, "glass_thickness_m", 0.0))

        # Model panel plane constant (signed distance from origin along n).
        centroid = FreeCAD.Vector(float(plane.centroid[0]), float(plane.centroid[1]), float(plane.centroid[2]))
        d_panel = float(n.dot(centroid))

        shape = getattr(glass_obj, "Shape", None)
        if shape is None or getattr(shape, "isNull", lambda: True)():
            failures.append({"panel": panel.index, "name": glass_name, "reason": "null_shape"})
            continue

        cand = _collect_glass_faces(shape, n)
        if len(cand) < 2:
            failures.append(
                {
                    "panel": panel.index,
                    "name": glass_name,
                    "reason": "missing_planar_faces",
                    "found": len(cand),
                }
            )
            continue

        checked += 1

        face_info = []
        deltas = []
        max_angle = 0.0

        for area, face, fn, dot in cand:
            angle = _deg(math.acos(_clamp(abs(float(dot)), -1.0, 1.0)))
            max_angle = max(max_angle, angle)
            try:
                com = face.CenterOfMass
            except Exception:
                com = getattr(glass_obj, "Placement", None)
                com = getattr(com, "Base", None)
            if com is None:
                failures.append({"panel": panel.index, "name": glass_name, "reason": "no_face_com"})
                break
            d_face = float(n.dot(com))
            delta = d_face - d_panel
            deltas.append(delta)
            face_info.append(
                {
                    "area": float(area),
                    "angle_deg": float(angle),
                    "delta_m": float(delta),
                }
            )

        if len(deltas) != 2:
            continue

        # Identify which face corresponds to the translated base face vs the outer face.
        base_delta = max(deltas)
        outer_delta = min(deltas)

        exp_base = seat
        exp_outer = seat - thickness

        base_err = abs(base_delta - exp_base)
        outer_err = abs(outer_delta - exp_outer)

        ok = (max_angle <= angle_tol_deg) and (base_err <= offset_tol_m) and (outer_err <= offset_tol_m)
        if not ok:
            failures.append(
                {
                    "panel": panel.index,
                    "name": glass_name,
                    "angle_deg": float(max_angle),
                    "base_delta_m": float(base_delta),
                    "outer_delta_m": float(outer_delta),
                    "expected_base_delta_m": float(exp_base),
                    "expected_outer_delta_m": float(exp_outer),
                    "base_err_m": float(base_err),
                    "outer_err_m": float(outer_err),
                    "tol_angle_deg": float(angle_tol_deg),
                    "tol_offset_m": float(offset_tol_m),
                    "faces": face_info,
                }
            )

    report = {
        "config": config,
        "out_dir": out_dir,
        "angle_tol_deg": angle_tol_deg,
        "offset_tol_m": offset_tol_m,
        "checked": checked,
        "failures": failures,
        "failed": len(failures),
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "glass_alignment_check.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"checked={checked} failed={len(failures)} report={report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
