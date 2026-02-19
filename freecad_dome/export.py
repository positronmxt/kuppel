"""Export utilities for the geodesic dome generator.

Handles IFC, STL, DXF, manifest, and panel accuracy report exports.
All FreeCAD-dependent imports are lazy so the module can be imported in
headless/test environments.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

__all__ = [
    "export_manifest",
    "export_ifc",
    "export_stl",
    "export_dxf",
    "collect_structural_objects",
    "write_panel_accuracy_report",
]


# ---------------------------------------------------------------------------
# Manifest export
# ---------------------------------------------------------------------------

def export_manifest(struts: Sequence[Any], destination: Path) -> None:
    """Write per-strut metadata as a JSON manifest."""
    manifest = [
        {
            "name": s.name,
            "length": s.length,
            "material": s.material,
            "group": s.group,
            "sequence": s.sequence,
            "ifc_guid": s.ifc_guid,
        }
        for s in struts
    ]
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Wrote manifest %s", destination)


# ---------------------------------------------------------------------------
# Structural object collection
# ---------------------------------------------------------------------------

def collect_structural_objects(doc: Any) -> List[Any]:
    """Return all Arch Structure / IfcMember / IfcPlate objects from a document."""
    if doc is None:
        return []
    objs = []
    for obj in getattr(doc, "Objects", []):
        if getattr(obj, "IfcType", "") in {"IfcMember", "Beam", "IfcPlate"}:
            objs.append(obj)
    return objs


# ---------------------------------------------------------------------------
# IFC export + fixers
# ---------------------------------------------------------------------------

def export_ifc(objects: Sequence[Any], destination: Path) -> None:
    """Export structural objects to IFC format."""
    if not objects:
        logging.warning("No structural objects to export to IFC")
        return
    try:
        import importIFC  # type: ignore

        importIFC.export(list(objects), str(destination))
    except ImportError:
        try:
            from importers import exportIFC  # type: ignore

            exportIFC.export(list(objects), str(destination))
        except Exception as exc:
            msg = str(exc)
            if "ifcopenshell" in msg.lower():
                logging.warning(
                    "IFC export failed because IfcOpenShell is missing in this FreeCAD environment. "
                    "On the FreeCAD snap you can usually install it with: snap run freecad.pip install ifcopenshell"
                )
            else:
                logging.warning("IFC export not available; skipping (%s)", exc)
            return
    _fix_ifc_length_units_to_meters(destination)
    _fix_ifc_geometry_scale_if_needed(destination)
    logging.info("Wrote IFC %s", destination)


def _fix_ifc_length_units_to_meters(path: Path) -> None:
    """Rewrite IFC header unit assignment to metres.

    FreeCAD's IFC exporter often writes LENGTHUNIT as millimetres, while this
    project authors geometry using metre-valued coordinates.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return

    pattern = re.compile(
        r"(IFCSIUNIT\(\s*(?:\*|\$)\s*,\s*\.LENGTHUNIT\.\s*,\s*)\.[A-Z]+\.\s*,\s*(\.METRE\.\s*\))"
    )
    fixed, n = pattern.subn(r"\1$,\2", text)
    if n <= 0:
        return

    try:
        path.write_text(fixed, encoding="utf-8")
        logging.info("Adjusted IFC LENGTHUNIT to metres (was prefixed)")
    except Exception:
        return


def _fix_ifc_geometry_scale_if_needed(path: Path) -> None:
    """Fix a common 1000x scale mismatch in IFC exports."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return

    pt_pat = re.compile(r"IFCCARTESIANPOINT\(\(([^)]*)\)\)")

    pt_max = 0.0
    pt_count = 0
    for m in pt_pat.finditer(text):
        coords = m.group(1)
        for tok in coords.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = abs(float(tok))
            except Exception:
                continue
            if v > pt_max:
                pt_max = v
        pt_count += 1
        if pt_count >= 800:
            break

    if pt_max <= 0:
        return

    scale = None
    if pt_max < 0.05:
        scale = 1000.0

    if scale is None:
        return

    def _fmt(x: float) -> str:
        return format(x, ".15g")

    def repl(m: re.Match) -> str:
        coords = m.group(1)
        parts = []
        for tok in coords.split(","):
            t = tok.strip()
            if not t:
                continue
            try:
                v = float(t)
                parts.append(_fmt(v * scale))
            except Exception:
                parts.append(t)
        return f"IFCCARTESIANPOINT(({','.join(parts)}))"

    fixed, n = pt_pat.subn(repl, text)
    if n <= 0:
        return
    try:
        path.write_text(fixed, encoding="utf-8")
        logging.info("Scaled IFC geometry points by %.0f (detected small export)", scale)
    except Exception:
        return


# ---------------------------------------------------------------------------
# STL / DXF exports
# ---------------------------------------------------------------------------

def export_stl(objects: Sequence[Any], destination: Path) -> None:
    """Export structural objects to STL format."""
    if not objects:
        logging.warning("No structural objects to export to STL")
        return
    try:
        import Mesh  # type: ignore
    except ImportError:
        logging.warning("Mesh not available; skipping STL export")
        return
    try:
        if hasattr(Mesh, "export"):
            Mesh.export(list(objects), str(destination))
            logging.info("Wrote STL %s", destination)
            return
    except Exception as exc:
        logging.warning("STL export failed (%s); skipping", exc)
        return

    logging.warning("Mesh.export not available; skipping STL export")


def export_dxf(objects: Sequence[Any], destination: Path) -> None:
    """Export structural objects to DXF format."""
    if not objects:
        logging.warning("No structural objects to export to DXF")
        return
    try:
        import importDXF  # type: ignore
    except ImportError:
        logging.warning("importDXF not available; skipping DXF export")
        return
    try:
        importDXF.export(list(objects), str(destination))
        logging.info("Wrote DXF %s", destination)
    except Exception as exc:
        logging.warning("DXF export failed (%s); skipping", exc)


# ---------------------------------------------------------------------------
# Panel accuracy report helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _vec_len(v: Any) -> float:
    return float((v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5)


def _vec_unit(v: Any) -> Any:
    ln = _vec_len(v)
    if ln <= 1e-18:
        return None
    return v.__class__(v.x / ln, v.y / ln, v.z / ln)


def _deg(rad: float) -> float:
    return rad * (180.0 / math.pi)


def _face_normal(face: Any) -> Any:
    try:
        u0, u1, v0, v1 = face.ParameterRange
        u = 0.5 * (u0 + u1)
        v = 0.5 * (v0 + v1)
        return face.normalAt(u, v)
    except Exception:
        try:
            return face.normalAt(0, 0)
        except Exception:
            return None


def _collect_planar_faces_parallel_to(shape: Any, expected_n: Any) -> List[Any]:
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
        dot = float(un.dot(expected_n))
        if abs(dot) < 0.98:
            continue
        try:
            area = float(getattr(f, "Area", 0.0))
        except Exception:
            area = 0.0
        cand.append((area, f, un, dot))

    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[:2]


def _max_vertex_plane_deviation(shape: Any, n: Any, d_plane: float) -> float | None:
    verts = list(getattr(shape, "Vertexes", []) or [])
    if not verts:
        return None
    dev = 0.0
    for v in verts:
        try:
            p = v.Point
        except Exception:
            continue
        dev = max(dev, abs(float(n.dot(p)) - d_plane))
    return float(dev)


def _ensure_reference_panel_faces(doc: Any, dome: Any, panel_builder: Any) -> int:
    """Create hidden reference panel faces ``RefPanel_####`` for all panels.

    These are purely for inspection/diagnostics and do not affect exports.
    Returns the number of newly created reference objects.
    """
    try:
        import FreeCAD  # type: ignore
    except Exception:
        return 0

    objs = list(getattr(doc, "Objects", []) or [])
    by_name = {str(getattr(o, "Name", "")): o for o in objs}

    group = by_name.get("ReferenceGeometry")
    if group is None:
        try:
            group = doc.addObject("App::DocumentObjectGroup", "ReferenceGeometry")
            try:
                group.Label = "ReferenceGeometry"
            except Exception:
                pass
        except Exception:
            group = None

    created = 0
    for panel in getattr(dome, "panels", []) or []:
        name = f"RefPanel_{int(panel.index):04d}"
        if name in by_name:
            continue

        plane = panel_builder._panel_plane_data(dome, panel)
        if plane is None:
            continue

        shape = panel_builder._make_face(plane)
        if shape is None:
            continue

        try:
            obj = doc.addObject("Part::Feature", name)
            obj.Label = name
            obj.Shape = shape
            try:
                if hasattr(obj, "addProperty") and not hasattr(obj, "PanelIndex"):
                    obj.addProperty("App::PropertyInteger", "PanelIndex", "Reference", "Panel index")
                if hasattr(obj, "addProperty") and not hasattr(obj, "PanelNormal"):
                    obj.addProperty("App::PropertyVector", "PanelNormal", "Reference", "Panel normal")
                if hasattr(obj, "addProperty") and not hasattr(obj, "PlaneD"):
                    obj.addProperty(
                        "App::PropertyFloat",
                        "PlaneD",
                        "Reference",
                        "Plane constant d where plane is n·p = d (meters)",
                    )
                obj.PanelIndex = int(panel.index)
                n = plane.normal
                obj.PanelNormal = FreeCAD.Vector(float(n[0]), float(n[1]), float(n[2]))
                c = plane.centroid
                nn = obj.PanelNormal
                ln = (nn.x * nn.x + nn.y * nn.y + nn.z * nn.z) ** 0.5
                if ln > 1e-18:
                    nn = FreeCAD.Vector(nn.x / ln, nn.y / ln, nn.z / ln)
                obj.PlaneD = float(nn.dot(FreeCAD.Vector(float(c[0]), float(c[1]), float(c[2]))))
            except Exception:
                pass

            vo = getattr(obj, "ViewObject", None)
            if vo is not None:
                try:
                    vo.Visibility = False
                except Exception:
                    pass
            if group is not None:
                try:
                    group.addObject(obj)
                except Exception:
                    pass
            created += 1
        except Exception:
            continue

    try:
        doc.recompute()
    except Exception:
        pass
    return created


def write_panel_accuracy_report(doc: Any, dome: Any, params: Any, report_path: Path) -> None:
    """Compare generated panel/glass geometry to the mathematical panel planes."""
    import FreeCAD  # type: ignore
    import Part  # type: ignore

    from .panels import PanelBuilder

    angle_tol_deg = float(os.environ.get("GLASS_ANGLE_TOL_DEG", "0.2"))
    offset_tol_m = float(os.environ.get("GLASS_OFFSET_TOL_M", "0.001"))
    plane_tol_m = float(os.environ.get("PANEL_PLANE_TOL_M", "0.0005"))

    panel_builder = PanelBuilder(params, document=None)

    ref_created = _ensure_reference_panel_faces(doc, dome, panel_builder)

    objs = list(getattr(doc, "Objects", []) or [])
    by_name = {str(getattr(o, "Name", "")): o for o in objs}

    total_panels = len(getattr(dome, "panels", []) or [])
    expected_glass = total_panels if float(getattr(params, "glass_thickness_m", 0.0)) > 0 else 0
    expected_panel_faces = total_panels

    found_glass = 0
    found_panel_faces = 0

    checked_panels = 0
    checked_glass = 0
    failures: list[dict[str, object]] = []

    max_panel_dev_m: float = 0.0
    max_glass_angle_deg: float = 0.0
    max_glass_base_err_m: float = 0.0
    max_glass_outer_err_m: float = 0.0

    missing_glass_panels: list[int] = []
    missing_panel_face_panels: list[int] = []

    thickness = float(getattr(params, "glass_thickness_m", 0.0))

    for panel in dome.panels:
        plane = panel_builder._panel_plane_data(dome, panel)
        if plane is None:
            continue

        n = FreeCAD.Vector(float(plane.normal[0]), float(plane.normal[1]), float(plane.normal[2]))
        n = _vec_unit(n)
        if n is None:
            continue

        centroid = FreeCAD.Vector(float(plane.centroid[0]), float(plane.centroid[1]), float(plane.centroid[2]))
        d_panel = float(n.dot(centroid))

        panel_name = f"Panel_{panel.index:04d}"
        panel_obj = by_name.get(panel_name)
        if panel_obj is not None:
            found_panel_faces += 1
            shape = getattr(panel_obj, "Shape", None)
            if shape is None or getattr(shape, "isNull", lambda: True)():
                failures.append({"panel": panel.index, "name": panel_name, "reason": "panel_null_shape"})
            else:
                dev = _max_vertex_plane_deviation(shape, n, d_panel)
                if dev is None:
                    failures.append({"panel": panel.index, "name": panel_name, "reason": "panel_no_vertices"})
                else:
                    checked_panels += 1
                    max_panel_dev_m = max(max_panel_dev_m, float(dev))
                    if dev > plane_tol_m:
                        failures.append(
                            {
                                "panel": panel.index,
                                "name": panel_name,
                                "reason": "panel_not_on_plane",
                                "max_dev_m": float(dev),
                                "tol_m": float(plane_tol_m),
                            }
                        )
        else:
            missing_panel_face_panels.append(int(panel.index))

        glass_name = f"GlassPanel_{panel.index:04d}"
        glass_obj = by_name.get(glass_name)
        if glass_obj is None:
            if expected_glass:
                missing_glass_panels.append(int(panel.index))
            continue

        found_glass += 1

        shape = getattr(glass_obj, "Shape", None)
        if shape is None or getattr(shape, "isNull", lambda: True)():
            failures.append({"panel": panel.index, "name": glass_name, "reason": "glass_null_shape"})
            continue

        cand = _collect_planar_faces_parallel_to(shape, n)
        if len(cand) < 2:
            failures.append({"panel": panel.index, "name": glass_name, "reason": "glass_missing_planar_faces"})
            continue

        seat = float(panel_builder._glass_seat_offset_m(dome, panel, plane))

        deltas: list[float] = []
        max_angle = 0.0
        faces_info: list[dict[str, float]] = []
        for area, face, fn, dot in cand:
            angle = _deg(math.acos(_clamp(abs(float(dot)), -1.0, 1.0)))
            max_angle = max(max_angle, angle)
            com = getattr(face, "CenterOfMass", None)
            if com is None:
                failures.append({"panel": panel.index, "name": glass_name, "reason": "glass_no_face_com"})
                deltas = []
                break
            delta = float(n.dot(com)) - d_panel
            deltas.append(delta)
            faces_info.append({"area": float(area), "angle_deg": float(angle), "delta_m": float(delta)})

        if len(deltas) != 2:
            continue

        checked_glass += 1
        base_delta = max(deltas)
        outer_delta = min(deltas)

        exp_base = seat
        exp_outer = seat - thickness
        base_err = abs(base_delta - exp_base)
        outer_err = abs(outer_delta - exp_outer)

        max_glass_angle_deg = max(max_glass_angle_deg, float(max_angle))
        max_glass_base_err_m = max(max_glass_base_err_m, float(base_err))
        max_glass_outer_err_m = max(max_glass_outer_err_m, float(outer_err))

        if (max_angle > angle_tol_deg) or (base_err > offset_tol_m) or (outer_err > offset_tol_m):
            failures.append(
                {
                    "panel": panel.index,
                    "name": glass_name,
                    "reason": "glass_misaligned",
                    "angle_deg": float(max_angle),
                    "base_delta_m": float(base_delta),
                    "outer_delta_m": float(outer_delta),
                    "expected_base_delta_m": float(exp_base),
                    "expected_outer_delta_m": float(exp_outer),
                    "base_err_m": float(base_err),
                    "outer_err_m": float(outer_err),
                    "tol_angle_deg": float(angle_tol_deg),
                    "tol_offset_m": float(offset_tol_m),
                    "faces": faces_info,
                }
            )

    report: Dict[str, Any] = {
        "total_panels": total_panels,
        "expected": {"glass": expected_glass, "panel_faces": expected_panel_faces},
        "found": {"glass": found_glass, "panel_faces": found_panel_faces},
        "reference": {"created": int(ref_created), "name_prefix": "RefPanel_"},
        "missing": {
            "glass_panels": missing_glass_panels[:50],
            "glass_missing_count": len(missing_glass_panels),
            "panel_face_panels": missing_panel_face_panels[:50],
            "panel_face_missing_count": len(missing_panel_face_panels),
        },
        "checked_panels": checked_panels,
        "checked_glass": checked_glass,
        "max": {
            "panel_plane_dev_m": float(max_panel_dev_m),
            "glass_angle_deg": float(max_glass_angle_deg),
            "glass_base_err_m": float(max_glass_base_err_m),
            "glass_outer_err_m": float(max_glass_outer_err_m),
        },
        "tol": {"glass_angle_deg": angle_tol_deg, "glass_offset_m": offset_tol_m, "panel_plane_m": plane_tol_m},
        "failed": len(failures),
        "failures": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
