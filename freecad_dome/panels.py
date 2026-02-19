"""Panel generation and validation helpers."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .parameters import DomeParameters
from .tessellation import Panel as PanelDef, TessellatedDome

__all__ = ["PanelInstance", "PanelBuilder"]


@dataclass(slots=True)
class PanelInstance:
    name: str
    area: float
    perimeter: float
    node_indices: tuple[int, ...]
    frame_name: Optional[str] = None


@dataclass(slots=True)
class _PanelPlaneData:
    points3d: List[Tuple[float, float, float]]
    coords2d: List[Tuple[float, float]]
    node_indices: Tuple[int, ...]
    centroid: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    axis_u: Tuple[float, float, float]
    axis_v: Tuple[float, float, float]


@dataclass(slots=True)
class _FrameResult:
    shape: Any
    profile_width_m: float
    profile_height_m: float
    inset_m: float


class PanelBuilder:
    def __init__(self, params: DomeParameters, document: Any | None = None):
        self.params = params
        self._doc = document
        # FreeCAD's internal length unit is millimeters (mm). Tessellation and parameters
        # in this project are stored in meters, so scale geometry sent to FreeCAD.
        self._fc_unit_scale: float = 1000.0

    def _fc_len(self, meters: float) -> float:
        return float(meters) * float(self._fc_unit_scale)

    def _fc_point(self, p: Tuple[float, float, float]) -> Tuple[float, float, float]:
        s = float(self._fc_unit_scale)
        return (float(p[0]) * s, float(p[1]) * s, float(p[2]) * s)

    def ensure_document(self) -> Optional[Any]:
        if self._doc is not None:
            return self._doc
        try:
            import FreeCAD  # type: ignore
        except ImportError:  # pragma: no cover - outside FreeCAD
            return None
        doc = FreeCAD.ActiveDocument
        if doc is None:
            doc = FreeCAD.newDocument("GeodesicDome")
        self._doc = doc
        return doc

    @property
    def document(self) -> Optional[Any]:
        return self._doc

    def create_panels(self, dome: TessellatedDome) -> List[PanelInstance]:
        doc = self.ensure_document()
        instances: List[PanelInstance] = []
        frame_requested = bool(doc) and self.params.generate_panel_frames
        frames_created = 0

        for panel in dome.panels:
            area = self._area(dome, panel)
            perimeter = self._perimeter(dome, panel)
            name = f"Panel_{panel.index:04d}"
            plane_data: _PanelPlaneData | None = None
            frame_name: Optional[str] = None
            create_face = bool(doc) and self.params.generate_panel_faces
            create_frame = frame_requested
            if doc and (create_face or create_frame):
                plane_data = self._panel_plane_data(dome, panel)
            if create_face and plane_data is not None:
                shape = self._make_face(plane_data)
                if shape is None:
                    logging.warning("Unable to create panel geometry for %s", name)
                else:
                    feature = doc.addObject("Part::Feature", name)
                    feature.Shape = shape
                    self._add_to_panels_group(doc, feature)
            # Optional glass solids (independent of face/frame generation).
            if doc and plane_data is not None and self.params.glass_thickness_m > 0:
                seat_offset = self._glass_seat_offset_m(dome, panel, plane_data)
                glass = self._make_glass_solid(plane_data, seat_offset_m=seat_offset)
                if glass is None:
                    logging.warning("Unable to create glass panel for %s", name)
                else:
                    self._create_glass_panel_object(doc, panel, plane_data, glass, seat_offset)
            if create_frame and plane_data is not None:
                frame_result = self._make_panel_frame(panel, plane_data, dome)
                frame_shape = frame_result.shape if frame_result is not None else None

                if frame_shape is None:
                    logging.warning("Unable to create frame geometry for %s", name)
                else:
                    frame_name = f"PanelFrame_{panel.index:04d}"
                    frame = doc.addObject("Part::Feature", frame_name)
                    frame.Shape = frame_shape
                    self._add_to_panels_group(doc, frame)
                    # Store effective parameters for diagnostics.
                    try:
                        if not hasattr(frame, "FrameProfileWidth"):
                            frame.addProperty(
                                "App::PropertyFloat",
                                "FrameProfileWidth",
                                "Dome",
                                "Effective frame profile width (m)",
                            )
                        if not hasattr(frame, "FrameProfileHeight"):
                            frame.addProperty(
                                "App::PropertyFloat",
                                "FrameProfileHeight",
                                "Dome",
                                "Effective frame profile height (m)",
                            )
                        if not hasattr(frame, "FrameInset"):
                            frame.addProperty(
                                "App::PropertyFloat",
                                "FrameInset",
                                "Dome",
                                "Effective frame inset (m)",
                            )
                        if frame_result is not None:
                            frame.FrameProfileWidth = float(frame_result.profile_width_m)
                            frame.FrameProfileHeight = float(frame_result.profile_height_m)
                            frame.FrameInset = float(frame_result.inset_m)
                    except Exception:
                        pass
                    frames_created += 1
            instances.append(
                PanelInstance(
                    name=name,
                    area=area,
                    perimeter=perimeter,
                    node_indices=panel.node_indices,
                    frame_name=frame_name,
                )
            )
        if doc:
            doc.recompute()

        if frame_requested and frames_created != len(dome.panels):
            logging.warning(
                "Panel frames created for %d/%d panels; reduce inset/profile to fit small/acute panels",
                frames_created,
                len(dome.panels),
            )
        return instances

    def create_glass_panels(self, dome: TessellatedDome) -> int:
        """Create only glass panel solids.

        This is useful when running in "struts" mode where base faces/frames are disabled.
        Returns number of created glass solids.
        """
        doc = self.ensure_document()
        if not doc:
            return 0
        if self.params.glass_thickness_m <= 0:
            return 0
        created = 0
        for panel in dome.panels:
            plane_data = self._panel_plane_data(dome, panel)
            if plane_data is None:
                continue
            seat_offset = self._glass_seat_offset_m(dome, panel, plane_data)
            glass = self._make_glass_solid(plane_data, seat_offset_m=seat_offset)
            if glass is None:
                continue
            self._create_glass_panel_object(doc, panel, plane_data, glass, seat_offset)
            created += 1
        doc.recompute()
        return created

    def _create_glass_panel_object(
        self,
        doc: Any,
        panel: PanelDef,
        plane_data: _PanelPlaneData,
        glass_shape: Any,
        seat_offset: float,
    ) -> Any:
        """Add a glass panel FreeCAD object with IFC metadata and styling.

        Centralizes the glass object creation, tagging, and visual setup so that
        both ``create_panels`` and ``create_glass_panels`` share the same code.
        """
        glass_name = f"GlassPanel_{panel.index:04d}"
        glass_obj = doc.addObject("Part::Feature", glass_name)
        glass_obj.Shape = glass_shape
        self._add_to_panels_group(doc, glass_obj)
        try:
            if hasattr(glass_obj, "addProperty") and not hasattr(glass_obj, "IfcType"):
                glass_obj.addProperty(
                    "App::PropertyString", "IfcType", "IFC", "IFC entity type for export",
                )
            if hasattr(glass_obj, "IfcType"):
                glass_obj.IfcType = "IfcPlate"

            if hasattr(glass_obj, "addProperty") and not hasattr(glass_obj, "PanelIndex"):
                glass_obj.addProperty(
                    "App::PropertyInteger", "PanelIndex", "Dome", "Source panel index",
                )
            if hasattr(glass_obj, "PanelIndex"):
                glass_obj.PanelIndex = int(panel.index)

            if hasattr(glass_obj, "addProperty") and not hasattr(glass_obj, "GlassThicknessM"):
                glass_obj.addProperty(
                    "App::PropertyFloat", "GlassThicknessM", "Dome", "Glass thickness (m)",
                )
            if hasattr(glass_obj, "GlassThicknessM"):
                glass_obj.GlassThicknessM = float(self.params.glass_thickness_m)

            if hasattr(glass_obj, "addProperty") and not hasattr(glass_obj, "GlassSeatOffset"):
                glass_obj.addProperty(
                    "App::PropertyFloat", "GlassSeatOffset", "Dome",
                    "Signed offset along panel normal (m) used to seat glass on struts",
                )
            if hasattr(glass_obj, "addProperty") and not hasattr(glass_obj, "GlassNormal"):
                glass_obj.addProperty(
                    "App::PropertyVector", "GlassNormal", "Dome",
                    "Panel plane normal used for glass orientation",
                )
            if hasattr(glass_obj, "GlassSeatOffset"):
                glass_obj.GlassSeatOffset = float(seat_offset)
            if hasattr(glass_obj, "GlassNormal"):
                try:
                    import FreeCAD  # type: ignore

                    nx, ny, nz = plane_data.normal
                    glass_obj.GlassNormal = FreeCAD.Vector(float(nx), float(ny), float(nz))
                except Exception:
                    pass
        except Exception:
            pass
        vo = getattr(glass_obj, "ViewObject", None)
        if vo is not None:
            try:
                vo.Transparency = 80
            except Exception:
                pass
            try:
                vo.ShapeColor = (0.2, 0.6, 1.0)
            except Exception:
                pass
        return glass_obj

    def _panels_group(self, doc: Any) -> Any | None:
        """Return a group for visible panel objects (faces/frames/glass)."""
        try:
            objs = list(getattr(doc, "Objects", []) or [])
        except Exception:
            objs = []
        by_name = {str(getattr(o, "Name", "")): o for o in objs}

        group = by_name.get("Panels")
        if group is None:
            try:
                group = doc.addObject("App::DocumentObjectGroup", "Panels")
                try:
                    group.Label = "Panels"
                except Exception:
                    pass
            except Exception:
                return None
        return group

    def _add_to_panels_group(self, doc: Any, obj: Any) -> None:
        group = self._panels_group(doc)
        if group is None:
            return
        try:
            group.addObject(obj)
        except Exception:
            pass

    def _shapes_collide(self, a, b, volume_tol_m3: float = 1e-9) -> bool:
        try:
            common = a.common(b)
        except Exception:
            return False
        if common is None:
            return False
        if hasattr(common, "isNull") and common.isNull():
            return False
        try:
            vol = float(getattr(common, "Volume", 0.0))
        except Exception:
            return False
        # FreeCAD volume is in (mm^3) when working in mm.
        tol_mm3 = float(volume_tol_m3) * float(self._fc_unit_scale ** 3)
        return vol > tol_mm3

    def _make_face(self, plane: _PanelPlaneData | None):
        if plane is None:
            return None
        try:
            import Part  # type: ignore
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover - headless
            return None

        verts = [Vector(*self._fc_point(point)) for point in plane.points3d]
        if len(verts) < 3:
            return None

        normal = Vector(*plane.normal)
        if normal.Length == 0 and len(verts) >= 3:
            normal = (verts[1] - verts[0]).cross(verts[2] - verts[0])
        if normal.Length == 0:
            return None
        normal.normalize()

        projected = verts + [verts[0]]

        wire = Part.makePolygon(projected)
        try:
            face = Part.Face(wire)
        except Exception as exc:  # pragma: no cover - FreeCAD-specific failure
            logging.debug("Planar face creation failed (%s); trying filled face", exc)
            try:
                face = Part.makeFilledFace([wire])
            except Exception as exc2:  # pragma: no cover - FreeCAD-specific failure
                logging.debug("Filled face creation failed: %s", exc2)
                return None
        return face

    def _make_glass_solid(self, plane: _PanelPlaneData | None, seat_offset_m: float = 0.0):
        if plane is None:
            return None
        thickness = float(self.params.glass_thickness_m)
        if thickness <= 0:
            return None
        gap = float(self.params.glass_gap_m)
        inset = max(0.0, gap * 0.5)

        try:
            import Part  # type: ignore
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover
            return None

        loop2d = plane.coords2d[:]
        if len(loop2d) < 3:
            return None
        if self._polygon_area_2d(loop2d) < 0:
            loop2d.reverse()

        loop = loop2d
        inset_try = inset
        if inset_try > 1e-9:
            for _ in range(12):
                candidate = self._validated_inset(loop2d, inset_try)
                if candidate and len(candidate) >= 3:
                    loop = candidate
                    break
                inset_try *= 0.85

        pts3 = [Vector(*self._fc_point(self._lift_point(plane, pt))) for pt in loop]
        if len(pts3) < 3:
            return None
        try:
            wire = Part.makePolygon(pts3 + [pts3[0]])
            face = Part.Face(wire)
        except Exception:
            try:
                wire = Part.makePolygon(pts3 + [pts3[0]])
                face = Part.makeFilledFace([wire])
            except Exception:
                return None

        n = Vector(*plane.normal)
        if n.Length == 0:
            n = (pts3[1] - pts3[0]).cross(pts3[2] - pts3[0])
        if n.Length == 0:
            return None
        n.normalize()

        # Panels normals are oriented inward; glass should sit on struts and extend outward.
        outward = -n
        try:
            solid = face.extrude(outward * float(self._fc_len(thickness)))
        except Exception:
            return None
        # Translate the base face from the panel plane onto the strut seating plane.
        try:
            if abs(float(seat_offset_m)) > 1e-12:
                solid.translate(n * float(self._fc_len(seat_offset_m)))
        except Exception:
            pass
        return solid if self._is_shape_valid(solid) else None

    def _glass_seat_offset_m(self, dome: TessellatedDome, panel: PanelDef, plane: _PanelPlaneData) -> float:
        """Compute signed offset along panel normal to place glass on strut outer seating planes.

        Returns 0 when strut metadata is unavailable (e.g. headless mode without struts).
        """
        n = self._normalize3(plane.normal)
        if n is None:
            return 0.0
        centroid = plane.centroid
        offsets: List[float] = []
        for strut in dome.struts:
            if panel.index not in strut.panel_indices:
                continue

            # Pick the strut side whose stored panel normal matches this panel.
            match = False
            pn = getattr(strut, "primary_normal", None)
            if pn is not None and self._dot(self._normalize3(pn) or pn, n) > 0.999:
                match = True
            else:
                sn = getattr(strut, "secondary_normal", None)
                if sn is not None and self._dot(self._normalize3(sn) or sn, n) > 0.999:
                    match = True
            if not match:
                continue

            d = self._strut_outer_seat_offset_for_panel(strut, n, centroid)
            if d is not None:
                offsets.append(float(d))

        if not offsets:
            return 0.0
        offsets.sort()
        mid = len(offsets) // 2
        if len(offsets) % 2 == 1:
            return offsets[mid]
        return 0.5 * (offsets[mid - 1] + offsets[mid])

    def _strut_outer_seat_offset_for_panel(
        self,
        strut,
        panel_normal: Tuple[float, float, float],
        panel_point: Tuple[float, float, float],
    ) -> float | None:
        """Signed distance (m) from panel plane to strut outer ridge plane along panel normal."""
        a = (
            strut.end[0] - strut.start[0],
            strut.end[1] - strut.start[1],
            strut.end[2] - strut.start[2],
        )
        a = self._normalize3(a)
        if a is None:
            return None
        m = (
            (strut.start[0] + strut.end[0]) * 0.5,
            (strut.start[1] + strut.end[1]) * 0.5,
            (strut.start[2] + strut.end[2]) * 0.5,
        )

        # w: outward direction across strut width (radial projected perpendicular to axis).
        dot_ma = self._dot(m, a)
        w = (m[0] - a[0] * dot_ma, m[1] - a[1] * dot_ma, m[2] - a[2] * dot_ma)
        w = self._normalize3(w)
        if w is None:
            # Fallback: any perpendicular.
            w = self._normalize3(self._cross(a, (0.0, 0.0, 1.0)))
            if w is None:
                w = self._normalize3(self._cross(a, (0.0, 1.0, 0.0)))
        if w is None:
            return None

        ridge = (
            m[0] + w[0] * (self.params.stock_width_m * 0.5),
            m[1] + w[1] * (self.params.stock_width_m * 0.5),
            m[2] + w[2] * (self.params.stock_width_m * 0.5),
        )
        rel = (
            ridge[0] - panel_point[0],
            ridge[1] - panel_point[1],
            ridge[2] - panel_point[2],
        )
        return float(self._dot(panel_normal, rel))

    def _make_panel_frame(
        self,
        panel: PanelDef,
        plane: _PanelPlaneData | None,
        dome: TessellatedDome,
        requested_width_m: float | None = None,
    ):
        if plane is None:
            return None
        inset = max(0.0, self.params.panel_frame_inset_m)
        requested_width = (
            float(requested_width_m)
            if requested_width_m is not None
            else self.params.panel_frame_profile_width_m
        )
        height = self.params.panel_frame_profile_height_m
        if requested_width <= 0 or height <= 0:
            return None
        try:
            import Part  # type: ignore
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover - headless/dev mode
            return None

        base_loop = plane.coords2d[:]
        outer_loop = base_loop
        outer_inset = inset
        if outer_inset > 1e-9:
            outer_loop = []
            for _ in range(10):
                candidate = self._validated_inset(base_loop, outer_inset)
                if candidate and len(candidate) >= 3:
                    outer_loop = candidate
                    break
                outer_inset *= 0.85
        if not outer_loop or len(outer_loop) < 3:
            logging.warning("Outer frame loop failed for panel %s", panel.index)
            return None
        if self._polygon_area_2d(outer_loop) < 0:
            outer_loop.reverse()

        # Build the inner loop. If the requested width cannot fit (acute/small panels),
        # automatically narrow the profile width per-panel until a valid solid exists.
        # Allow per-panel narrowing down to ~5% of requested width (but not below 1mm).
        # Some panels can be very acute and won't accommodate a wide frame.
        min_width = max(0.001, float(requested_width) * 0.05)

        outer_points = [Vector(*self._fc_point(self._lift_point(plane, pt))) for pt in outer_loop]
        if len(outer_points) < 3:
            return None
        normal_vec = Vector(*plane.normal)
        if normal_vec.Length == 0:
            normal_vec = (outer_points[1] - outer_points[0]).cross(outer_points[2] - outer_points[0])
        if normal_vec.Length == 0:
            return None
        normal_vec.normalize()

        width_try = float(requested_width)
        for attempt in range(30):
            if width_try < min_width:
                break
            inner_inset = outer_inset + width_try
            inner_loop = self._validated_inset(base_loop, inner_inset)
            if not inner_loop or len(inner_loop) < 3:
                width_try *= 0.85
                continue
            if self._polygon_area_2d(inner_loop) > 0:
                inner_loop.reverse()

            if any(not self._point_in_polygon(pt, outer_loop) for pt in inner_loop):
                width_try *= 0.85
                continue

            inner_points = [Vector(*self._fc_point(self._lift_point(plane, pt))) for pt in inner_loop]
            if len(inner_points) < 3:
                width_try *= 0.85
                continue

            try:
                outer_wire = Part.makePolygon(outer_points + [outer_points[0]])
                inner_wire = Part.makePolygon(inner_points + [inner_points[0]])
            except Exception:
                width_try *= 0.85
                continue

            face = None
            try:
                face = Part.Face([outer_wire, inner_wire])
            except Exception:
                try:
                    face = Part.makeFilledFace([outer_wire, inner_wire])
                except Exception:
                    face = None

            if face is None:
                width_try *= 0.85
                continue

            try:
                solid = face.extrude(normal_vec * float(self._fc_len(height)))
            except Exception:
                solid = None
            if solid is None:
                width_try *= 0.85
                continue

            if hasattr(solid, "isValid") and not solid.isValid():
                width_try *= 0.85
                continue

            if attempt > 0 and width_try < requested_width * 0.999:
                logging.info(
                    "Panel %s: auto-narrowed frame width from %.3fmm to %.3fmm",
                    panel.index,
                    requested_width * 1000.0,
                    width_try * 1000.0,
                )
            return _FrameResult(shape=solid, profile_width_m=width_try, profile_height_m=height, inset_m=outer_inset)

        # Fallback: build a simple per-edge strip frame (no global inner loop).
        # This is less clean at corners but is robust for very skinny/degenerate panels.
        strip_width = max(0.0005, min_width)
        strip_shapes = []
        for idx in range(len(outer_loop)):
            a2 = outer_loop[idx]
            b2 = outer_loop[(idx + 1) % len(outer_loop)]
            dx = b2[0] - a2[0]
            dy = b2[1] - a2[1]
            ln = math.hypot(dx, dy)
            if ln < 1e-12:
                continue
            ux = dx / ln
            uy = dy / ln
            # For CCW outer loop, inward is left of the edge.
            nx = -uy
            ny = ux
            a_in = (a2[0] + nx * strip_width, a2[1] + ny * strip_width)
            b_in = (b2[0] + nx * strip_width, b2[1] + ny * strip_width)
            quad2 = [a2, b2, b_in, a_in]

            try:
                quad3 = [Vector(*self._fc_point(self._lift_point(plane, pt))) for pt in quad2]
                wire = Part.makePolygon(quad3 + [quad3[0]])
                face = Part.Face(wire)
                seg = face.extrude(normal_vec * float(self._fc_len(height)))
                if hasattr(seg, "isValid") and not seg.isValid():
                    continue
                strip_shapes.append(seg)
            except Exception:
                continue

        if strip_shapes:
            try:
                fallback = Part.makeCompound(strip_shapes) if len(strip_shapes) > 1 else strip_shapes[0]
                logging.info(
                    "Panel %s: used strip-frame fallback at %.3fmm width",
                    panel.index,
                    strip_width * 1000.0,
                )
                return _FrameResult(shape=fallback, profile_width_m=strip_width, profile_height_m=height, inset_m=outer_inset)
            except Exception:
                pass

        logging.warning("Inner frame loop failed for panel %s", panel.index)
        return None


    def _validated_inset(self, base_loop: List[Tuple[float, float]], distance: float) -> List[Tuple[float, float]]:
        loop = self._inset_polygon(base_loop, distance)
        if len(loop) < 3:
            return []
        area = abs(self._polygon_area_2d(loop))
        if area < 1e-12:
            return []
        return loop

    def _build_frame_segment(
        self,
        panel: PanelDef,
        plane: _PanelPlaneData,
        outer_points: List[Tuple[float, float, float]],
        inner_points: List[Tuple[float, float, float]],
        index: int,
        dome: TessellatedDome,
        height: float,
    ):
        try:
            import Part  # type: ignore
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover - headless/dev mode
            return None

        count = len(outer_points)
        next_idx = (index + 1) % count
        quad = [
            outer_points[index],
            outer_points[next_idx],
            inner_points[next_idx],
            inner_points[index],
        ]
        if len({self._quantize_point(pt) for pt in quad}) < 3:
            return None

        try:
            wire = Part.makePolygon([Vector(*pt) for pt in quad + [quad[0]]])
            face = Part.Face(wire)
        except Exception as exc:
            logging.debug("Frame segment face failed for panel %s: %s", panel.index, exc)
            return None

        normal_vec = Vector(*plane.normal)
        if normal_vec.Length == 0:
            return None
        normal_vec.normalize()
        try:
            solid = face.extrude(normal_vec * height)
        except Exception as exc:
            logging.debug("Frame segment extrusion failed for panel %s: %s", panel.index, exc)
            return None

        node_indices = plane.node_indices
        a_idx = node_indices[index]
        b_idx = node_indices[(index + 1) % len(node_indices)]
        vertex_a = dome.nodes[a_idx]
        vertex_b = dome.nodes[b_idx]
        ref_normal = self._edge_plane_normal(vertex_a, vertex_b)
        if ref_normal is not None:
            point_on_plane = (0.0, 0.0, 0.0)
            to_centroid = (
                plane.centroid[0] - point_on_plane[0],
                plane.centroid[1] - point_on_plane[1],
                plane.centroid[2] - point_on_plane[2],
            )
            if self._dot(ref_normal, to_centroid) < 0:
                ref_normal = (-ref_normal[0], -ref_normal[1], -ref_normal[2])
            edge_length = self._distance(vertex_a, vertex_b)
            span_hint = max(edge_length * 1.2, height * 6.0)
            trimmed = self._cut_with_plane(
                solid,
                point_on_plane,
                ref_normal,
                plane.centroid,
                span_hint,
            )
            if trimmed is None or not self._is_shape_valid(trimmed):
                logging.debug(
                    "Frame segment cut failed for panel %s edge (%d,%d)", panel.index, a_idx, b_idx
                )
            else:
                solid = trimmed

        # Miter both ends at the inset vertices so adjacent segments meet cleanly.
        vertex_trim_points = [outer_points[index], outer_points[next_idx]]
        for trim_point in vertex_trim_points:
            trim_normal = self._vertex_trim_normal(trim_point, plane)
            if trim_normal is None:
                continue
            trimmed = self._cut_with_plane(
                solid,
                trim_point,
                trim_normal,
                plane.centroid,
                span_hint=height * 4.0,
            )
            if trimmed is None or not self._is_shape_valid(trimmed):
                continue
            solid = trimmed

        return solid

    def _panel_plane_data(self, dome: TessellatedDome, panel: PanelDef) -> _PanelPlaneData | None:
        indices = panel.node_indices
        if len(indices) < 3:
            return None
        ordered_indices = list(indices)
        points = [dome.nodes[idx] for idx in ordered_indices]
        centroid = self._average_point(points)
        # Use tessellation-provided normal for consistent inward orientation when available.
        normal = self._normalize3(panel.normal)
        if normal is None:
            normal = self._normalize3(self._polygon_normal(points))
        if normal is None:
            return None
        axis_u = self._build_axis(normal)
        if axis_u is None:
            return None
        axis_v = self._cross(normal, axis_u)
        axis_v = self._normalize3(axis_v)
        if axis_v is None:
            return None

        coords2d: List[Tuple[float, float]] = []
        for point in points:
            rel = (
                point[0] - centroid[0],
                point[1] - centroid[1],
                point[2] - centroid[2],
            )
            coords2d.append((self._dot(rel, axis_u), self._dot(rel, axis_v)))

        if self._polygon_area_2d(coords2d) < 0:
            coords2d.reverse()
            ordered_indices.reverse()
            points.reverse()
        return _PanelPlaneData(
            # Keep the shared 3D vertices. When the dome nodes are planarized, these points lie
            # in the panel plane and _lift_point becomes an identity transform.
            points3d=points,
            coords2d=coords2d,
            node_indices=tuple(ordered_indices),
            centroid=centroid,
            normal=normal,
            axis_u=axis_u,
            axis_v=axis_v,
        )

    def _lift_point(self, plane: _PanelPlaneData, coord: Tuple[float, float]) -> Tuple[float, float, float]:
        return self._lift_point_tuple(plane.centroid, plane.axis_u, plane.axis_v, coord)

    @staticmethod
    def _lift_point_tuple(
        centroid: Tuple[float, float, float],
        axis_u: Tuple[float, float, float],
        axis_v: Tuple[float, float, float],
        coord: Tuple[float, float],
    ) -> Tuple[float, float, float]:
        x, y = coord
        return (
            centroid[0] + axis_u[0] * x + axis_v[0] * y,
            centroid[1] + axis_u[1] * x + axis_v[1] * y,
            centroid[2] + axis_u[2] * x + axis_v[2] * y,
        )

    def _inset_polygon(
        self, coords: List[Tuple[float, float]], distance: float
    ) -> List[Tuple[float, float]]:
        if distance <= 1e-9:
            return coords[:]
        if len(coords) < 3:
            return []
        inset_poly: List[Tuple[float, float]] = []
        # Prevent extremely long miters at acute angles from creating wildly
        # distorted/self-intersecting inset polygons.
        # Keep exactly one vertex per corner (no bevel duplicates) so the
        # resulting inner loop has proper "sharp" corners.
        miter_limit = 4.0
        for idx in range(len(coords)):
            prev = coords[(idx - 1) % len(coords)]
            curr = coords[idx]
            nxt = coords[(idx + 1) % len(coords)]
            d1 = self._normalize2((curr[0] - prev[0], curr[1] - prev[1]))
            d2 = self._normalize2((nxt[0] - curr[0], nxt[1] - curr[1]))
            if d1 is None or d2 is None:
                return []
            n1 = (-d1[1], d1[0])
            n2 = (-d2[1], d2[0])
            p1 = (curr[0] + n1[0] * distance, curr[1] + n1[1] * distance)
            p2 = (curr[0] + n2[0] * distance, curr[1] + n2[1] * distance)
            intersection = self._intersect_lines(p1, d1, p2, d2)
            if intersection is None:
                # Adjacent edges are nearly parallel (very obtuse/flat corner).
                # Use a stable single-point fallback (midpoint of the two offsets).
                inset_poly.append(((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5))
                continue

            miter_len = math.hypot(intersection[0] - curr[0], intersection[1] - curr[1])
            if miter_len > (miter_limit * distance):
                # Clamp the miter point back toward the corner.
                dx = intersection[0] - curr[0]
                dy = intersection[1] - curr[1]
                ln = math.hypot(dx, dy)
                if ln <= 1e-18:
                    inset_poly.append(intersection)
                else:
                    scale = (miter_limit * distance) / ln
                    inset_poly.append((curr[0] + dx * scale, curr[1] + dy * scale))
            else:
                inset_poly.append(intersection)
        if self._polygon_area_2d(inset_poly) < 0:
            inset_poly.reverse()
        return inset_poly

    def _align_loop(
        self,
        reference: List[Tuple[float, float]],
        loop: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if not reference or len(reference) != len(loop):
            return loop
        best_offset = 0
        best_score = float("inf")
        count = len(loop)
        for offset in range(count):
            score = 0.0
            for idx, ref in enumerate(reference):
                other = loop[(idx + offset) % count]
                dx = ref[0] - other[0]
                dy = ref[1] - other[1]
                score += dx * dx + dy * dy
                if score >= best_score:
                    break
            if score < best_score:
                best_score = score
                best_offset = offset
        if best_offset == 0:
            return loop
        return [loop[(idx + best_offset) % count] for idx in range(count)]

    @staticmethod
    def _normalize2(vec: Tuple[float, float]) -> Tuple[float, float] | None:
        length = math.hypot(vec[0], vec[1])
        if length == 0:
            return None
        return (vec[0] / length, vec[1] / length)

    @staticmethod
    def _intersect_lines(
        p1: Tuple[float, float],
        d1: Tuple[float, float],
        p2: Tuple[float, float],
        d2: Tuple[float, float],
    ) -> Tuple[float, float] | None:
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:
            return None
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / denom
        return (p1[0] + d1[0] * t, p1[1] + d1[1] * t)

    @staticmethod
    def _polygon_area_2d(points: List[Tuple[float, float]]) -> float:
        if not points:
            return 0.0
        area = 0.0
        for idx, point in enumerate(points):
            nxt = points[(idx + 1) % len(points)]
            area += point[0] * nxt[1] - nxt[0] * point[1]
        return area * 0.5

    @staticmethod
    def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        # Ray casting algorithm; considers boundary as inside.
        x, y = point
        inside = False
        n = len(polygon)
        if n < 3:
            return False
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            # Check if point is on the segment.
            dx = x2 - x1
            dy = y2 - y1
            px = x - x1
            py = y - y1
            cross = dx * py - dy * px
            if abs(cross) < 1e-12:
                dot = px * dx + py * dy
                if dot >= -1e-12 and dot <= (dx * dx + dy * dy) + 1e-12:
                    return True

            intersects = ((y1 > y) != (y2 > y)) and (
                x < (dx * (y - y1) / (y2 - y1 + 1e-30) + x1)
            )
            if intersects:
                inside = not inside
        return inside

    @staticmethod
    def _average_point(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        if not points:
            return (0.0, 0.0, 0.0)
        count = len(points)
        sx = sum(p[0] for p in points) / count
        sy = sum(p[1] for p in points) / count
        sz = sum(p[2] for p in points) / count
        return (sx, sy, sz)

    @staticmethod
    def _polygon_normal(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        nx = ny = nz = 0.0
        for idx, point in enumerate(points):
            nxt = points[(idx + 1) % len(points)]
            nx += (point[1] - nxt[1]) * (point[2] + nxt[2])
            ny += (point[2] - nxt[2]) * (point[0] + nxt[0])
            nz += (point[0] - nxt[0]) * (point[1] + nxt[1])
        return (nx, ny, nz)

    @staticmethod
    def _normalize3(vec: Tuple[float, float, float]) -> Tuple[float, float, float] | None:
        length = math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        if length == 0:
            return None
        return (vec[0] / length, vec[1] / length, vec[2] / length)

    @staticmethod
    def _cross(
        a: Tuple[float, float, float], b: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _build_axis(self, normal: Tuple[float, float, float]) -> Tuple[float, float, float] | None:
        reference = (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0)
        axis = self._cross(normal, reference)
        return self._normalize3(axis)

    @staticmethod
    def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _edge_plane_normal(
        self, vertex_a: Tuple[float, float, float], vertex_b: Tuple[float, float, float]
    ) -> Tuple[float, float, float] | None:
        normal = self._cross(vertex_a, vertex_b)
        return self._normalize3(normal)

    def _vertex_trim_normal(
        self,
        point: Tuple[float, float, float],
        plane: _PanelPlaneData,
    ) -> Tuple[float, float, float] | None:
        direction = (
            point[0] - plane.centroid[0],
            point[1] - plane.centroid[1],
            point[2] - plane.centroid[2],
        )
        direction = self._normalize3(direction)
        if direction is None:
            return None
        normal = self._cross(direction, plane.normal)
        return self._normalize3(normal)

    @staticmethod
    def _quantize_point(point: Tuple[float, float, float], digits: int = 9) -> Tuple[float, float, float]:
        return (round(point[0], digits), round(point[1], digits), round(point[2], digits))

    def _cut_with_plane(
        self,
        shape,
        point: Tuple[float, float, float],
        normal: Tuple[float, float, float],
        keep_point: Tuple[float, float, float],
        span_hint: float | None = None,
    ):
        try:
            import Part  # type: ignore
            import FreeCAD  # type: ignore
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover - headless/dev mode
            return shape

        normal_vec = Vector(*normal)
        if normal_vec.Length == 0:
            return None
        normal_vec.normalize()

        span_m = float(span_hint or 0.0)
        span_m = max(
            span_m,
            float(self.params.panel_frame_profile_width_m) * 6.0,
            float(self.params.panel_frame_profile_height_m) * 6.0,
            float(self.params.panel_frame_inset_m) * 4.0,
            0.5,
        )
        span = float(self._fc_len(span_m))
        plane = Part.makePlane(span, span)
        placement = FreeCAD.Placement(Vector(*self._fc_point(point)), FreeCAD.Rotation(Vector(0, 0, 1), normal_vec))
        plane.Placement = placement

        keep_vec = Vector(*self._fc_point(keep_point))
        point_vec = Vector(*self._fc_point(point))
        direction = normal_vec
        if (keep_vec - point_vec).dot(normal_vec) < 0:
            direction = -normal_vec
        prism = plane.extrude(direction * span)
        try:
            return shape.common(prism)
        except Exception:
            return None

    @staticmethod
    def _is_shape_valid(shape) -> bool:
        if shape is None:
            return False
        if hasattr(shape, "isNull") and shape.isNull():
            return False
        if hasattr(shape, "isValid") and not shape.isValid():
            return False
        bbox = getattr(shape, "BoundBox", None)
        if bbox is not None:
            try:
                if hasattr(bbox, "isValid") and not bbox.isValid():
                    return False
            except Exception:
                pass
        shape_type = getattr(shape, "ShapeType", "").lower()
        if shape_type == "solid":
            return True
        if shape_type == "compound":
            solids = getattr(shape, "Solids", [])
            return any(getattr(solid, "ShapeType", "").lower() == "solid" for solid in solids)
        return False
        return True

    def _area(self, dome: TessellatedDome, panel: PanelDef) -> float:
        points = [dome.nodes[idx] for idx in panel.node_indices]
        return self._polygon_area(points)

    def _perimeter(self, dome: TessellatedDome, panel: PanelDef) -> float:
        points = [dome.nodes[idx] for idx in panel.node_indices]
        return self._polygon_perimeter(points)

    @staticmethod
    def _distance(p, q) -> float:
        return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2) ** 0.5

    @staticmethod
    def _triangle_area(a, b, c) -> float:
        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        return 0.5 * (cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) ** 0.5

    @staticmethod
    def _polygon_area(points) -> float:
        if len(points) < 3:
            return 0.0
        origin = points[0]
        area = 0.0
        for idx in range(1, len(points) - 1):
            area += PanelBuilder._triangle_area(origin, points[idx], points[idx + 1])
        return area

    @staticmethod
    def _polygon_perimeter(points) -> float:
        if len(points) < 2:
            return 0.0
        total = 0.0
        for idx, point in enumerate(points):
            nxt = points[(idx + 1) % len(points)]
            total += PanelBuilder._distance(point, nxt)
        return total
