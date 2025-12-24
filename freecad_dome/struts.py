"""Strut builders that integrate tessellation data with FreeCAD."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from uuid import uuid5, NAMESPACE_URL

from .parameters import DomeParameters
from .tessellation import Strut, TessellatedDome

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import Arch  # type: ignore
    import FreeCAD  # type: ignore
    from FreeCAD import Vector  # type: ignore


@dataclass(slots=True)
class StrutInstance:
    name: str
    length: float
    material: str
    group: str
    sequence: int
    ifc_guid: str


class StrutBuilder:
    """Create FreeCAD Arch/Part objects for each strut."""

    def __init__(self, params: DomeParameters):
        self.params = params
        self._doc: Optional[Any] = None
        # FreeCAD internal length unit is millimeters (mm). This project stores all
        # geometric parameters and tessellation coordinates in meters, so we scale
        # all geometry we send to FreeCAD by 1000.
        self._fc_unit_scale: float = 1000.0
        # Map (strut.index, "start"|"end") -> list of cut planes (point, normal)
        # Planes are stored as tuples of floats.
        self._endpoint_cut_planes: Dict[Tuple[int, str], List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = {}

    def _fc_len(self, meters: float) -> float:
        return float(meters) * float(self._fc_unit_scale)

    def _fc_point(self, p: Tuple[float, float, float]) -> Tuple[float, float, float]:
        s = float(self._fc_unit_scale)
        return (float(p[0]) * s, float(p[1]) * s, float(p[2]) * s)

    def ensure_document(self) -> Optional[Any]:
        try:
            import FreeCAD  # type: ignore
        except ImportError:  # pragma: no cover - outside FreeCAD
            return None

        doc = FreeCAD.ActiveDocument
        if doc is None:
            doc = FreeCAD.newDocument("GeodesicDome")
        return doc

    def create_struts(self, dome: TessellatedDome) -> List[StrutInstance]:
        doc = self.ensure_document()
        self._doc = doc
        instances: List[StrutInstance] = []

        helper_bases: List[Any] = []

        # Precompute node-fit cut planes so strut ends can be trimmed to fit together.
        self._endpoint_cut_planes = self._compute_endpoint_cut_planes(dome)

        grouped = self._group_by_length(dome.struts)

        for (group_label, struts) in grouped:
            for sequence, strut in enumerate(struts, start=1):
                name = f"Strut_{group_label}_{sequence:03d}"
                if doc:
                    base = self._make_arch_structure(doc, name, strut, group_label=group_label)
                    if base is not None:
                        helper_bases.append(base)
                guid = self._guid_for(name, strut)
                instances.append(
                    StrutInstance(
                        name=name,
                        length=strut.length,
                        material=self.params.material,
                        group=group_label,
                        sequence=sequence,
                        ifc_guid=guid,
                    )
                )

        if doc:
            try:
                doc.recompute()
            finally:
                # Ensure helper geometry stays hidden and user-manageable.
                try:
                    self._finalize_helper_geometry(doc, helper_bases)
                except Exception:
                    pass
        return instances

    @property
    def document(self) -> Optional[Any]:
        return self._doc

    def _make_arch_structure(self, doc: Any, name: str, strut: Strut, group_label: str) -> Any | None:
        try:
            import Arch  # type: ignore
            import FreeCAD  # type: ignore
            import Part  # type: ignore
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover - headless/dev mode
            return None

        shape = None
        bevel_debug: dict[str, object] = {}
        if self.params.use_bevels:
            shape, bevel_debug = self._build_beveled_shape(strut)
        else:
            bevel_debug = {"bevel_used": False, "reason": "bevel_disabled"}
        if shape is None:
            shape = self._simple_prism(strut)
            bevel_debug.setdefault("bevel_used", False)
            bevel_debug.setdefault("reason", "prismatic_fallback")
            # Populate intended end-plane metadata even for prismatic fallbacks.
            # (The geometry may still be cut; this is for QA/debugging and macros.)
            try:
                for end_label, key_point, key_normal in (
                    ("start", "start_point", "start_normal"),
                    ("end", "end_point", "end_normal"),
                ):
                    planes = self._endpoint_cut_planes.get((strut.index, end_label), [])
                    if planes:
                        pt_t, n_t = planes[0]
                        bevel_debug.setdefault(key_point, tuple(map(float, pt_t)))
                        bevel_debug.setdefault(key_normal, tuple(map(float, n_t)))
            except Exception:
                pass
        if shape is None:
            logging.error("Unable to create geometry for strut %s", name)
            return None
        self._log_shape_stats(name, strut, shape)

        base = doc.addObject("Part::Feature", f"{name}_Geom")
        base.Shape = shape
        # Store direction on the geometry object as well (useful for TechDraw views).
        try:
            if hasattr(base, "addProperty") and not hasattr(base, "StrutDirection"):
                base.addProperty(
                    "App::PropertyVector",
                    "StrutDirection",
                    "Dome",
                    "Unit vector along strut length (for drawings)",
                )
            if hasattr(base, "StrutDirection"):
                d = Vector(*strut.end) - Vector(*strut.start)
                if d.Length > 1e-12:
                    d.normalize()
                base.StrutDirection = d
        except Exception:
            pass
        base_view = getattr(base, "ViewObject", None)
        if base_view is not None:
            try:
                base_view.Visibility = False
            except Exception:
                pass

        # Keep helper solids in a dedicated group so they can be inspected/deleted if needed.
        self._add_to_helper_group(doc, base)

        structure = Arch.makeStructure(baseobj=base, name=name)
        self._assign_ifc_type(structure)
        structure.Label = name

        # Put visible struts in a dedicated folder for easy on/off toggling.
        self._add_to_struts_group(doc, structure, group_label=group_label)

        # Store metadata for future IFC exports.
        structure_view = getattr(structure, "ViewObject", None)
        if structure_view is not None:
            structure_view.ShapeColor = (0.72, 0.52, 0.04)
        if hasattr(structure, "IfcMaterial"):
            structure.IfcMaterial = self.params.material_spec().name

        # Ensure critical manufacturing metadata is available in IFC exports.
        try:
            if hasattr(structure, "addProperty") and not hasattr(structure, "StrutLengthM"):
                structure.addProperty(
                    "App::PropertyFloat",
                    "StrutLengthM",
                    "Dome",
                    "Strut length (m) from tessellation model",
                )
            if hasattr(structure, "StrutLengthM"):
                structure.StrutLengthM = float(strut.length)
            if hasattr(structure, "addProperty") and not hasattr(structure, "StrutLengthMM"):
                structure.addProperty(
                    "App::PropertyFloat",
                    "StrutLengthMM",
                    "Dome",
                    "Strut length (mm) from tessellation model",
                )
            if hasattr(structure, "StrutLengthMM"):
                structure.StrutLengthMM = float(strut.length) * 1000.0
            if hasattr(structure, "addProperty") and not hasattr(structure, "StrutFamily"):
                structure.addProperty(
                    "App::PropertyString",
                    "StrutFamily",
                    "Dome",
                    "Length family/group label",
                )
            if hasattr(structure, "StrutFamily"):
                structure.StrutFamily = str(group_label)
        except Exception:
            pass

        self._annotate_kerf(structure)
        self._store_direction(structure, strut)
        self._store_cut_plane_metadata(structure, bevel_debug)

        # Store per-end separation planes for debugging/macros.
        self._store_separation_plane_metadata(structure, strut)

        # Some Arch/TechDraw recomputes may flip base visibility; re-hide now.
        try:
            bv = getattr(base, "ViewObject", None)
            if bv is not None:
                bv.Visibility = False
        except Exception:
            pass

        return base

    def _struts_group(self, doc: Any) -> Any | None:
        """Return a group for visible strut objects."""
        try:
            objs = list(getattr(doc, "Objects", []) or [])
        except Exception:
            objs = []
        by_name = {str(getattr(o, "Name", "")): o for o in objs}

        group = by_name.get("Struts")
        if group is None:
            try:
                group = doc.addObject("App::DocumentObjectGroup", "Struts")
                try:
                    group.Label = "Struts"
                except Exception:
                    pass
            except Exception:
                return None
        return group

    def _struts_length_group(self, doc: Any, group_label: str) -> Any | None:
        parent = self._struts_group(doc)
        if parent is None:
            return None

        safe = "".join(ch for ch in str(group_label) if ch.isalnum() or ch in ("_", "-"))
        if not safe:
            safe = "Unknown"
        name = f"Struts_{safe}"

        try:
            objs = list(getattr(doc, "Objects", []) or [])
        except Exception:
            objs = []
        by_name = {str(getattr(o, "Name", "")): o for o in objs}

        group = by_name.get(name)
        if group is None:
            try:
                group = doc.addObject("App::DocumentObjectGroup", name)
                try:
                    group.Label = f"Struts {group_label}"
                except Exception:
                    pass
            except Exception:
                return None

        # Ensure subgroup is nested under the main Struts group.
        try:
            parent.addObject(group)
        except Exception:
            pass
        return group

    def _add_to_struts_group(self, doc: Any, obj: Any, group_label: str) -> None:
        group = self._struts_length_group(doc, group_label)
        if group is None:
            group = self._struts_group(doc)
        if group is None:
            return
        try:
            group.addObject(obj)
        except Exception:
            pass

    def _helper_group(self, doc: Any) -> Any | None:
        """Return a group for helper geometry objects."""
        try:
            objs = list(getattr(doc, "Objects", []) or [])
        except Exception:
            objs = []
        by_name = {str(getattr(o, "Name", "")): o for o in objs}

        group = by_name.get("StrutHelperGeometry")
        if group is None:
            try:
                group = doc.addObject("App::DocumentObjectGroup", "StrutHelperGeometry")
                try:
                    group.Label = "StrutHelperGeometry"
                except Exception:
                    pass
                vo = getattr(group, "ViewObject", None)
                if vo is not None:
                    try:
                        vo.Visibility = False
                    except Exception:
                        pass
            except Exception:
                return None
        return group

    def _add_to_helper_group(self, doc: Any, obj: Any) -> None:
        group = self._helper_group(doc)
        if group is None:
            return
        try:
            group.addObject(obj)
        except Exception:
            pass

    def _finalize_helper_geometry(self, doc: Any, helper_bases: List[Any]) -> None:
        """Post-recompute cleanup to prevent visible duplicates."""
        # Hide every Strut_*_Geom to prevent double-visualization.
        for obj in helper_bases:
            try:
                vo = getattr(obj, "ViewObject", None)
                if vo is not None:
                    vo.Visibility = False
            except Exception:
                pass

        # Also hide the group if it exists.
        group = self._helper_group(doc)
        if group is not None:
            try:
                vo = getattr(group, "ViewObject", None)
                if vo is not None:
                    vo.Visibility = False
            except Exception:
                pass

    def _assign_ifc_type(self, structure: Any) -> None:
        if not hasattr(structure, "IfcType"):
            return
        try:
            structure.IfcType = "IfcMember"
        except Exception:
            try:
                structure.IfcType = "Beam"
            except Exception:
                pass

    def _build_beveled_shape(self, strut: Strut):
        try:
            import FreeCAD  # type: ignore
            import Part  # type: ignore
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover - headless/dev mode
            return None, {}

        start = Vector(*self._fc_point(strut.start))
        end = Vector(*self._fc_point(strut.end))
        direction = end - start
        length = direction.Length
        stock_max = self._fc_len(max(self.params.stock_width_m, self.params.stock_height_m))
        min_length = stock_max * 0.5
        prism_only_length = stock_max * 3.0
        if length <= min_length:
            return None, {"bevel_used": False}

        axis_vec = Vector(direction)
        axis_vec.normalize()
        padding = self._fc_len(self.params.kerf_m * 2 + self.params.clearance_m)
        if length <= prism_only_length:
            # Strut is too short to survive bevel cuts; use rectangular prism.
            midpoint_vec = (start + end) * 0.5
            solid = self._make_aligned_box(length, axis_vec, start, padding, midpoint_vec=midpoint_vec)
            if solid is None:
                return None, {"bevel_used": False, "reason": "too_short_for_bevel_failed"}
            span = float(length + padding + self._fc_len(self.params.radius_m))
            # Apply primary endpoint cuts even on the prism so it still fits nodes/belt.
            solid = self._apply_primary_endpoint_cuts_to_prism(solid, strut, keep_point=midpoint_vec, span=span)
            solid2 = self._trim_to_belt_top_if_needed(solid, strut, keep_point=midpoint_vec, span=span)
            debug: dict[str, object] = {"bevel_used": False, "reason": "too_short_for_bevel"}
            try:
                for end_label, key_point, key_normal in (
                    ("start", "start_point", "start_normal"),
                    ("end", "end_point", "end_normal"),
                ):
                    planes = self._endpoint_cut_planes.get((strut.index, end_label), [])
                    if planes:
                        pt_t, n_t = planes[0]
                        debug[key_point] = tuple(map(float, pt_t))
                        debug[key_normal] = tuple(map(float, n_t))
            except Exception:
                pass
            return (solid2 if solid2 is not None else solid, debug)
        midpoint_vec = (start + end) * 0.5
        solid = self._make_aligned_box(length, axis_vec, start, padding, midpoint_vec)

        keep_point = midpoint_vec

        # Node-fit algorithm:
        # - Cut both ends to a common node plane (radial plane through node)
        # - Add two separation planes per end so neighboring struts don't overlap in the node plane
        # - Apply separation planes only to a short end-cap region so the whole strut doesn't become wedge-shaped
        bevel_debug: dict[str, object] = {"bevel_used": True}

        span = float(length + padding + self._fc_len(self.params.radius_m))
        cap_len = max(stock_max * 2.0, stock_max + self._fc_len(self.params.clearance_m + self.params.kerf_m))

        solid, start_debug = self._apply_endpoint_node_fit_cuts(
            solid,
            strut,
            end_label="start",
            node_point=start,
            axis_from_node=axis_vec,
            keep_point=keep_point,
            span=span,
            cap_length=cap_len,
        )
        if solid is None:
            return None, {**bevel_debug, **start_debug, "bevel_used": False, "reason": "start_cut_failed"}

        solid, end_debug = self._apply_endpoint_node_fit_cuts(
            solid,
            strut,
            end_label="end",
            node_point=end,
            axis_from_node=-axis_vec,
            keep_point=keep_point,
            span=span,
            cap_length=cap_len,
        )
        if solid is None:
            return None, {**bevel_debug, **end_debug, "bevel_used": False, "reason": "end_cut_failed"}

        # If hemisphere clipping is active, the lowest "vertical" struts should terminate on top
        # of the belt struts (ring made of boundary struts). We do NOT cut the belt struts for this.
        solid2 = self._trim_to_belt_top_if_needed(solid, strut, keep_point=keep_point, span=span)
        if solid2 is None:
            return None, {**bevel_debug, "bevel_used": False, "reason": "belt_top_trim_failed"}
        solid = solid2

        # Store the primary node planes for the existing debug fields/macros.
        if "start_node_plane" in start_debug:
            point_t, normal_t = start_debug["start_node_plane"]  # type: ignore[misc]
            bevel_debug["start_point"] = point_t
            bevel_debug["start_normal"] = normal_t
        if "end_node_plane" in end_debug:
            point_t, normal_t = end_debug["end_node_plane"]  # type: ignore[misc]
            bevel_debug["end_point"] = point_t
            bevel_debug["end_normal"] = normal_t

        # Longitudinal outer-face split cuts (frame/panel seating):
        # For each strut edge we typically have 2 adjacent panels. Their planes both contain the
        # strut axis, so their normals are perpendicular to the strut axis. We cut the *outer face*
        # into two facets so each facet is parallel to its adjacent panel plane.
        outer_planes = self._outer_face_split_planes(strut, axis_vec, midpoint_vec)
        if outer_planes:
            for idx, (pt, n) in enumerate(outer_planes, start=1):
                solid2 = self._cut_with_plane(solid, pt, n, keep_point, span)
                if solid2 is None:
                    bevel_debug[f"outer{idx}_point"] = tuple(map(float, pt))
                    bevel_debug[f"outer{idx}_normal"] = tuple(map(float, n))
                    return None, {**bevel_debug, "bevel_used": False, "reason": f"outer_cut_failed_{idx}"}
                solid = solid2
                bevel_debug[f"outer{idx}_point"] = tuple(map(float, pt))
                bevel_debug[f"outer{idx}_normal"] = tuple(map(float, n))
        else:
            # Fallback: the legacy tangent-side trim (keeps strut from protruding past the sphere).
            side_plane = self._side_plane(strut)
            if side_plane is not None:
                plane_point, plane_normal = side_plane
                solid = self._cut_with_plane(solid, plane_point, plane_normal, keep_point, span)
                if solid is None:
                    bevel_debug["side_point"] = tuple(map(float, plane_point))
                    bevel_debug["side_normal"] = tuple(map(float, plane_normal))
                    return None, {**bevel_debug, "bevel_used": False, "reason": "side_cut_failed"}
                bevel_debug["side_point"] = tuple(map(float, plane_point))
                bevel_debug["side_normal"] = tuple(map(float, plane_normal))

        bbox = getattr(solid, "BoundBox", None)
        if bbox is not None:
            try:
                diag = bbox.DiagonalLength
            except Exception:
                diag = None
            if diag is None or diag < strut.length * 0.9:
                logging.debug(
                    "Bevel trimmed strut %d diag %s << expected %.3f; using prismatic fallback",
                    strut.index,
                    f"{diag:.3f}" if diag is not None else "invalid",
                    strut.length,
                )
                fallback = self._simple_prism(strut)
                if fallback is not None:
                    return fallback, {**bevel_debug, "bevel_used": False, "reason": "bbox_diag_too_small"}

        if not self._is_shape_valid(solid):
            logging.debug(
                "Bevel boolean failed for strut length %.3f; using prismatic fallback",
                strut.length,
            )
            fallback = self._simple_prism(strut)
            if fallback is None:
                logging.error("Prismatic fallback failed for strut length %.3f", strut.length)
                return None, {**bevel_debug, "bevel_used": False, "reason": "fallback_failed"}
            return fallback, {**bevel_debug, "bevel_used": False, "reason": "invalid_bevel_shape"}
        return solid, bevel_debug

    def _trim_to_belt_top_if_needed(self, solid, strut: Strut, keep_point, span: float):
        """Trim struts that terminate at the belt so they sit on top of the belt struts.

        - Applies only when hemisphere clipping is enabled.
        - Applies only to non-belt struts (i.e., not both endpoints on the belt plane).
        - Cuts with a horizontal plane at z = belt_height + stock_height/2.
        """
        try:
            import FreeCAD  # type: ignore
        except ImportError:  # pragma: no cover
            return solid

        hemi = float(getattr(self.params, "hemisphere_ratio", 1.0) or 1.0)
        if hemi >= 0.999999:
            return solid

        belt_height = float(self.params.radius_m) * (1.0 - 2.0 * hemi)
        eps = max(1e-6, float(self.params.radius_m) * 1e-5)
        z0 = float(strut.start[2])
        z1 = float(strut.end[2])

        start_on_belt = abs(z0 - belt_height) <= eps
        end_on_belt = abs(z1 - belt_height) <= eps
        if start_on_belt and end_on_belt:
            # Belt strut: do not trim by belt-top.
            return solid

        # Seat struts onto the belt. Using the exact half-height can leave a visible gap
        # in some workflows; bias slightly lower by the configured clearance.
        belt_top_z_m = belt_height + float(self.params.stock_height_m) * 0.5 - float(self.params.clearance_m)
        belt_top_z = self._fc_len(belt_top_z_m)

        # Primary case: strut endpoint is on the belt plane and the strut continues upward.
        touches_belt = start_on_belt or end_on_belt
        should_trim = False
        if touches_belt and max(z0, z1) > belt_height + eps:
            should_trim = True

        # Fallback robustness: if a non-belt strut's solid extends below belt_top,
        # trim it even when the tessellation endpoints don't land exactly on the belt plane.
        if not should_trim:
            try:
                bb = getattr(solid, "BoundBox", None)
                if bb is not None and float(bb.ZMin) < float(belt_top_z) - 2e-4:
                    should_trim = True
            except Exception:
                pass

        if not should_trim:
            return solid
        # Center the finite cutting plane near the strut; otherwise a plane at (0,0,z)
        # can miss geometry located around the dome radius.
        try:
            kp = keep_point if isinstance(keep_point, FreeCAD.Vector) else FreeCAD.Vector(keep_point)
            plane_point = FreeCAD.Vector(float(kp.x), float(kp.y), float(belt_top_z))
        except Exception:
            plane_point = FreeCAD.Vector(0.0, 0.0, float(belt_top_z))
        plane_normal = FreeCAD.Vector(0.0, 0.0, 1.0)

        # Always keep the half-space above the belt-top plane.
        keep_above = plane_point + plane_normal * float(max(1.0, self._fc_len(0.001)))
        trimmed = self._cut_with_plane(solid, plane_point, plane_normal, keep_above, span)
        return trimmed if trimmed is not None and self._is_shape_valid(trimmed) else None

    def _outer_face_split_planes(self, strut: Strut, axis_vec, midpoint_vec):
        """Return up to 2 planes that split the strut outer face for panel seating.

        The planes are parallel to the two adjacent panel planes and pass through a ridge line
        located on the strut's outer face (roughly at +width/2 along the radial projection).
        """
        try:
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover
            return []

        normals: list[tuple[float, float, float]] = []
        pn = getattr(strut, "primary_normal", None)
        if pn is not None:
            normals.append(pn)
        sn = getattr(strut, "secondary_normal", None)
        if sn is not None:
            normals.append(sn)
        if not normals:
            return []

        a = Vector(axis_vec)
        if a.Length == 0:
            return []
        a.normalize()

        m = Vector(midpoint_vec)

        # Outward direction across the strut width: project radial to plane perpendicular to axis.
        w = Vector(m)
        w = w - a * float(w.dot(a))
        if w.Length <= 1e-9:
            # Fallback: any perpendicular.
            w = a.cross(Vector(0, 0, 1))
            if w.Length <= 1e-9:
                w = a.cross(Vector(0, 1, 0))
        if w.Length <= 1e-9:
            return []
        w.normalize()

        ridge = m + w * float(self._fc_len(self.params.stock_width_m) * 0.5)

        planes = []
        for n_raw in normals[:2]:
            n = Vector(*n_raw)
            if n.Length <= 1e-12:
                continue
            # The split plane is meant to contain a ridge line parallel to the strut axis.
            # That requires the plane normal to be perpendicular to the axis.
            # Panel normals can have a small axial component (e.g. non-planar spherical polygons
            # approximated by a best-fit plane), which would otherwise create a diagonal cut.
            n = n - a * float(n.dot(a))
            if n.Length <= 1e-12:
                continue
            n.normalize()
            # Ensure normal points "inward" relative to the outer direction so keep-point selection is stable.
            if float(n.dot(w)) > 0:
                n = -n
            planes.append((ridge, n))

        # If the two normals are effectively identical, keep just one.
        if len(planes) == 2:
            try:
                if planes[0][1].getAngle(planes[1][1]) < 1e-3:
                    planes = planes[:1]
            except Exception:
                pass
        return planes

    def _apply_endpoint_node_fit_cuts(
        self,
        solid,
        strut: Strut,
        end_label: str,
        node_point,
        axis_from_node,
        keep_point,
        span: float,
        cap_length: float,
    ):
        """Apply node-fit cuts at a single endpoint.

        Cuts are applied only to a short end-cap to keep the strut mostly prismatic.
        Returns (new_solid_or_None, debug_dict).
        """
        try:
            import Part  # type: ignore
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover
            return solid, {}

        debug: dict[str, object] = {}
        end_key = (strut.index, end_label)
        planes = self._endpoint_cut_planes.get(end_key, [])
        if not planes:
            debug["reason"] = f"missing_node_planes_{end_label}"
            return None, debug

        # Split strut into end-cap + remainder using a plane perpendicular to the strut axis.
        axis_vec = Vector(axis_from_node)
        if axis_vec.Length == 0:
            debug["reason"] = f"zero_axis_{end_label}"
            return None, debug
        axis_vec.normalize()

        node_vec = Vector(node_point)
        # Keep cap split plane within the strut span; if it goes past the other end,
        # half-space selection can become unstable and separation cuts can collapse to empty.
        strut_len_fc = float(self._fc_len(strut.length))
        max_cap = float(max(1e-3, min(cap_length, strut_len_fc * 0.45)))

        # A stable reference point inside the end-cap along the axis.
        # Keep it *deep* into the cap so that endpoint planes slightly away from the node
        # (e.g. belt seating at belt_height + stock_height/2) still keep the correct half-space.
        end_interior = node_vec + axis_vec * float(max(1e-3, max_cap * 0.8))

        cap_plane_point = node_vec + axis_vec * float(max_cap)

        # End cap: keep side containing node.
        cap = self._cut_with_plane(solid, cap_plane_point, axis_vec, node_vec, span)
        if cap is None:
            debug["reason"] = f"cap_split_failed_{end_label}"
            return None, debug
        # Remainder: keep side containing the midpoint/keep_point.
        remainder = self._cut_with_plane(solid, cap_plane_point, axis_vec, keep_point, span)
        if remainder is None:
            debug["reason"] = f"remainder_split_failed_{end_label}"
            return None, debug

        # Apply node plane + separation planes to the end-cap only.
        cap_cut = cap
        for idx, (pt_t, n_t) in enumerate(planes):
            pt = Vector(*pt_t)
            n = Vector(*n_t)
            if n.Length == 0:
                continue
            # Keep the half-space containing a point inside this end-cap.
            tmp = self._cut_with_plane(cap_cut, pt, n, end_interior, span)
            if tmp is None or not self._is_shape_valid(tmp):
                # If a separation plane over-trims, keep the node-plane-only result.
                # If the node plane itself fails (idx=0), abort.
                if idx == 0:
                    debug["reason"] = f"node_plane_cut_failed_{end_label}"
                    return None, debug
                debug["reason"] = f"sep_plane_cut_skipped_{end_label}_{idx}"
                break
            cap_cut = tmp

        # Fuse back together.
        try:
            fused = remainder.fuse(cap_cut)
        except Exception:
            # As a fallback, return a compound (still printable/inspectable).
            try:
                fused = Part.makeCompound([remainder, cap_cut])
            except Exception:
                debug["reason"] = f"fuse_failed_{end_label}"
                return None, debug

        # Record the primary node plane (first plane in list).
        if planes:
            point_t, normal_t = planes[0]
            debug[f"{end_label}_node_plane"] = (tuple(map(float, point_t)), tuple(map(float, normal_t)))
        return fused, debug

    def _compute_endpoint_cut_planes(
        self, dome: TessellatedDome
    ) -> Dict[Tuple[int, str], List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]]:
        """Compute node-fit cut planes for each strut endpoint.

        Per endpoint we return:
        - Node end plane (either radial/tangent or strut-axis/square, depending on parameters)
        - Optional separation planes (between this strut and its neighbors around the node)
        """
        nodes = dome.nodes
        # node_index -> list[(strut, end_label, dir_vec_from_node)]
        incident: Dict[int, List[Tuple[Strut, str, Tuple[float, float, float]]]] = {}

        for s in dome.struts:
            sx, sy, sz = s.start
            ex, ey, ez = s.end
            incident.setdefault(s.start_index, []).append((s, "start", (ex - sx, ey - sy, ez - sz)))
            incident.setdefault(s.end_index, []).append((s, "end", (sx - ex, sy - ey, sz - ez)))

        def _norm(v: Tuple[float, float, float]) -> float:
            return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

        def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
            n = _norm(v)
            if n <= 1e-12:
                return (0.0, 0.0, 0.0)
            return (v[0] / n, v[1] / n, v[2] / n)

        def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )

        def _sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
            return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

        def _mul(a: Tuple[float, float, float], s: float) -> Tuple[float, float, float]:
            return (a[0] * s, a[1] * s, a[2] * s)

        # Result mapping.
        endpoint_planes: Dict[Tuple[int, str], List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = {}

        plane_mode = str(getattr(self.params, "node_fit_plane_mode", "radial") or "radial")
        use_separation = bool(getattr(self.params, "node_fit_use_separation_planes", True))

        for node_idx, items in incident.items():
            if not items:
                continue
            p = nodes[node_idx]
            # Most nodes lie on the sphere surface, where the "node plane" normal is the
            # radial vector. When hemisphere clipping is enabled we add a *planar belt*
            # (see `tessellation._add_planar_belt`) whose nodes lie on the belt Z plane.
            #
            # IMPORTANT: Truncation also creates nodes that are not on the sphere radius,
            # but they are NOT belt nodes. Therefore belt detection must be based on Z,
            # not on the radius deviation.
            hemisphere_ratio = float(getattr(self.params, "hemisphere_ratio", 1.0) or 1.0)
            belt_node = False
            belt_height = None
            belt_eps = max(1e-6, float(self.params.radius_m) * 1e-5)
            if hemisphere_ratio < 1.0:
                belt_height = float(self.params.radius_m) * (1.0 - 2.0 * hemisphere_ratio)
                if abs(float(p[2]) - belt_height) <= belt_eps:
                    belt_node = True

            # Radial normal for regular spherical nodes.
            r = _normalize(p)
            if _norm(r) <= 1e-12:
                # Degenerate node at origin; skip node-fit.
                continue
            # At belt nodes we want a consistent horizontal reference plane in XY.
            if belt_node:
                r = (0.0, 0.0, 1.0)

            # Build orthonormal basis (u, v) for the tangent plane at the node.
            ref = (1.0, 0.0, 0.0) if abs(r[0]) < 0.9 else (0.0, 1.0, 0.0)
            u = _normalize(_cross(r, ref))
            v = _normalize(_cross(r, u))

            # Compute tangent directions and angles.
            enriched: List[Tuple[float, Strut, str, Tuple[float, float, float], Tuple[float, float, float]]] = []
            for s, end_label, d in items:
                a = _normalize(d)
                # Project to tangent plane.
                t = _sub(a, _mul(r, _dot(a, r)))
                t = _normalize(t)
                if _norm(t) <= 1e-12:
                    # At belt nodes r is vertical, so near-vertical struts can project to ~0.
                    # We still need to keep them in the endpoint plane set so they get the
                    # belt-top trim. Use XY projection (or a stable fallback) for ordering.
                    if belt_node:
                        t = _normalize((a[0], a[1], 0.0))
                        if _norm(t) <= 1e-12:
                            t = u
                    else:
                        continue
                ang = math.atan2(_dot(t, v), _dot(t, u))
                enriched.append((ang, s, end_label, a, t))

            if not enriched:
                continue

            enriched.sort(key=lambda row: row[0])

            # Identify belt-ring members incident to this belt node (typically exactly 2).
            belt_members: List[Tuple[Strut, Tuple[float, float, float]]] = []
            if belt_node and belt_height is not None:
                for _ang, s0, _end0, _a0, t0 in enriched:
                    try:
                        if (
                            abs(float(s0.start[2]) - float(belt_height)) <= belt_eps
                            and abs(float(s0.end[2]) - float(belt_height)) <= belt_eps
                        ):
                            belt_members.append((s0, t0))
                    except Exception:
                        continue

            count = len(enriched)
            for i, (_ang, s, end_label, a, t) in enumerate(enriched):
                prev_t = enriched[(i - 1) % count][4]
                next_t = enriched[(i + 1) % count][4]

                # Primary node end plane.
                planes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []

                # 'radial' matches the sphere tangent (legacy behavior).
                # 'axis' makes a square cut at the endpoint, so the strut runs straight node-to-node.
                if belt_node and belt_height is not None:
                    # Belt handling:
                    # - Belt struts (both endpoints on belt plane) define the belt itself.
                    #   Keep their endpoint cut on the belt plane.
                    # - Non-belt struts that terminate at the belt should be trimmed to sit on TOP
                    #   of the belt (belt plane + half stock height).
                    is_belt_strut = (
                        abs(float(s.start[2]) - belt_height) <= belt_eps
                        and abs(float(s.end[2]) - belt_height) <= belt_eps
                    )
                    if is_belt_strut:
                        # Belt struts define the belt ring. Use a proper miter plane based on
                        # the two belt-member directions at this node (in the belt XY plane).
                        # This avoids odd "tooth" cuts that can happen with a radial plane.
                        mate_t = None
                        for s_other, t_other in belt_members:
                            if getattr(s_other, "index", None) != getattr(s, "index", None):
                                mate_t = t_other
                                break
                        if mate_t is not None:
                            bis = _normalize((t[0] + mate_t[0], t[1] + mate_t[1], t[2] + mate_t[2]))
                            if _norm(bis) <= 1e-12:
                                bis = t
                        else:
                            bis = t
                        n_miter = _normalize(_cross(r, bis))
                        if _norm(n_miter) <= 1e-12:
                            n_miter = a
                        planes.append((self._fc_point(p), n_miter))
                        # Do not apply separation planes to belt-ring members.
                        # Separation planes are mainly for multi-strut nodes; on belt nodes
                        # they can carve "teeth" into the belt when vertical struts are present.
                        # The miter plane above should be sufficient to keep belt members from
                        # overlapping while preserving clean ends.
                        endpoint_planes[(s.index, end_label)] = planes
                        continue
                    else:
                        # Non-belt struts: trim to sit on top of the belt.
                        cut_z = float(belt_height) + float(self.params.stock_height_m) * 0.5 - float(self.params.clearance_m)
                        cut_point = (float(p[0]), float(p[1]), cut_z)
                        planes.append((self._fc_point(cut_point), r))
                        # Do not apply separation planes at belt nodes for belt-seated ends.
                        # The belt seat is a simple horizontal plane; separation planes here
                        # tend to carve "teeth" into the inner face of the strut end-cap.
                        endpoint_planes[(s.index, end_label)] = planes
                        continue
                else:
                    if plane_mode == "axis":
                        planes.append((self._fc_point(p), a))
                    else:
                        planes.append((self._fc_point(p), r))

                if not use_separation:
                    endpoint_planes[(s.index, end_label)] = planes
                    continue

                # Separation plane between prev and this.
                bis_prev = _normalize((prev_t[0] + t[0], prev_t[1] + t[1], prev_t[2] + t[2]))
                if _norm(bis_prev) > 1e-12:
                    n1 = _normalize(_cross(r, bis_prev))
                    if _dot(t, n1) < 0:
                        n1 = (-n1[0], -n1[1], -n1[2])
                    planes.append((planes[0][0], n1))

                # Separation plane between this and next.
                bis_next = _normalize((t[0] + next_t[0], t[1] + next_t[1], t[2] + next_t[2]))
                if _norm(bis_next) > 1e-12:
                    n2 = _normalize(_cross(r, bis_next))
                    if _dot(t, n2) < 0:
                        n2 = (-n2[0], -n2[1], -n2[2])
                    planes.append((planes[0][0], n2))

                endpoint_planes[(s.index, end_label)] = planes

        return endpoint_planes

    def _store_separation_plane_metadata(self, structure: Any, strut: Strut) -> None:
        if not hasattr(structure, "addProperty"):
            return
        try:
            import FreeCAD  # type: ignore

            def _add(name: str, typ: str, group: str, desc: str) -> None:
                if hasattr(structure, name):
                    return
                try:
                    structure.addProperty(typ, name, group, desc)
                except Exception:
                    pass

            for suffix in ("Start", "End"):
                _add(f"{suffix}Sep1Point", "App::PropertyVector", "Strut", f"{suffix} separation plane 1 point")
                _add(f"{suffix}Sep1Normal", "App::PropertyVector", "Strut", f"{suffix} separation plane 1 normal")
                _add(f"{suffix}Sep2Point", "App::PropertyVector", "Strut", f"{suffix} separation plane 2 point")
                _add(f"{suffix}Sep2Normal", "App::PropertyVector", "Strut", f"{suffix} separation plane 2 normal")

            for end_label, suffix in (("start", "Start"), ("end", "End")):
                planes = self._endpoint_cut_planes.get((strut.index, end_label), [])
                # planes[0] is node plane; planes[1] and planes[2] (if present) are separation planes.
                if len(planes) >= 2:
                    pt, n = planes[1]
                    structure.__setattr__(f"{suffix}Sep1Point", FreeCAD.Vector(*pt))
                    structure.__setattr__(f"{suffix}Sep1Normal", FreeCAD.Vector(*n))
                if len(planes) >= 3:
                    pt, n = planes[2]
                    structure.__setattr__(f"{suffix}Sep2Point", FreeCAD.Vector(*pt))
                    structure.__setattr__(f"{suffix}Sep2Normal", FreeCAD.Vector(*n))
        except Exception:  # pragma: no cover
            pass

    def _store_cut_plane_metadata(self, structure: Any, debug: dict[str, object]) -> None:
        if not hasattr(structure, "addProperty"):
            return
        if not debug:
            return
        try:
            import FreeCAD  # type: ignore

            def _add(name: str, typ: str, group: str, desc: str) -> None:
                if hasattr(structure, name):
                    return
                try:
                    structure.addProperty(typ, name, group, desc)
                except Exception:
                    pass

            _add("BevelUsed", "App::PropertyBool", "Strut", "Whether beveled end cuts were applied")
            _add("StartCutPoint", "App::PropertyVector", "Strut", "Start cut plane point")
            _add("StartCutNormal", "App::PropertyVector", "Strut", "Start cut plane normal")
            _add("EndCutPoint", "App::PropertyVector", "Strut", "End cut plane point")
            _add("EndCutNormal", "App::PropertyVector", "Strut", "End cut plane normal")
            _add("SideCutPoint", "App::PropertyVector", "Strut", "Optional side cut plane point")
            _add("SideCutNormal", "App::PropertyVector", "Strut", "Optional side cut plane normal")
            _add("OuterCut1Point", "App::PropertyVector", "Strut", "Outer split cut plane 1 point")
            _add("OuterCut1Normal", "App::PropertyVector", "Strut", "Outer split cut plane 1 normal")
            _add("OuterCut2Point", "App::PropertyVector", "Strut", "Outer split cut plane 2 point")
            _add("OuterCut2Normal", "App::PropertyVector", "Strut", "Outer split cut plane 2 normal")
            _add("BevelFailureReason", "App::PropertyString", "Strut", "Reason bevel was not applied")

            structure.BevelUsed = bool(debug.get("bevel_used", False))
            if "start_point" in debug:
                x, y, z = debug["start_point"]  # type: ignore[misc]
                structure.StartCutPoint = FreeCAD.Vector(float(x), float(y), float(z))
            if "start_normal" in debug:
                x, y, z = debug["start_normal"]  # type: ignore[misc]
                structure.StartCutNormal = FreeCAD.Vector(float(x), float(y), float(z))
            if "end_point" in debug:
                x, y, z = debug["end_point"]  # type: ignore[misc]
                structure.EndCutPoint = FreeCAD.Vector(float(x), float(y), float(z))
            if "end_normal" in debug:
                x, y, z = debug["end_normal"]  # type: ignore[misc]
                structure.EndCutNormal = FreeCAD.Vector(float(x), float(y), float(z))
            if "side_point" in debug:
                x, y, z = debug["side_point"]  # type: ignore[misc]
                structure.SideCutPoint = FreeCAD.Vector(float(x), float(y), float(z))
            if "side_normal" in debug:
                x, y, z = debug["side_normal"]  # type: ignore[misc]
                structure.SideCutNormal = FreeCAD.Vector(float(x), float(y), float(z))
            if "outer1_point" in debug:
                x, y, z = debug["outer1_point"]  # type: ignore[misc]
                structure.OuterCut1Point = FreeCAD.Vector(float(x), float(y), float(z))
            if "outer1_normal" in debug:
                x, y, z = debug["outer1_normal"]  # type: ignore[misc]
                structure.OuterCut1Normal = FreeCAD.Vector(float(x), float(y), float(z))
            if "outer2_point" in debug:
                x, y, z = debug["outer2_point"]  # type: ignore[misc]
                structure.OuterCut2Point = FreeCAD.Vector(float(x), float(y), float(z))
            if "outer2_normal" in debug:
                x, y, z = debug["outer2_normal"]  # type: ignore[misc]
                structure.OuterCut2Normal = FreeCAD.Vector(float(x), float(y), float(z))
            structure.BevelFailureReason = str(debug.get("reason", ""))
        except Exception:  # pragma: no cover - FreeCAD-specific failure
            pass

    def _simple_prism(self, strut: Strut):
        try:
            import FreeCAD  # type: ignore
            import Part  # type: ignore
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover - headless/dev mode
            return None

        start = Vector(*self._fc_point(strut.start))
        end = Vector(*self._fc_point(strut.end))
        direction = end - start
        length = direction.Length
        if length == 0:
            return None

        axis_vec = Vector(direction)
        axis_vec.normalize()

        padding = self._fc_len(self.params.clearance_m)
        midpoint_vec = (start + end) * 0.5
        solid = self._make_aligned_box(length, axis_vec, start, padding, midpoint_vec)
        if solid is None:
            return None

        span = float(length + padding + self._fc_len(self.params.radius_m))

        # If bevels are enabled but we fell back to a prism, still apply the node-fit
        # endpoint cuts (node plane + optional separation planes) so the strut is
        # actually trimmed for node connection.
        if bool(getattr(self.params, "use_bevels", False)):
            stock_max = float(self._fc_len(max(self.params.stock_width_m, self.params.stock_height_m)))
            cap_len = max(stock_max * 2.0, stock_max + float(self._fc_len(self.params.clearance_m + self.params.kerf_m)))
            solid2, _ = self._apply_endpoint_node_fit_cuts(
                solid,
                strut,
                end_label="start",
                node_point=start,
                axis_from_node=axis_vec,
                keep_point=midpoint_vec,
                span=span,
                cap_length=cap_len,
            )
            if solid2 is not None:
                solid = solid2
            solid3, _ = self._apply_endpoint_node_fit_cuts(
                solid,
                strut,
                end_label="end",
                node_point=end,
                axis_from_node=-axis_vec,
                keep_point=midpoint_vec,
                span=span,
                cap_length=cap_len,
            )
            if solid3 is not None:
                solid = solid3

            # Also apply the outer-face split planes used for panel seating.
            outer_planes = self._outer_face_split_planes(strut, axis_vec, midpoint_vec)
            if outer_planes:
                for pt, n in outer_planes:
                    solid4 = self._cut_with_plane(solid, pt, n, midpoint_vec, span)
                    if solid4 is not None and self._is_shape_valid(solid4):
                        solid = solid4

        # Even when bevels/booleans fail and we fall back to a prism, we still need
        # the hemisphere belt join to work (vertical-ish struts should not hang below the belt).
        solid2 = self._trim_to_belt_top_if_needed(solid, strut, keep_point=midpoint_vec, span=span)
        return solid2 if solid2 is not None else solid

    def _apply_primary_endpoint_cuts_to_prism(self, solid, strut: Strut, keep_point, span: float):
        """Apply only the primary node plane at each end to a prismatic fallback solid."""
        try:
            from FreeCAD import Vector  # type: ignore
        except Exception:  # pragma: no cover
            return solid

        out = solid
        for end_label in ("start", "end"):
            planes = self._endpoint_cut_planes.get((strut.index, end_label), [])
            if not planes:
                continue
            pt_t, n_t = planes[0]
            pt = Vector(*pt_t)
            n = Vector(*n_t)
            if n.Length <= 1e-12:
                continue
            tmp = self._cut_with_plane(out, pt, n, keep_point, span)
            if tmp is not None and self._is_shape_valid(tmp):
                out = tmp
        return out

    def _endpoint_plane(
        self,
        point: Tuple[float, float, float],
        normal: Tuple[float, float, float] | None,
        fallback_direction,
        keep_point,
    ) -> Optional[Tuple['Vector', 'Vector']]:
        try:
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover - headless/dev mode
            return None

        normal_vec = Vector(*normal) if normal is not None else Vector(fallback_direction)
        if normal_vec.Length == 0:
            normal_vec = Vector(fallback_direction)
        if normal_vec.Length == 0:
            return None
        normal_vec.normalize()
        anchor = Vector(*point)
        keep_vec = keep_point if isinstance(keep_point, Vector) else Vector(keep_point)
        if (keep_vec - anchor).dot(normal_vec) < 0:
            normal_vec = -normal_vec
        return anchor, normal_vec

    def _side_plane(self, strut: Strut) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        try:
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover - headless/dev mode
            return None

        start = Vector(*self._fc_point(strut.start))
        end = Vector(*self._fc_point(strut.end))
        midpoint = (start + end) * 0.5
        normal = -Vector(midpoint)  # aim width outward so width normal points to sphere center
        if normal.Length == 0:
            return None
        normal.normalize()
        return (
            (midpoint.x, midpoint.y, midpoint.z),
            (normal.x, normal.y, normal.z),
        )

    def _log_shape_stats(self, name: str, strut: Strut, shape) -> None:
        bbox = getattr(shape, "BoundBox", None)
        diag = None
        if bbox is not None:
            try:
                diag = bbox.DiagonalLength
            except Exception:
                diag = None
        volume = getattr(shape, "Volume", None)
        if diag is not None and diag < self._fc_len(strut.length) * 0.25:
            logging.warning(
                "Strut %s bounding diag %.5f m far shorter than expected %.5f m",
                name,
                float(diag) / float(self._fc_unit_scale),
                float(strut.length),
            )
        if volume is not None:
            # FreeCAD volume is in (mm^3) when working in mm units.
            vol_m3 = float(volume) / float(self._fc_unit_scale ** 3)
            if vol_m3 <= 1e-6:
                logging.warning("Strut %s volume %.6g m^3 extremely small", name, vol_m3)

    def _make_aligned_box(self, length: float, axis_vec, start, padding: float, midpoint_vec=None):
        import FreeCAD  # type: ignore
        import Part  # type: ignore
        from FreeCAD import Vector  # type: ignore

        total_length = float(length + padding)
        stock_w = float(self._fc_len(self.params.stock_width_m))
        stock_h = float(self._fc_len(self.params.stock_height_m))
        box = Part.makeBox(total_length, stock_w, stock_h)
        center_offset = FreeCAD.Placement(
            Vector(-float(padding) / 2.0, -stock_w / 2.0, -stock_h / 2.0),
            FreeCAD.Rotation(),
        )

        # Align X with strut axis, then rotate around X so Y aligns with radial projection.
        x_axis = Vector(axis_vec)
        x_axis.normalize()
        rotation = FreeCAD.Rotation(Vector(1, 0, 0), x_axis)
        if midpoint_vec is not None:
            radial = Vector(midpoint_vec)
            if radial.Length > 0:
                radial.normalize()
                # Express radial in the current frame (after aligning X).
                radial_local = rotation.inverted().multVec(radial)
                # Project to YZ plane (remove X component).
                radial_local = Vector(0, radial_local.y, radial_local.z)
                if radial_local.Length > 0:
                    radial_local.normalize()
                    # Angle from +Y toward +Z around +X.
                    angle = FreeCAD.Vector(0, 1, 0).getAngle(radial_local)
                    sign = 1.0 if radial_local.z >= 0 else -1.0
                    rotation = rotation.multiply(FreeCAD.Rotation(Vector(1, 0, 0), sign * angle * 180.0 / 3.141592653589793))
        base = FreeCAD.Placement(start, rotation)
        box.Placement = base.multiply(center_offset)
        return box

    def _cut_with_plane(self, shape, point, normal, keep_point, span):
        try:
            import FreeCAD  # type: ignore
            import Part  # type: ignore
            from FreeCAD import Vector  # type: ignore
        except ImportError:  # pragma: no cover - headless/dev mode
            return shape

        normal_vec = Vector(normal)
        if normal_vec.Length == 0:
            return shape
        normal_vec.normalize()
        width = max(float(self._fc_len(self.params.stock_width_m)) * 6.0, float(span))
        height = max(float(self._fc_len(self.params.stock_height_m)) * 6.0, float(span))
        plane = Part.makePlane(width, height)
        rotation = FreeCAD.Rotation(Vector(0, 0, 1), normal_vec)
        point_vec = point if isinstance(point, Vector) else Vector(point)
        # Center the finite plane on the cut point; Part.makePlane creates a face from (0,0)
        # to (width,height), so anchoring it at the cut point can miss the solid after rotation.
        center_offset = FreeCAD.Placement(Vector(-width / 2.0, -height / 2.0, 0), FreeCAD.Rotation())
        plane.Placement = FreeCAD.Placement(point_vec, rotation).multiply(center_offset)
        keep_vec = keep_point if isinstance(keep_point, Vector) else Vector(keep_point)
        direction = normal_vec if (keep_vec - point_vec).dot(normal_vec) >= 0 else -normal_vec
        depth = float(span) * 6.0
        prism = plane.extrude(direction * depth)
        try:
            result = shape.common(prism)
        except Exception:  # pragma: no cover - boolean failure
            return None
        if self._is_shape_valid(result):
            return result

        # Retry opposite half-space; helps if keep_point ended up on the wrong side.
        prism2 = plane.extrude((-direction) * depth)
        try:
            result2 = shape.common(prism2)
        except Exception:  # pragma: no cover
            return None
        return result2 if self._is_shape_valid(result2) else None

    def _is_shape_valid(self, shape) -> bool:
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
        if getattr(shape, "ShapeType", "").lower() == "solid" and hasattr(shape, "Volume"):
            try:
                if shape.Volume <= 1e-9:
                    return False
            except Exception:
                pass
        return True

    def _group_by_length(self, struts: List[Strut]) -> List[Tuple[str, List[Strut]]]:
        tolerance = max(self.params.clearance_m, 1e-4)
        buckets: Dict[int, List[Strut]] = {}
        for strut in struts:
            key = round(strut.length / tolerance)
            buckets.setdefault(key, []).append(strut)

        grouped: List[Tuple[str, List[Strut]]] = []
        for idx, key in enumerate(sorted(buckets), start=1):
            grouped.append((f"L{idx:02d}", buckets[key]))
        return grouped

    def _guid_for(self, name: str, strut: Strut) -> str:
        payload = f"{name}:{strut.length:.6f}:{strut.start}:{strut.end}"
        return uuid5(NAMESPACE_URL, payload).hex.upper()

    def _annotate_kerf(self, structure: Any) -> None:
        if hasattr(structure, "Kerf"):
            structure.Kerf = self.params.kerf_m
        elif hasattr(structure, "addProperty"):
            try:
                structure.addProperty("App::PropertyFloat", "Kerf", "Strut", "Saw kerf")
                structure.Kerf = self.params.kerf_m
            except Exception:  # pragma: no cover - FreeCAD-specific failure
                pass

    def _store_direction(self, structure: Any, strut: Strut) -> None:
        if not hasattr(structure, "addProperty"):
            return
        try:
            import FreeCAD  # type: ignore

            structure.addProperty(
                "App::PropertyVector", "StrutDirection", "Strut", "Unit direction vector"
            )
            dx = float(strut.end[0] - strut.start[0])
            dy = float(strut.end[1] - strut.start[1])
            dz = float(strut.end[2] - strut.start[2])
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            if length < 1e-9:
                return
            structure.StrutDirection = FreeCAD.Vector(  # type: ignore[name-defined]
                dx / length, dy / length, dz / length
            )
        except Exception:  # pragma: no cover - FreeCAD-specific failure
            pass
