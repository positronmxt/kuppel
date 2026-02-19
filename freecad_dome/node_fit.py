"""Shared node-fit data used by both strut and connector builders.

The :class:`NodeFitData` is computed once by the pipeline (during the
strut-generation step) and shared with the node-connector step so both
modules work with identical angular ordering, incident maps, and separation
bisectors.  This eliminates the duplication that previously existed between
``struts.py`` and ``node_connectors.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from . import vec3 as v3
from .tessellation import Strut, TessellatedDome, Vector3

__all__ = [
    "IncidentEntry",
    "NodeFitInfo",
    "NodeFitData",
    "compute_node_fit_data",
]

IncidentEntry = Tuple[Strut, str, Vector3]  # (strut, end_label, direction_from_node)


@dataclass(slots=True)
class NodeFitInfo:
    """Per-node fit metadata shared between struts and connectors."""

    node_index: int
    position: Vector3
    radial: Vector3                        # outward normal (radial or vertical for belt)
    is_belt_node: bool
    incident: List[IncidentEntry]
    angular_order: List[int]               # strut indices in clockwise angular order
    tangent_directions: Dict[int, Vector3] # strut_index → tangent-projected direction


@dataclass
class NodeFitData:
    """Collection of per-node fit info for the entire dome."""

    nodes: Dict[int, NodeFitInfo] = field(default_factory=dict)

    def get(self, node_index: int) -> NodeFitInfo | None:
        return self.nodes.get(node_index)


def compute_node_fit_data(
    dome: TessellatedDome,
    hemisphere_ratio: float,
    radius_m: float,
) -> NodeFitData:
    """Compute shared node-fit metadata for every node in the dome.

    This is the single source of truth for incident maps, angular ordering,
    and radial normals — consumed by both :class:`StrutBuilder` and
    :class:`NodeConnectorBuilder`.
    """
    _norm = v3.norm
    _normalize = v3.normalize
    _dot = v3.dot
    _cross = v3.cross
    _sub = v3.sub
    _scale = v3.scale

    # Build incident map.
    incident: Dict[int, List[IncidentEntry]] = {}
    for s in dome.struts:
        sx, sy, sz = s.start
        ex, ey, ez = s.end
        incident.setdefault(s.start_index, []).append(
            (s, "start", (ex - sx, ey - sy, ez - sz))
        )
        incident.setdefault(s.end_index, []).append(
            (s, "end", (sx - ex, sy - ey, sz - ez))
        )

    # Belt detection.
    belt_height: float | None = None
    belt_eps = max(1e-6, radius_m * 1e-5)
    if hemisphere_ratio < 1.0:
        belt_height = radius_m * (1.0 - 2.0 * hemisphere_ratio)

    result = NodeFitData()

    for node_idx in range(len(dome.nodes)):
        items = incident.get(node_idx)
        if not items:
            continue

        p = dome.nodes[node_idx]
        is_belt = False
        if belt_height is not None:
            is_belt = abs(p[2] - belt_height) <= belt_eps

        radial = _normalize(p)
        if _norm(radial) <= 1e-12:
            continue
        if is_belt:
            radial = (0.0, 0.0, 1.0)

        # Tangent-plane basis.
        ref = (1.0, 0.0, 0.0) if abs(radial[0]) < 0.9 else (0.0, 1.0, 0.0)
        u = _normalize(_cross(radial, ref))
        v = _normalize(_cross(radial, u))

        # Angular ordering.
        enriched: List[Tuple[float, Strut, str, Vector3, Vector3]] = []
        for s, end_label, d in items:
            a = _normalize(d)
            # Project to tangent plane.
            t = _sub(a, _scale(radial, _dot(a, radial)))
            t = _normalize(t)
            if _norm(t) <= 1e-12:
                if is_belt:
                    t = _normalize((a[0], a[1], 0.0))
                    if _norm(t) <= 1e-12:
                        t = u
                else:
                    t = u
            ang = math.atan2(_dot(t, v), _dot(t, u))
            enriched.append((ang, s, end_label, a, t))

        if not enriched:
            continue
        enriched.sort(key=lambda row: row[0])

        angular_order: List[int] = [s.index for _, s, _, _, _ in enriched]
        tangent_dirs: Dict[int, Vector3] = {
            s.index: t for _, s, _, _, t in enriched
        }

        result.nodes[node_idx] = NodeFitInfo(
            node_index=node_idx,
            position=p,
            radial=radial,
            is_belt_node=is_belt,
            incident=items,
            angular_order=angular_order,
            tangent_directions=tangent_dirs,
        )

    return result
