"""Dome tessellation and strut metadata generation."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Tuple, Set

from . import icosahedron
from .parameters import DomeParameters

__all__ = [
    "Vector3",
    "Edge",
    "Face",
    "Strut",
    "Panel",
    "TessellatedDome",
    "tessellate",
    "validate_structure",
]

Vector3 = Tuple[float, float, float]
Edge = Tuple[int, int]
Face = Tuple[int, int, int]


@dataclass(slots=True)
class Strut:
    index: int
    start_index: int
    end_index: int
    start: Vector3
    end: Vector3
    panel_indices: Tuple[int, ...]
    primary_normal: Vector3
    secondary_normal: Vector3 | None
    length: float

    @property
    def midpoint(self) -> Vector3:
        x1, y1, z1 = self.start
        x2, y2, z2 = self.end
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5, (z1 + z2) * 0.5)

    @property
    def direction(self) -> Vector3:
        x1, y1, z1 = self.start
        x2, y2, z2 = self.end
        return (x2 - x1, y2 - y1, z2 - z1)


def _distance(a: Vector3, b: Vector3) -> float:
    x1, y1, z1 = a
    x2, y2, z2 = b
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


@dataclass(slots=True)
class Panel:
    index: int
    node_indices: Tuple[int, ...]
    normal: Vector3

    def edges(self) -> List[Edge]:
        edges: List[Edge] = []
        count = len(self.node_indices)
        for i in range(count):
            a = self.node_indices[i]
            b = self.node_indices[(i + 1) % count]
            edges.append(_ordered_edge(a, b))
        return edges


@dataclass(slots=True)
class TessellatedDome:
    nodes: List[Vector3]
    struts: List[Strut]
    panels: List[Panel]

    def summary(self) -> str:
        return f"{len(self.nodes)} nodes / {len(self.struts)} struts"


def tessellate(mesh: icosahedron.IcosahedronMesh, params: DomeParameters) -> TessellatedDome:
    """Return a tessellated dome honoring frequency, truncation, and hemisphere settings."""

    frequency = max(1, params.frequency)
    nodes, _, faces = _subdivide(mesh, frequency, params.radius_m)

    if params.use_truncation and params.truncation_ratio > 0:
        base_mesh = icosahedron.IcosahedronMesh(nodes=nodes, faces=faces)
        truncated_mesh, polygons = icosahedron.truncate_mesh(base_mesh, params.truncation_ratio)
        nodes = truncated_mesh.nodes
    else:
        polygons = [list(face) for face in faces]

    nodes, polygons = _clip_polygons(nodes, polygons, params.radius_m, params.hemisphere_ratio)
    if not nodes or not polygons:
        return TessellatedDome(nodes=[], struts=[], panels=[])

    # Optional legacy behavior: close the bottom with a planar cap at the belt plane.
    if params.hemisphere_ratio < 1.0 and params.generate_belt_cap:
        belt_height = params.radius_m * (1 - 2 * params.hemisphere_ratio)
        nodes, polygons = _add_planar_belt(nodes, polygons, belt_height)

    # Panels with more than 3 vertices are generally not planar on the sphere. If we create each
    # panel by projecting its own vertices to its own best-fit plane, shared edges/vertices no
    # longer coincide between neighboring panels (visible as "some corners too low/high").
    #
    # Planarize the node positions so each polygon becomes planar while keeping vertices shared.
    # This yields consistent edge heights and parallel shared edges between adjacent panels.
    if any(len(poly) > 3 for poly in polygons):
        belt_height = None
        belt_nodes: Set[int] = set()
        if params.hemisphere_ratio < 1.0:
            belt_height = params.radius_m * (1 - 2 * params.hemisphere_ratio)
            belt_nodes = {
                i for i, p in enumerate(nodes) if abs(p[2] - belt_height) <= 1e-6
            }
        nodes = _planarize_nodes(nodes, polygons, belt_height=belt_height, belt_nodes=belt_nodes)

    edge_to_panels = _edges_from_polygons(polygons)
    panels: List[Panel] = []
    panel_normals: List[Vector3] = []
    for idx, node_indices in enumerate(polygons):
        points = [nodes[i] for i in node_indices]
        normal = _polygon_normal(points)
        centroid = _polygon_centroid(points)
        if _vector_dot(normal, centroid) > 0:
            normal = (-normal[0], -normal[1], -normal[2])
        panels.append(Panel(index=idx, node_indices=tuple(node_indices), normal=normal))
        panel_normals.append(normal)

    struts: List[Strut] = []
    for idx, (edge, panel_ids) in enumerate(sorted(edge_to_panels.items())):
        u, v = edge
        midpoint = (
            (nodes[u][0] + nodes[v][0]) * 0.5,
            (nodes[u][1] + nodes[v][1]) * 0.5,
            (nodes[u][2] + nodes[v][2]) * 0.5,
        )
        primary_normal, secondary_normal = _select_strut_normals(
            panel_ids, panel_normals, midpoint
        )
        struts.append(
            Strut(
                index=idx,
                start_index=u,
                end_index=v,
                start=nodes[u],
                end=nodes[v],
                panel_indices=tuple(panel_ids),
                primary_normal=primary_normal,
                secondary_normal=secondary_normal,
                length=_distance(nodes[u], nodes[v]),
            )
        )

    return TessellatedDome(nodes=nodes, struts=struts, panels=panels)


def _planarize_nodes(
    nodes: List[Vector3],
    polygons: List[List[int]],
    *,
    belt_height: float | None = None,
    belt_nodes: Set[int] | None = None,
    iterations: int = 35,
    relax: float = 0.65,
) -> List[Vector3]:
    """Iteratively planarize polygon faces while keeping shared vertices consistent.

    Each iteration:
    - computes a plane for each polygon from current node positions
    - projects each node to the incident polygon planes (averaged)
    - relaxes toward that average projection

    When belt_height is provided, nodes in belt_nodes are constrained to that Z level.
    """
    if not nodes or not polygons:
        return nodes

    if belt_nodes is None:
        belt_nodes = set()

    # Map node -> incident polygon ids.
    incident: List[List[int]] = [[] for _ in range(len(nodes))]
    for pid, poly in enumerate(polygons):
        for ni in poly:
            if 0 <= ni < len(incident):
                incident[ni].append(pid)

    cur = [(float(x), float(y), float(z)) for (x, y, z) in nodes]

    def _proj_point_to_plane(p: Vector3, c: Vector3, n: Vector3) -> Vector3:
        # p_proj = p - n * dot(n, p-c)
        d = _vector_dot(n, (p[0] - c[0], p[1] - c[1], p[2] - c[2]))
        return (p[0] - n[0] * d, p[1] - n[1] * d, p[2] - n[2] * d)

    def _solve_3x3(M: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]], b: Vector3) -> Vector3 | None:
        # Gaussian elimination with partial pivoting.
        a = [
            [float(M[0][0]), float(M[0][1]), float(M[0][2]), float(b[0])],
            [float(M[1][0]), float(M[1][1]), float(M[1][2]), float(b[1])],
            [float(M[2][0]), float(M[2][1]), float(M[2][2]), float(b[2])],
        ]
        for col in range(3):
            pivot = col
            piv_val = abs(a[pivot][col])
            for r in range(col + 1, 3):
                v = abs(a[r][col])
                if v > piv_val:
                    pivot = r
                    piv_val = v
            if piv_val <= 1e-12:
                return None
            if pivot != col:
                a[col], a[pivot] = a[pivot], a[col]
            inv = 1.0 / a[col][col]
            for c in range(col, 4):
                a[col][c] *= inv
            for r in range(3):
                if r == col:
                    continue
                factor = a[r][col]
                if abs(factor) <= 1e-18:
                    continue
                for c in range(col, 4):
                    a[r][c] -= factor * a[col][c]
        return (a[0][3], a[1][3], a[2][3])

    def _best_point_for_planes(plane_ids: List[int], planes: List[Tuple[Vector3, Vector3]], fallback: Vector3) -> Vector3:
        # Minimize sum_i (n_i·x - d_i)^2, where d_i = n_i·c_i.
        # Normal equations: (sum n n^T) x = sum n d.
        m00 = m01 = m02 = 0.0
        m10 = m11 = m12 = 0.0
        m20 = m21 = m22 = 0.0
        b0 = b1 = b2 = 0.0
        for pid in plane_ids:
            c, n = planes[pid]
            d = _vector_dot(n, c)
            nx, ny, nz = n
            m00 += nx * nx
            m01 += nx * ny
            m02 += nx * nz
            m10 += ny * nx
            m11 += ny * ny
            m12 += ny * nz
            m20 += nz * nx
            m21 += nz * ny
            m22 += nz * nz
            b0 += nx * d
            b1 += ny * d
            b2 += nz * d
        sol = _solve_3x3(((m00, m01, m02), (m10, m11, m12), (m20, m21, m22)), (b0, b1, b2))
        if sol is not None:
            return sol
        # Fallback: average projections.
        projs: List[Vector3] = []
        for pid in plane_ids:
            c, n = planes[pid]
            projs.append(_proj_point_to_plane(fallback, c, n))
        return _avg(projs) if projs else fallback

    def _avg(points: List[Vector3]) -> Vector3:
        inv = 1.0 / float(len(points))
        sx = sum(p[0] for p in points)
        sy = sum(p[1] for p in points)
        sz = sum(p[2] for p in points)
        return (sx * inv, sy * inv, sz * inv)

    relax = max(0.0, min(1.0, float(relax)))
    for _ in range(max(1, int(iterations))):
        # Compute current planes for each polygon.
        planes: List[Tuple[Vector3, Vector3]] = []
        for poly in polygons:
            pts = [cur[i] for i in poly]
            c = _avg(pts)
            n = _polygon_normal(pts)
            ln = sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
            if ln <= 1e-12:
                # Degenerate: fall back to a stable inward direction.
                n = (0.0, 0.0, -1.0)
            else:
                n = (n[0] / ln, n[1] / ln, n[2] / ln)
            # Keep normals oriented inward (toward origin).
            if _vector_dot(n, c) > 0:
                n = (-n[0], -n[1], -n[2])
            planes.append((c, n))

        # Update each node toward the average of its plane projections.
        nxt: List[Vector3] = []
        for ni, p in enumerate(cur):
            inc = incident[ni]
            if not inc:
                nxt.append(p)
                continue
            t = _best_point_for_planes(inc, planes, p)
            q = (
                p[0] * (1.0 - relax) + t[0] * relax,
                p[1] * (1.0 - relax) + t[1] * relax,
                p[2] * (1.0 - relax) + t[2] * relax,
            )
            if belt_height is not None and ni in belt_nodes:
                q = (q[0], q[1], float(belt_height))
            nxt.append(q)
        cur = nxt

    return cur


def validate_structure(
    dome: TessellatedDome, params: DomeParameters, tolerance: float | None = None
) -> Dict[str, object]:
    tol = tolerance or max(params.clearance_m * 0.5, 1e-4)
    node_valence = [0] * len(dome.nodes)
    stray_endpoints: List[Dict[str, object]] = []
    strut_edges = set()
    for strut in dome.struts:
        node_valence[strut.start_index] += 1
        node_valence[strut.end_index] += 1
        strut_edges.add(_ordered_edge(strut.start_index, strut.end_index))
        for endpoint, idx in ((strut.start, strut.start_index), (strut.end, strut.end_index)):
            node = dome.nodes[idx]
            deviation = _distance(endpoint, node)
            if deviation > tol:
                stray_endpoints.append(
                    {
                        "strut": strut.index,
                        "node": idx,
                        "deviation": deviation,
                    }
                )

    valence_counter = Counter(node_valence)
    low_valence = [(idx, val) for idx, val in enumerate(node_valence) if val < 3]
    radius_errors = [abs(_distance(node, (0.0, 0.0, 0.0)) - params.radius_m) for node in dome.nodes]
    max_radius_error = max(radius_errors) if radius_errors else 0.0
    length_histogram = _length_histogram(dome.struts)

    missing_panel_edges: List[Dict[str, object]] = []
    panel_areas: List[float] = []
    for panel in dome.panels:
        area = _polygon_area([dome.nodes[idx] for idx in panel.node_indices])
        panel_areas.append(area)
        for edge in panel.edges():
            if edge not in strut_edges:
                missing_panel_edges.append({"panel": panel.index, "edge": edge})

    panel_area_stats = {}
    if panel_areas:
        panel_area_stats = {
            "min": min(panel_areas),
            "max": max(panel_areas),
            "avg": sum(panel_areas) / len(panel_areas),
        }

    if stray_endpoints:
        logging.error("Found %d strut endpoints off their nodes", len(stray_endpoints))
    if low_valence:
        logging.warning("%d nodes have valence < 3", len(low_valence))
    logging.info("Node valence distribution: %s", sorted(valence_counter.items()))
    logging.info("Strut length families: %s", length_histogram)
    if missing_panel_edges:
        logging.error(
            "%d panel edges lack struts (showing first 5): %s",
            len(missing_panel_edges),
            missing_panel_edges[:5],
        )
    if panel_area_stats:
        logging.info("Panel area stats (min/max/avg): %s", panel_area_stats)

    return {
        "max_radius_error": max_radius_error,
        "valence_distribution": dict(valence_counter),
        "low_valence_nodes": low_valence,
        "stray_endpoints": stray_endpoints,
        "length_histogram": length_histogram,
        "panel_area_stats": panel_area_stats,
        "missing_panel_edges": missing_panel_edges,
    }


def _subdivide(
    mesh: icosahedron.IcosahedronMesh, frequency: int, radius: float
) -> Tuple[List[Vector3], List[Edge], List[Face]]:
    """Subdivide all faces of the mesh using barycentric interpolation."""

    node_cache: Dict[Tuple[float, float, float], int] = {}
    nodes: List[Vector3] = []
    faces: List[Tuple[int, int, int]] = []

    def add_vertex(vec: Vector3) -> int:
        normalized = _normalize(vec, radius)
        key = _quantize(normalized)
        idx = node_cache.get(key)
        if idx is None:
            idx = len(nodes)
            node_cache[key] = idx
            nodes.append(normalized)
        return idx

    for ia, ib, ic in mesh.faces:
        va, vb, vc = mesh.nodes[ia], mesh.nodes[ib], mesh.nodes[ic]
        local_indices: Dict[Tuple[int, int], int] = {}

        for i in range(frequency + 1):
            for j in range(frequency + 1 - i):
                k = frequency - i - j
                coeff = 1.0 / frequency
                px = (va[0] * i + vb[0] * j + vc[0] * k) * coeff
                py = (va[1] * i + vb[1] * j + vc[1] * k) * coeff
                pz = (va[2] * i + vb[2] * j + vc[2] * k) * coeff
                idx = add_vertex((px, py, pz))
                local_indices[(i, j)] = idx

        for i in range(frequency):
            for j in range(frequency - i):
                v0 = local_indices[(i, j)]
                v1 = local_indices[(i + 1, j)]
                v2 = local_indices[(i, j + 1)]
                faces.append((v0, v1, v2))
                if i + j < frequency - 1:
                    v3 = local_indices[(i + 1, j + 1)]
                    faces.append((v1, v3, v2))

    edges = _faces_to_edges(faces)
    return nodes, edges, faces


def _faces_to_edges(faces: Iterable[Tuple[int, int, int]]) -> List[Edge]:
    edge_set = set()
    for a, b, c in faces:
        edge_set.add(_ordered_edge(a, b))
        edge_set.add(_ordered_edge(b, c))
        edge_set.add(_ordered_edge(c, a))
    return sorted(edge_set)


def _ordered_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _clip_polygons(
    nodes: List[Vector3],
    polygons: List[List[int]],
    radius: float,
    hemisphere_ratio: float,
) -> Tuple[List[Vector3], List[List[int]]]:
    if hemisphere_ratio >= 1.0:
        return nodes, polygons

    z_min = radius * (1 - 2 * hemisphere_ratio)
    working_nodes = list(nodes)
    kept_polygons: List[List[int]] = []
    used_nodes: set[int] = set()
    split_cache: Dict[Edge, int] = {}

    for poly in polygons:
        clipped = _clip_polygon_above_plane(poly, working_nodes, z_min, split_cache)
        if len(clipped) >= 3:
            kept_polygons.append(clipped)
            used_nodes.update(clipped)

    removed_polygons = len(polygons) - len(kept_polygons)
    if removed_polygons:
        logging.info(
            "Hemisphere clip removed %d polygons entirely (ratio=%.3f)",
            removed_polygons,
            hemisphere_ratio,
        )

    if not kept_polygons:
        return [], []

    index_map: Dict[int, int] = {}
    filtered_nodes: List[Vector3] = []
    for old_idx, point in enumerate(working_nodes):
        if old_idx in used_nodes:
            index_map[old_idx] = len(filtered_nodes)
            filtered_nodes.append(point)

    removed_nodes = len(working_nodes) - len(filtered_nodes)
    if removed_nodes:
        logging.info(
            "Hemisphere clip removed %d nodes (ratio=%.3f)", removed_nodes, hemisphere_ratio
        )

    remapped_polygons = [[index_map[idx] for idx in poly] for poly in kept_polygons]

    # Keep the dome bottom open by default (base is usually a slab). We still snap any nodes
    # on/near the belt plane to the exact height for numerical stability.
    snapped_nodes = []
    tol = 1e-6
    for x, y, z in filtered_nodes:
        if abs(z - z_min) <= tol:
            snapped_nodes.append((x, y, z_min))
        else:
            snapped_nodes.append((x, y, z))
    return snapped_nodes, remapped_polygons


def _clip_polygon_above_plane(
    polygon: List[int],
    nodes: List[Vector3],
    z_min: float,
    cache: Dict[Edge, int],
) -> List[int]:
    if not polygon:
        return []

    result: List[int] = []
    prev_idx = polygon[-1]
    prev_point = nodes[prev_idx]
    prev_inside = prev_point[2] >= z_min
    for curr_idx in polygon:
        curr_point = nodes[curr_idx]
        curr_inside = curr_point[2] >= z_min
        if curr_inside:
            if not prev_inside:
                result.append(
                    _edge_plane_intersection(prev_idx, curr_idx, nodes, z_min, cache)
                )
            result.append(curr_idx)
        elif prev_inside:
            result.append(
                _edge_plane_intersection(prev_idx, curr_idx, nodes, z_min, cache)
            )
        prev_idx = curr_idx
        prev_inside = curr_inside

    if len(result) >= 3 and result[0] == result[-1]:
        result.pop()
    return result


def _edge_plane_intersection(
    a_idx: int,
    b_idx: int,
    nodes: List[Vector3],
    z_min: float,
    cache: Dict[Edge, int],
) -> int:
    edge = _ordered_edge(a_idx, b_idx)
    cached = cache.get(edge)
    if cached is not None:
        return cached

    ax, ay, az = nodes[a_idx]
    bx, by, bz = nodes[b_idx]
    denom = bz - az
    if denom == 0:
        t = 0.5
    else:
        t = (z_min - az) / denom
    t = max(0.0, min(1.0, t))
    x = ax + (bx - ax) * t
    y = ay + (by - ay) * t
    idx = len(nodes)
    nodes.append((x, y, z_min))
    cache[edge] = idx
    return idx


def _add_planar_belt(
    nodes: List[Vector3], polygons: List[List[int]], belt_height: float
) -> Tuple[List[Vector3], List[List[int]]]:
    boundary_loops = _extract_boundary_loops(polygons)
    if not boundary_loops:
        return nodes, polygons

    new_nodes = list(nodes)
    new_polygons = list(polygons)
    base_map: Dict[int, int] = {}
    tol = 1e-6

    for loop in boundary_loops:
        oriented = _orient_loop_ccw(loop, new_nodes)
        if len(oriented) < 3:
            continue
        base_indices: List[int] = []
        for idx in oriented:
            base_idx = base_map.get(idx)
            if base_idx is None:
                x, y, z = new_nodes[idx]
                if abs(z - belt_height) <= tol:
                    base_idx = idx
                else:
                    base_idx = len(new_nodes)
                    new_nodes.append((x, y, belt_height))
                base_map[idx] = base_idx
            base_indices.append(base_idx)

        loop_len = len(oriented)
        for pos in range(loop_len):
            a = oriented[pos]
            b = oriented[(pos + 1) % loop_len]
            base_a = base_map[a]
            base_b = base_map[b]
            tri1 = [a, b, base_b]
            tri2 = [a, base_b, base_a]
            if len({tri1[0], tri1[1], tri1[2]}) == 3:
                new_polygons.append(tri1)
            if len({tri2[0], tri2[1], tri2[2]}) == 3:
                new_polygons.append(tri2)

        simplified_base = _simplify_loop(base_indices)
        if len(simplified_base) >= 3:
            new_polygons.append(simplified_base)

    logging.info(
        "Added planar belt loops=%d extra_nodes=%d",
        len(boundary_loops),
        len(new_nodes) - len(nodes),
    )
    return new_nodes, new_polygons


def _extract_boundary_loops(polygons: List[List[int]]) -> List[List[int]]:
    if not polygons:
        return []
    edge_counts: Dict[Edge, int] = {}
    directed_edges: List[Tuple[int, int]] = []
    for polygon in polygons:
        count = len(polygon)
        for i in range(count):
            a = polygon[i]
            b = polygon[(i + 1) % count]
            directed_edges.append((a, b))
            edge = _ordered_edge(a, b)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_edges = {
        edge for edge, occurrences in edge_counts.items() if occurrences == 1
    }
    if not boundary_edges:
        return []

    adjacency: Dict[int, Set[int]] = {}
    for a, b in directed_edges:
        edge = _ordered_edge(a, b)
        if edge in boundary_edges:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

    loops: List[List[int]] = []
    unused_edges = set(boundary_edges)
    while unused_edges:
        start_edge = unused_edges.pop()
        start, nxt = start_edge
        loop = [start, nxt]
        prev = start
        current = nxt
        while True:
            neighbors = adjacency.get(current, set()) - {prev}
            if not neighbors:
                break
            neighbor = neighbors.pop()
            undirected = _ordered_edge(current, neighbor)
            if undirected in unused_edges:
                unused_edges.remove(undirected)
            loop.append(neighbor)
            if neighbor == start:
                loop.pop()
                loops.append(loop)
                break
            prev, current = current, neighbor

    return loops


def _orient_loop_ccw(loop: List[int], nodes: List[Vector3]) -> List[int]:
    if len(loop) < 3:
        return loop
    area = 0.0
    for idx, node_index in enumerate(loop):
        x1, y1, _ = nodes[node_index]
        x2, y2, _ = nodes[loop[(idx + 1) % len(loop)]]
        area += x1 * y2 - x2 * y1
    if area < 0:
        return list(reversed(loop))
    return loop


def _simplify_loop(loop: List[int]) -> List[int]:
    simplified: List[int] = []
    for idx in loop:
        if not simplified or simplified[-1] != idx:
            simplified.append(idx)
    if len(simplified) > 1 and simplified[0] == simplified[-1]:
        simplified.pop()
    return simplified


def _edges_from_polygons(polygons: List[List[int]]) -> Dict[Edge, List[int]]:
    edge_map: Dict[Edge, List[int]] = {}
    for panel_idx, polygon in enumerate(polygons):
        count = len(polygon)
        for i in range(count):
            edge = _ordered_edge(polygon[i], polygon[(i + 1) % count])
            panel_list = edge_map.setdefault(edge, [])
            panel_list.append(panel_idx)
    return edge_map


def _polygon_normal(points: List[Vector3]) -> Vector3:
    if len(points) < 3:
        return (0.0, 0.0, 1.0)
    nx = ny = nz = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])
    length = sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def _polygon_centroid(points: List[Vector3]) -> Vector3:
    if not points:
        return (0.0, 0.0, 0.0)
    count = len(points)
    sx = sum(p[0] for p in points) / count
    sy = sum(p[1] for p in points) / count
    sz = sum(p[2] for p in points) / count
    return (sx, sy, sz)


def _normalize(vec: Vector3, radius: float) -> Vector3:
    length = sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    if length == 0:
        raise ValueError("Cannot normalize zero-length vector")
    scale = radius / length
    return (vec[0] * scale, vec[1] * scale, vec[2] * scale)


def _quantize(vec: Vector3, digits: int = 9) -> Tuple[float, float, float]:
    return (round(vec[0], digits), round(vec[1], digits), round(vec[2], digits))


def _distance(a: Vector3, b: Vector3) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _vector_dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length_histogram(struts: List[Strut], precision: int = 4) -> Dict[float, int]:
    hist: Dict[float, int] = {}
    for strut in struts:
        key = round(strut.length, precision)
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items()))


def _triangle_area(a: Vector3, b: Vector3, c: Vector3) -> float:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)


def _polygon_area(points: List[Vector3]) -> float:
    if len(points) < 3:
        return 0.0
    origin = points[0]
    area = 0.0
    for idx in range(1, len(points) - 1):
        area += _triangle_area(origin, points[idx], points[idx + 1])
    return area


def _select_strut_normals(
    panel_ids: List[int], panel_normals: List[Vector3], midpoint: Vector3
) -> Tuple[Vector3, Vector3 | None]:
    if not panel_ids:
        return (0.0, 0.0, 1.0), None

    best_idx = panel_ids[0]
    best_dot = float("inf")
    for pid in panel_ids:
        normal = panel_normals[pid]
        dot = _vector_dot(normal, midpoint)
        if dot < best_dot:
            best_dot = dot
            best_idx = pid

    primary = panel_normals[best_idx]
    secondary = None
    for pid in panel_ids:
        if pid == best_idx:
            continue
        secondary = panel_normals[pid]
        break
    return primary, secondary
