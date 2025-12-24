"""Utilities for constructing base icosahedron meshes.

The base icosahedron orientation matters for reproducible strut/panel labeling.
We orient it so that a vertex lies on the +Z axis ("top view" looks at a tip).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, acos, cos, sin, sqrt
from typing import Dict, List, Sequence, Set, Tuple

Vector3 = Tuple[float, float, float]
Face = Tuple[int, int, int]


@dataclass(slots=True)
class IcosahedronMesh:
    nodes: List[Vector3]
    faces: List[Face]

    def scaled(self, scale: float) -> "IcosahedronMesh":
        return IcosahedronMesh(
            nodes=[(x * scale, y * scale, z * scale) for x, y, z in self.nodes],
            faces=list(self.faces),
        )


def regular_icosahedron() -> IcosahedronMesh:
    """Return a normalized icosahedron centered at the origin."""

    phi = (1 + sqrt(5)) / 2
    raw_nodes: List[Vector3] = [
        (-1.0, phi, 0.0),
        (1.0, phi, 0.0),
        (-1.0, -phi, 0.0),
        (1.0, -phi, 0.0),
        (0.0, -1.0, phi),
        (0.0, 1.0, phi),
        (0.0, -1.0, -phi),
        (0.0, 1.0, -phi),
        (phi, 0.0, -1.0),
        (phi, 0.0, 1.0),
        (-phi, 0.0, -1.0),
        (-phi, 0.0, 1.0),
    ]

    nodes = [_normalize_to_radius(vec, 1.0) for vec in raw_nodes]

    # Rotate so one vertex points along +Z.
    # Pick a stable vertex from the canonical raw node set (index 5: (0, 1, phi)).
    nodes = _rotate_mesh_to_vertex_up(nodes, vertex_index=5)

    faces: List[Face] = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    return IcosahedronMesh(nodes, faces)


def _rotate_mesh_to_vertex_up(nodes: List[Vector3], vertex_index: int) -> List[Vector3]:
    if not nodes:
        return nodes
    idx = max(0, min(int(vertex_index), len(nodes) - 1))
    v = _normalize_vector(nodes[idx])
    zhat: Vector3 = (0.0, 0.0, 1.0)

    dot = _dot(v, zhat)
    if dot >= 1.0 - 1e-12:
        return nodes
    if dot <= -1.0 + 1e-12:
        # 180° flip around X axis (any axis in XY plane works; choose X for determinism).
        return [(_x, -_y, -_z) for (_x, _y, _z) in nodes]

    axis = _cross(v, zhat)
    axis_len = _length(axis)
    if axis_len <= 1e-12:
        return nodes
    axis_n = (axis[0] / axis_len, axis[1] / axis_len, axis[2] / axis_len)
    angle = acos(max(-1.0, min(1.0, dot)))
    return [_rotate_axis_angle(p, axis_n, angle) for p in nodes]


def _rotate_axis_angle(vec: Vector3, axis_unit: Vector3, angle_rad: float) -> Vector3:
    # Rodrigues' rotation formula.
    x, y, z = vec
    ax, ay, az = axis_unit
    c = cos(angle_rad)
    s = sin(angle_rad)

    # v*cos(theta)
    vx = x * c
    vy = y * c
    vz = z * c

    # (axis x v)*sin(theta)
    cx, cy, cz = _cross((ax, ay, az), (x, y, z))
    wx = cx * s
    wy = cy * s
    wz = cz * s

    # axis*(axis·v)*(1-cos(theta))
    k = _dot((ax, ay, az), (x, y, z)) * (1.0 - c)
    ux = ax * k
    uy = ay * k
    uz = az * k

    return (vx + wx + ux, vy + wy + uy, vz + wz + uz)


def build_icosahedron(radius: float) -> IcosahedronMesh:
    """Construct an icosahedron scaled to the given circumscribed radius."""

    mesh = regular_icosahedron()
    return mesh.scaled(radius)


def truncate_mesh(mesh: IcosahedronMesh, ratio: float) -> Tuple[IcosahedronMesh, List[List[int]]]:
    """Return a truncated variant of the input mesh plus polygon loops.

    ``ratio`` represents the fractional distance from each vertex along its
    incident edges at which the truncation plane intersects. Values near zero
    keep the original mesh, while values approaching ``0.5`` carve deeper into
    each vertex. Resulting pentagons/hexagons are triangulated so downstream
    code can continue to assume triangular faces.
    """

    if ratio <= 0:
        return mesh, [list(face) for face in mesh.faces]
    if ratio >= 0.5:
        raise ValueError("Truncation ratio must be < 0.5 for triangular faces")

    radius = max(_length(node) for node in mesh.nodes)
    dir_vertex: Dict[Tuple[int, int], int] = {}
    new_nodes: List[Vector3] = []
    faces: List[Face] = []
    polygons: List[List[int]] = []

    def vertex_on_edge(origin: int, target: int) -> int:
        key = (origin, target)
        idx = dir_vertex.get(key)
        if idx is not None:
            return idx
        pa = mesh.nodes[origin]
        pb = mesh.nodes[target]
        px = pa[0] + (pb[0] - pa[0]) * ratio
        py = pa[1] + (pb[1] - pa[1]) * ratio
        pz = pa[2] + (pb[2] - pa[2]) * ratio
        normalized = _normalize_to_radius((px, py, pz), radius)
        idx = len(new_nodes)
        dir_vertex[key] = idx
        new_nodes.append(normalized)
        return idx

    # Triangulate truncated versions of original triangular faces (hexagons).
    for a, b, c in mesh.faces:
        hex_loop = [
            vertex_on_edge(a, b),
            vertex_on_edge(b, a),
            vertex_on_edge(b, c),
            vertex_on_edge(c, b),
            vertex_on_edge(c, a),
            vertex_on_edge(a, c),
        ]
        polygons.append(hex_loop)
        faces.extend(_fan_triangulate(hex_loop))

    # Add faces created at each truncated vertex (pentagons for an icosahedron).
    neighbor_map = _build_neighbor_map(len(mesh.nodes), mesh.faces)
    for vertex_idx, neighbors in enumerate(neighbor_map):
        if not neighbors:
            continue
        ordered = _order_neighbors(vertex_idx, neighbors, mesh.nodes)
        polygon = [vertex_on_edge(vertex_idx, nb) for nb in ordered]
        polygons.append(polygon)
        faces.extend(_fan_triangulate(polygon))

    return IcosahedronMesh(nodes=new_nodes, faces=faces), polygons


def _fan_triangulate(loop: Sequence[int]) -> List[Face]:
    if len(loop) < 3:
        return []
    root = loop[0]
    return [(root, loop[i], loop[i + 1]) for i in range(1, len(loop) - 1)]


def _build_neighbor_map(count: int, faces: Sequence[Face]) -> List[Set[int]]:
    neighbors: List[Set[int]] = [set() for _ in range(count)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def _order_neighbors(vertex: int, neighbor_set: Set[int], nodes: Sequence[Vector3]) -> List[int]:
    origin = nodes[vertex]
    normal = _normalize_vector(origin)
    ref = (0.0, 0.0, 1.0)
    if abs(_dot(normal, ref)) > 0.9:
        ref = (0.0, 1.0, 0.0)
    tangent = _normalize_vector(_cross(ref, normal))
    bitangent = _cross(normal, tangent)

    ordered = []
    for nb in neighbor_set:
        vec = _subtract(nodes[nb], origin)
        proj_x = _dot(vec, tangent)
        proj_y = _dot(vec, bitangent)
        angle = atan2(proj_y, proj_x)
        ordered.append((angle, nb))
    ordered.sort()
    return [idx for _, idx in ordered]


def _normalize_to_radius(vec: Vector3, radius: float) -> Vector3:
    length = _length(vec)
    scale = radius / length if length else 0
    return (vec[0] * scale, vec[1] * scale, vec[2] * scale)


def _length(vec: Vector3) -> float:
    return sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def _subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize_vector(vec: Vector3) -> Vector3:
    length = _length(vec)
    if length == 0:
        raise ValueError("Cannot normalize zero-length vector")
    return (vec[0] / length, vec[1] / length, vec[2] / length)
