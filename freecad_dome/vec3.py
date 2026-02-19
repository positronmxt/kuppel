"""Shared 3-component vector algebra helpers (pure Python, no FreeCAD dependency).

All functions operate on ``Vector3 = Tuple[float, float, float]`` values.
Used by ``struts.py``, ``node_connectors.py``, ``tessellation.py``, and others
to avoid duplicating the same math in every module.
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = [
    "Vector3",
    "norm",
    "normalize",
    "dot",
    "cross",
    "sub",
    "add",
    "scale",
    "angle_between",
    "lerp",
]

Vector3 = Tuple[float, float, float]

ZERO: Vector3 = (0.0, 0.0, 0.0)


def norm(v: Vector3) -> float:
    """Euclidean length of *v*."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalize(v: Vector3) -> Vector3:
    """Unit vector in the direction of *v*, or (0,0,0) if degenerate."""
    n = norm(v)
    if n <= 1e-12:
        return ZERO
    return (v[0] / n, v[1] / n, v[2] / n)


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(v: Vector3, s: float) -> Vector3:
    return (v[0] * s, v[1] * s, v[2] * s)


def angle_between(a: Vector3, b: Vector3) -> float:
    """Angle in radians between two vectors."""
    na, nb = norm(a), norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    c = dot(a, b) / (na * nb)
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


def lerp(a: Vector3, b: Vector3, t: float) -> Vector3:
    """Linear interpolation between *a* and *b* at parameter *t*."""
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )
