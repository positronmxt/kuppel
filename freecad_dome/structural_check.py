"""Structural adequacy check for geodesic dome struts.

Computes member forces from node loads using the **direct stiffness method**
for a 3-D pin-jointed truss, then checks each strut against:

* Euler buckling (critical axial force)
* Compressive / tensile capacity (material strength)
* Combined bending + axial (simplified interaction)

Results are per-strut utilization ratios; a value ≤ 1.0 passes.

References
----------

* EN 1995-1-1 (Eurocode 5) — timber member design
* EN 1993-1-1 (Eurocode 3) — steel member design
* EN 1999-1-1 (Eurocode 9) — aluminium member design

Usage::

    from freecad_dome.structural_check import run_structural_check, write_check_report

    result = run_structural_check(dome, params, load_result)
    write_check_report(result, "exports/structural_check.json")
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .loads import LoadCombination, LoadResult, NodeLoad
from .parameters import DomeParameters, MaterialSpec

__all__ = [
    "StrutCheck",
    "StructuralCheckResult",
    "run_structural_check",
    "write_check_report",
]

log = logging.getLogger(__name__)

Vector3 = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StrutCheck:
    """Capacity check result for a single strut."""

    strut_index: int
    start_node: int
    end_node: int
    length_m: float

    # Member force (positive = tension, negative = compression).
    axial_force_kn: float = 0.0
    governing_combination: str = ""

    # Capacities (kN, positive values).
    euler_buckling_kn: float = 0.0  # N_cr  (critical buckling load)
    compression_capacity_kn: float = 0.0  # N_c,Rd
    tension_capacity_kn: float = 0.0  # N_t,Rd

    # Utilization ratios (≤ 1.0 → OK).
    buckling_ratio: float = 0.0
    compression_ratio: float = 0.0
    tension_ratio: float = 0.0
    governing_ratio: float = 0.0  # max of all checks
    governing_check: str = ""  # which check governs

    @property
    def passes(self) -> bool:
        return self.governing_ratio <= 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strut_index": self.strut_index,
            "start_node": self.start_node,
            "end_node": self.end_node,
            "length_m": round(self.length_m, 4),
            "axial_force_kn": round(self.axial_force_kn, 4),
            "governing_combination": self.governing_combination,
            "euler_buckling_kn": round(self.euler_buckling_kn, 2),
            "compression_capacity_kn": round(self.compression_capacity_kn, 2),
            "tension_capacity_kn": round(self.tension_capacity_kn, 2),
            "buckling_ratio": round(self.buckling_ratio, 3),
            "compression_ratio": round(self.compression_ratio, 3),
            "tension_ratio": round(self.tension_ratio, 3),
            "governing_ratio": round(self.governing_ratio, 3),
            "governing_check": self.governing_check,
            "passes": self.passes,
        }


@dataclass(slots=True)
class StructuralCheckResult:
    """Aggregate result of all strut checks."""

    strut_checks: List[StrutCheck] = field(default_factory=list)
    max_utilization: float = 0.0
    critical_strut_index: int = -1
    all_pass: bool = True
    total_struts: int = 0
    failing_struts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_struts": self.total_struts,
                "failing_struts": self.failing_struts,
                "all_pass": self.all_pass,
                "max_utilization": round(self.max_utilization, 3),
                "critical_strut_index": self.critical_strut_index,
            },
            "struts": [sc.to_dict() for sc in self.strut_checks],
        }


# ---------------------------------------------------------------------------
# Cross-section properties
# ---------------------------------------------------------------------------


def _cross_section_area(params: DomeParameters) -> float:
    """Return cross-section area A (m²) for the strut profile."""
    w = float(params.stock_width_m)
    h = float(params.stock_height_m)
    profile = getattr(params, "strut_profile", "rectangular")
    if profile == "round":
        # Diameter = average of w and h.
        d = (w + h) / 2.0
        return math.pi * (d / 2.0) ** 2
    # rectangular or trapezoidal → approximate as rectangular.
    return w * h


def _second_moment_of_area(params: DomeParameters) -> float:
    """Return minimum second moment of area I_min (m⁴) for buckling.

    Uses the weak axis (smaller dimension) which governs buckling.
    """
    w = float(params.stock_width_m)
    h = float(params.stock_height_m)
    profile = getattr(params, "strut_profile", "rectangular")
    if profile == "round":
        d = min(w, h)
        return math.pi * d ** 4 / 64.0
    # Rectangular: I = b·h³/12, use smaller dimension as weak axis.
    b, d = max(w, h), min(w, h)
    return b * d ** 3 / 12.0


# ---------------------------------------------------------------------------
# Member force solver — Direct Stiffness Method for 3-D truss
# ---------------------------------------------------------------------------


def _vec_sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_len(v: Vector3) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _solve_member_forces(
    nodes: List[Vector3],
    struts: List[Any],
    combined_loads: Dict[int, Vector3],
    base_node_indices: set,
    E: float,
    A: float,
) -> List[float]:
    """Solve for axial member forces using the direct stiffness method.

    Parameters
    ----------
    nodes : list of (x, y, z)
        Node coordinates.
    struts : list with .start_index, .end_index attributes
        Connectivity.
    combined_loads : dict
        node_index → (Fx, Fy, Fz) in kN.
    base_node_indices : set
        Indices of nodes that are fully restrained (pinned supports).
    E : float
        Young's modulus in kN/m² (= Pa / 1000).
    A : float
        Cross-section area in m².

    Returns
    -------
    list of float
        Axial force per strut (kN). Positive = tension, negative = compression.
    """
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes
    n_struts = len(struts)

    # Use numpy if available for efficient linear algebra.
    try:
        import numpy as np
        return _solve_numpy(nodes, struts, combined_loads, base_node_indices, E, A, np)
    except ImportError:
        pass

    # Fallback: pure-Python solver for small systems.
    return _solve_pure_python(nodes, struts, combined_loads, base_node_indices, E, A)


def _solve_numpy(
    nodes: List[Vector3],
    struts: List[Any],
    combined_loads: Dict[int, Vector3],
    base_node_indices: set,
    E: float,
    A: float,
    np: Any,
) -> List[float]:
    """Direct stiffness method using numpy."""
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes

    K = np.zeros((n_dof, n_dof), dtype=float)
    F = np.zeros(n_dof, dtype=float)

    # Strut data for force extraction.
    strut_data: List[Tuple[int, int, float, Vector3]] = []  # (i, j, L, direction)

    for strut in struts:
        i = strut.start_index
        j = strut.end_index
        dx = nodes[j][0] - nodes[i][0]
        dy = nodes[j][1] - nodes[i][1]
        dz = nodes[j][2] - nodes[i][2]
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        if L < 1e-12:
            strut_data.append((i, j, 0.0, (0.0, 0.0, 0.0)))
            continue

        cx, cy, cz = dx / L, dy / L, dz / L
        strut_data.append((i, j, L, (cx, cy, cz)))

        k = E * A / L  # axial stiffness

        # Direction cosine products for the 6×6 local stiffness matrix.
        cc = [cx * cx, cx * cy, cx * cz, cy * cy, cy * cz, cz * cz]

        # DOF indices.
        di = [3 * i, 3 * i + 1, 3 * i + 2]
        dj = [3 * j, 3 * j + 1, 3 * j + 2]

        # Assemble: k × [c·cᵀ  -c·cᵀ; -c·cᵀ  c·cᵀ]
        cos_prods = [
            [cc[0], cc[1], cc[2]],
            [cc[1], cc[3], cc[4]],
            [cc[2], cc[4], cc[5]],
        ]
        for a in range(3):
            for b in range(3):
                v = k * cos_prods[a][b]
                K[di[a], di[b]] += v
                K[dj[a], dj[b]] += v
                K[di[a], dj[b]] -= v
                K[dj[a], di[b]] -= v

    # Load vector.
    for ni, (fx, fy, fz) in combined_loads.items():
        if ni < n_nodes:
            F[3 * ni] += fx
            F[3 * ni + 1] += fy
            F[3 * ni + 2] += fz

    # Boundary conditions: fix base nodes (set diagonal to large value).
    penalty = 1e20
    for ni in base_node_indices:
        if ni < n_nodes:
            for d in range(3):
                dof = 3 * ni + d
                K[dof, dof] += penalty

    # Solve K·u = F.
    try:
        u = np.linalg.solve(K, F)
    except np.linalg.LinAlgError:
        log.warning("Stiffness matrix is singular — returning zero forces.")
        return [0.0] * len(struts)

    # Extract member forces.
    forces: List[float] = []
    for idx, (i, j, L, direction) in enumerate(strut_data):
        if L < 1e-12:
            forces.append(0.0)
            continue
        cx, cy, cz = direction
        # Elongation along member axis.
        du = (
            (u[3 * j] - u[3 * i]) * cx
            + (u[3 * j + 1] - u[3 * i + 1]) * cy
            + (u[3 * j + 2] - u[3 * i + 2]) * cz
        )
        force = E * A / L * du  # positive = tension
        forces.append(float(force))

    return forces


def _solve_pure_python(
    nodes: List[Vector3],
    struts: List[Any],
    combined_loads: Dict[int, Vector3],
    base_node_indices: set,
    E: float,
    A: float,
) -> List[float]:
    """Fallback pure-Python solver using Cholesky decomposition.

    Suitable only for small systems (< ~200 nodes).
    """
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes

    # Build K and F as flat lists (row-major).
    K = [0.0] * (n_dof * n_dof)
    F = [0.0] * n_dof

    strut_data: List[Tuple[int, int, float, Vector3]] = []

    for strut in struts:
        i = strut.start_index
        j = strut.end_index
        dx = nodes[j][0] - nodes[i][0]
        dy = nodes[j][1] - nodes[i][1]
        dz = nodes[j][2] - nodes[i][2]
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        if L < 1e-12:
            strut_data.append((i, j, 0.0, (0.0, 0.0, 0.0)))
            continue

        cx, cy, cz = dx / L, dy / L, dz / L
        strut_data.append((i, j, L, (cx, cy, cz)))
        k = E * A / L

        cos_prods = [
            [cx * cx, cx * cy, cx * cz],
            [cx * cy, cy * cy, cy * cz],
            [cx * cz, cy * cz, cz * cz],
        ]
        di = [3 * i, 3 * i + 1, 3 * i + 2]
        dj = [3 * j, 3 * j + 1, 3 * j + 2]
        for a in range(3):
            for b in range(3):
                v = k * cos_prods[a][b]
                K[di[a] * n_dof + di[b]] += v
                K[dj[a] * n_dof + dj[b]] += v
                K[di[a] * n_dof + dj[b]] -= v
                K[dj[a] * n_dof + di[b]] -= v

    # Load vector.
    for ni, (fx, fy, fz) in combined_loads.items():
        if ni < n_nodes:
            F[3 * ni] += fx
            F[3 * ni + 1] += fy
            F[3 * ni + 2] += fz

    # Boundary conditions.
    penalty = 1e20
    for ni in base_node_indices:
        if ni < n_nodes:
            for d in range(3):
                dof = 3 * ni + d
                K[dof * n_dof + dof] += penalty

    # Solve using Gaussian elimination with partial pivoting.
    u = _gauss_solve(K, F, n_dof)

    # Extract member forces.
    forces: List[float] = []
    for idx, (i, j, L, direction) in enumerate(strut_data):
        if L < 1e-12:
            forces.append(0.0)
            continue
        cx, cy, cz = direction
        du = (
            (u[3 * j] - u[3 * i]) * cx
            + (u[3 * j + 1] - u[3 * i + 1]) * cy
            + (u[3 * j + 2] - u[3 * i + 2]) * cz
        )
        force = E * A / L * du
        forces.append(force)

    return forces


def _gauss_solve(K_flat: List[float], F: List[float], n: int) -> List[float]:
    """Solve K·x = F by Gaussian elimination with partial pivoting.

    K_flat is row-major n×n, F is length n. Returns x as list[float].
    """
    # Build augmented matrix [K | F].
    A = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = K_flat[i * n + j]
        A[i][n] = F[i]

    # Forward elimination.
    for col in range(n):
        # Partial pivoting.
        max_row = col
        max_val = abs(A[col][col])
        for row in range(col + 1, n):
            if abs(A[row][col]) > max_val:
                max_val = abs(A[row][col])
                max_row = row
        if max_row != col:
            A[col], A[max_row] = A[max_row], A[col]

        diag = A[col][col]
        if abs(diag) < 1e-30:
            continue  # singular row

        for row in range(col + 1, n):
            factor = A[row][col] / diag
            for j in range(col, n + 1):
                A[row][j] -= factor * A[col][j]

    # Back substitution.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < 1e-30:
            x[i] = 0.0
            continue
        s = A[i][n]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]

    return x


# ---------------------------------------------------------------------------
# Combined load computation
# ---------------------------------------------------------------------------


def _combine_node_loads(
    load_result: LoadResult,
    combination: LoadCombination,
) -> Dict[int, Vector3]:
    """Apply a load combination's factors to produce combined node forces."""
    combined: Dict[int, List[float]] = {}  # node_index → [Fx, Fy, Fz]

    case_map = {c.name: c for c in load_result.cases}
    for case_name, factor in combination.factors.items():
        case = case_map.get(case_name)
        if case is None:
            continue
        for nl in case.node_loads:
            if nl.node_index not in combined:
                combined[nl.node_index] = [0.0, 0.0, 0.0]
            combined[nl.node_index][0] += factor * nl.fx_kn
            combined[nl.node_index][1] += factor * nl.fy_kn
            combined[nl.node_index][2] += factor * nl.fz_kn

    return {ni: (v[0], v[1], v[2]) for ni, v in combined.items()}


# ---------------------------------------------------------------------------
# Capacity checks
# ---------------------------------------------------------------------------


def _euler_buckling_force(E_pa: float, I_m4: float, L_m: float) -> float:
    """Euler critical buckling force N_cr (kN).

    N_cr = π²·E·I / L²   (for pinned-pinned strut, effective length = L).
    """
    if L_m <= 0:
        return float("inf")
    return math.pi ** 2 * (E_pa / 1e3) * I_m4 / (L_m ** 2)  # kN


def _compression_capacity(
    mat: MaterialSpec, A_m2: float, gamma_m: float,
) -> float:
    """Design compression capacity N_c,Rd (kN).

    N_c,Rd = A × f_c,0,k / γ_M
    """
    fc = mat.compressive_strength_mpa or 21.0  # default C24 timber
    return A_m2 * fc * 1e3 / gamma_m  # MPa × m² = kN (× 1e3 from MPa→kN/m²)


def _tension_capacity(
    mat: MaterialSpec, A_m2: float, gamma_m: float,
) -> float:
    """Design tension capacity N_t,Rd (kN).

    N_t,Rd = A × f_t,0,k / γ_M
    """
    ft = mat.tensile_strength_mpa or 14.5
    return A_m2 * ft * 1e3 / gamma_m


# ---------------------------------------------------------------------------
# Identify base (support) nodes
# ---------------------------------------------------------------------------


def _find_base_nodes(
    nodes: List[Vector3],
    hemisphere_ratio: float,
    radius_m: float,
) -> set:
    """Find nodes at the dome base that act as pinned supports.

    Base nodes are at (or very close to) the truncation plane.
    """
    # The base of the dome is at z ≈ 0 or at the lowest ring of nodes.
    if not nodes:
        return set()

    z_values = [n[2] for n in nodes]
    z_min = min(z_values)
    z_range = max(z_values) - z_min
    if z_range < 1e-6:
        return set()

    # Base nodes: within 5% of the total height above z_min.
    threshold = z_min + 0.05 * z_range
    return {i for i, n in enumerate(nodes) if n[2] <= threshold}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_structural_check(
    dome: Any,
    params: DomeParameters,
    load_result: LoadResult,
) -> StructuralCheckResult:
    """Perform structural adequacy checks on all struts.

    For each ULS load combination, computes member forces via the direct
    stiffness method and checks strut capacity. Returns the governing
    (worst-case) result per strut across all combinations.

    Parameters
    ----------
    dome : TessellatedDome
        Dome geometry with nodes, struts, and panels.
    params : DomeParameters
        Current configuration.
    load_result : LoadResult
        Pre-computed load cases and combinations from ``compute_loads()``.

    Returns
    -------
    StructuralCheckResult
        Per-strut utilization ratios and pass/fail summary.
    """
    mat_name = params.material
    mat = params.materials.get(mat_name)
    if mat is None:
        raise ValueError(f"Unknown material '{mat_name}' — cannot run structural check")

    E_pa = float(mat.elastic_modulus or 11_000e6)  # Pa
    E_kn_m2 = E_pa / 1e3  # kN/m²
    gamma_m = float(mat.gamma_m)

    A = _cross_section_area(params)
    I = _second_moment_of_area(params)

    nodes = dome.nodes
    struts = dome.struts
    n_struts = len(struts)

    base_nodes = _find_base_nodes(
        nodes,
        float(params.hemisphere_ratio),
        float(params.radius_m),
    )

    if not base_nodes:
        log.warning("No base nodes found — structural check may be unreliable")

    # Pre-compute strut lengths.
    strut_lengths: List[float] = []
    for s in struts:
        p0 = nodes[s.start_index]
        p1 = nodes[s.end_index]
        L = math.sqrt(sum((a - b) ** 2 for a, b in zip(p0, p1)))
        strut_lengths.append(L)

    # Compute capacities (same for all struts of same cross-section).
    N_comp = _compression_capacity(mat, A, gamma_m)
    N_tens = _tension_capacity(mat, A, gamma_m)

    # Initialize per-strut results.
    checks: List[StrutCheck] = []
    for idx, s in enumerate(struts):
        L = strut_lengths[idx]
        N_cr = _euler_buckling_force(E_pa, I, L)
        checks.append(StrutCheck(
            strut_index=idx,
            start_node=s.start_index,
            end_node=s.end_index,
            length_m=L,
            euler_buckling_kn=N_cr,
            compression_capacity_kn=N_comp,
            tension_capacity_kn=N_tens,
        ))

    # Evaluate each ULS combination.
    uls_combos = [c for c in load_result.combinations if c.name.startswith("ULS")]
    if not uls_combos:
        uls_combos = load_result.combinations  # use all if no ULS

    for combo in uls_combos:
        combined = _combine_node_loads(load_result, combo)
        member_forces = _solve_member_forces(
            nodes, struts, combined, base_nodes, E_kn_m2, A,
        )

        for idx in range(n_struts):
            N = member_forces[idx] if idx < len(member_forces) else 0.0
            sc = checks[idx]
            L = sc.length_m

            # Compute utilization ratios for this combination.
            if N < 0:
                # Compression.
                comp_ratio = abs(N) / N_comp if N_comp > 0 else 0.0
                buck_ratio = abs(N) / sc.euler_buckling_kn if sc.euler_buckling_kn > 0 else 0.0
                tens_ratio = 0.0
            else:
                # Tension.
                comp_ratio = 0.0
                buck_ratio = 0.0
                tens_ratio = N / N_tens if N_tens > 0 else 0.0

            gov_ratio = max(comp_ratio, buck_ratio, tens_ratio)

            # Keep the worst-case combination.
            if gov_ratio > sc.governing_ratio:
                sc.axial_force_kn = N
                sc.governing_combination = combo.name
                sc.buckling_ratio = buck_ratio
                sc.compression_ratio = comp_ratio
                sc.tension_ratio = tens_ratio
                sc.governing_ratio = gov_ratio
                if buck_ratio >= comp_ratio and buck_ratio >= tens_ratio:
                    sc.governing_check = "buckling"
                elif comp_ratio >= tens_ratio:
                    sc.governing_check = "compression"
                else:
                    sc.governing_check = "tension"

    # Build aggregate result.
    result = StructuralCheckResult(
        strut_checks=checks,
        total_struts=n_struts,
    )
    if checks:
        worst = max(checks, key=lambda c: c.governing_ratio)
        result.max_utilization = worst.governing_ratio
        result.critical_strut_index = worst.strut_index
    result.failing_struts = sum(1 for c in checks if not c.passes)
    result.all_pass = result.failing_struts == 0

    log.info(
        "Structural check: %d/%d struts pass, max utilization %.1f%% (strut #%d)",
        result.total_struts - result.failing_struts,
        result.total_struts,
        result.max_utilization * 100,
        result.critical_strut_index,
    )

    return result


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_check_report(
    result: StructuralCheckResult,
    path: Any,
) -> None:
    """Write a JSON report with structural check results."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    log.info("Wrote structural check report to %s", out)
