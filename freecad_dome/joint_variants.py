"""Variant configuration for frame/joint generation.

This module is intentionally small: it defines the knobs we will use to compare
multiple construction approaches (panel-first, node-hub, relief/undercut, etc.).

It is pure-Python so it can be used by headless checkers and exporters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def _coerce_enum(value: str | None, enum_type: type[Enum], *, default: Enum) -> Enum:
    if value is None:
        return default
    raw = str(value).strip()
    if not raw:
        return default
    raw_norm = raw.lower().replace("-", "_").replace(" ", "_")
    for item in enum_type:  # type: ignore[assignment]
        if str(item.value).lower() == raw_norm:
            return item
    # Allow passing member name (e.g. PANEL_INSET)
    for item in enum_type:  # type: ignore[assignment]
        if str(item.name).lower() == raw_norm:
            return item
    allowed = ", ".join(str(i.value) for i in enum_type)  # type: ignore[arg-type]
    raise ValueError(f"Invalid {enum_type.__name__}: {raw!r}. Allowed: {allowed}")


class FrameConstruction(str, Enum):
    # Current approach: frame is generated from each panel plane.
    PANEL_INSET = "panel_inset"
    # Future approach: node/hub drives compatibility across panels.
    NODE_HUB = "node_hub"


class CornerTreatment(str, Enum):
    # Standard miter inside the panel plane.
    MITER = "miter"
    # Remove additional material near inner corner to avoid multi-panel clashes.
    RELIEF = "relief"


@dataclass(frozen=True, slots=True)
class JointVariant:
    construction: FrameConstruction = FrameConstruction.PANEL_INSET
    corner: CornerTreatment = CornerTreatment.MITER
    relief_depth_m: float = 0.0

    def validate(self) -> None:
        if self.relief_depth_m < 0:
            raise ValueError("relief_depth_m must be >= 0")


def load_joint_variant_from_env(
    *,
    construction_env: str = "DOME_JOINT_CONSTRUCTION",
    corner_env: str = "DOME_JOINT_CORNER",
    relief_env: str = "DOME_JOINT_RELIEF_DEPTH_M",
) -> JointVariant:
    """Load a `JointVariant` from environment variables.

    Defaults keep current behavior:
    - construction=panel_inset
    - corner=miter
    - relief_depth_m=0
    """

    import os

    construction = _coerce_enum(
        os.environ.get(construction_env),
        FrameConstruction,
        default=FrameConstruction.PANEL_INSET,
    )
    corner = _coerce_enum(
        os.environ.get(corner_env),
        CornerTreatment,
        default=CornerTreatment.MITER,
    )
    relief_raw = os.environ.get(relief_env)
    relief_depth_m = float(relief_raw) if relief_raw not in (None, "") else 0.0
    variant = JointVariant(
        construction=construction,  # type: ignore[arg-type]
        corner=corner,  # type: ignore[arg-type]
        relief_depth_m=relief_depth_m,
    )
    variant.validate()
    return variant
