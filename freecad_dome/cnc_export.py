"""CNC export — per-strut STEP files and cutting table for manufacturing.

Generates:

* **Individual STEP files** for each unique strut type (FreeCAD required).
* **Cutting table CSV** with lengths, angles, and stock dimensions.
* **Node plate STEP files** for CNC-cut connector plates (when available).

The module classifies struts into unique *manufacturing types* based on
length, miter angles, and bevel angles (with configurable tolerance).
Each type gets one representative STEP file; the cutting table lists the
quantity of each type.

Directory structure::

    cnc_export/
      struts/
        Strut_A_L1245mm.step
        Strut_B_L1187mm.step
        ...
      plates/
        Plate_N001.step
        ...
      cutting_table.csv
      cnc_manifest.json

Usage::

    from freecad_dome.cnc_export import (
        classify_strut_types,
        write_cutting_table,
        write_cnc_manifest,
        export_strut_step_files,
    )
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .production import StrutCutSpec, compute_strut_cuts
from .parameters import DomeParameters
from .tessellation import TessellatedDome

__all__ = [
    "StrutType",
    "CncExportResult",
    "classify_strut_types",
    "write_cutting_table",
    "write_cnc_manifest",
    "export_strut_step_files",
    "export_plate_step_files",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StrutType:
    """A unique manufacturing type — one or more struts with identical cuts.

    All struts in the same type can be manufactured with the same
    machine setup (same stock size, same miter/bevel angles, same length).
    """

    type_id: str  # e.g. "A", "B", "C"
    representative_index: int  # strut index of the representative
    length_mm: float  # net cutting length
    raw_length_mm: float  # with kerf allowance
    stock_width_mm: float
    stock_height_mm: float
    start_miter_deg: float
    end_miter_deg: float
    start_bevel_deg: float
    end_bevel_deg: float
    quantity: int  # how many struts of this type
    strut_indices: List[int] = field(default_factory=list)

    @property
    def filename(self) -> str:
        """Naming scheme: Strut_{type_id}_L{length}mm.step"""
        return f"Strut_{self.type_id}_L{round(self.length_mm)}mm.step"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type_id": self.type_id,
            "representative_index": self.representative_index,
            "length_mm": round(self.length_mm, 1),
            "raw_length_mm": round(self.raw_length_mm, 1),
            "stock_width_mm": round(self.stock_width_mm, 1),
            "stock_height_mm": round(self.stock_height_mm, 1),
            "start_miter_deg": round(self.start_miter_deg, 2),
            "end_miter_deg": round(self.end_miter_deg, 2),
            "start_bevel_deg": round(self.start_bevel_deg, 2),
            "end_bevel_deg": round(self.end_bevel_deg, 2),
            "quantity": self.quantity,
            "filename": self.filename,
            "strut_indices": self.strut_indices,
        }


@dataclass
class CncExportResult:
    """Summary of CNC export outputs."""

    strut_types: List[StrutType] = field(default_factory=list)
    total_struts: int = 0
    unique_types: int = 0
    step_files_written: int = 0
    plate_files_written: int = 0
    cutting_table_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_struts": self.total_struts,
            "unique_types": self.unique_types,
            "step_files_written": self.step_files_written,
            "plate_files_written": self.plate_files_written,
            "cutting_table_path": self.cutting_table_path,
            "strut_types": [st.to_dict() for st in self.strut_types],
        }


# ---------------------------------------------------------------------------
# Type classification
# ---------------------------------------------------------------------------

# Tolerances for grouping struts into the same manufacturing type.
_LENGTH_TOL_MM = 0.5   # length tolerance
_ANGLE_TOL_DEG = 0.5   # miter/bevel angle tolerance


def _type_key(cut: StrutCutSpec) -> Tuple[int, int, int, int, int]:
    """Quantize a cut spec into a hashable type key.

    Two struts with the same key can be manufactured identically.
    """
    return (
        round(cut.net_length_mm / _LENGTH_TOL_MM),
        round(cut.start_miter_deg / _ANGLE_TOL_DEG),
        round(cut.end_miter_deg / _ANGLE_TOL_DEG),
        round(cut.start_bevel_deg / _ANGLE_TOL_DEG),
        round(cut.end_bevel_deg / _ANGLE_TOL_DEG),
    )


def classify_strut_types(
    dome: TessellatedDome,
    params: DomeParameters,
) -> List[StrutType]:
    """Classify dome struts into unique manufacturing types.

    Uses the production module's ``compute_strut_cuts()`` to get per-strut
    cut specifications, then groups by (length, miter, bevel) within
    tolerance.

    Returns a list of :class:`StrutType` sorted by type_id.
    """
    cuts = compute_strut_cuts(dome, params)
    if not cuts:
        return []

    # Group by quantized key.
    buckets: Dict[Tuple[int, ...], List[StrutCutSpec]] = {}
    for cut in cuts:
        key = _type_key(cut)
        buckets.setdefault(key, []).append(cut)

    # Sort buckets by length (ascending) and assign type labels.
    sorted_keys = sorted(buckets.keys(), key=lambda k: k[0])

    types: List[StrutType] = []
    for idx, key in enumerate(sorted_keys):
        members = buckets[key]
        rep = members[0]
        label = chr(ord("A") + idx) if idx < 26 else f"T{idx:02d}"
        types.append(StrutType(
            type_id=label,
            representative_index=rep.strut_index,
            length_mm=rep.net_length_mm,
            raw_length_mm=rep.raw_length_mm,
            stock_width_mm=rep.stock_width_mm,
            stock_height_mm=rep.stock_height_mm,
            start_miter_deg=rep.start_miter_deg,
            end_miter_deg=rep.end_miter_deg,
            start_bevel_deg=rep.start_bevel_deg,
            end_bevel_deg=rep.end_bevel_deg,
            quantity=len(members),
            strut_indices=[m.strut_index for m in members],
        ))

    return types


# ---------------------------------------------------------------------------
# Cutting table CSV
# ---------------------------------------------------------------------------


def write_cutting_table(
    strut_types: List[StrutType],
    path: Any,
) -> None:
    """Write a CSV cutting table with one row per unique strut type.

    Columns: Type, Qty, NetLength_mm, RawLength_mm, StockW_mm, StockH_mm,
    StartMiter_deg, EndMiter_deg, StartBevel_deg, EndBevel_deg, STEP_File.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "Type",
        "Qty",
        "NetLength_mm",
        "RawLength_mm",
        "StockW_mm",
        "StockH_mm",
        "StartMiter_deg",
        "EndMiter_deg",
        "StartBevel_deg",
        "EndBevel_deg",
        "STEP_File",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for st in strut_types:
            writer.writerow([
                st.type_id,
                st.quantity,
                round(st.length_mm, 1),
                round(st.raw_length_mm, 1),
                round(st.stock_width_mm, 1),
                round(st.stock_height_mm, 1),
                round(st.start_miter_deg, 2),
                round(st.end_miter_deg, 2),
                round(st.start_bevel_deg, 2),
                round(st.end_bevel_deg, 2),
                st.filename,
            ])

    log.info("Wrote cutting table to %s (%d types)", out, len(strut_types))


# ---------------------------------------------------------------------------
# CNC manifest JSON
# ---------------------------------------------------------------------------


def write_cnc_manifest(
    result: CncExportResult,
    path: Any,
) -> None:
    """Write a JSON manifest summarizing the CNC export."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    log.info("Wrote CNC manifest to %s", out)


# ---------------------------------------------------------------------------
# STEP export (FreeCAD-dependent)
# ---------------------------------------------------------------------------


def _find_strut_object_by_index(
    doc: Any,
    strut_index: int,
) -> Any:
    """Find a FreeCAD document object for a given tessellation strut index.

    Looks for the ``StrutIndex`` custom property or falls back to name
    pattern matching.
    """
    for obj in getattr(doc, "Objects", []):
        # Check custom property first.
        if getattr(obj, "StrutIndex", None) == strut_index:
            return obj
        # Fallback: match name pattern "Strut_*_{seq:03d}*"
        name = getattr(obj, "Name", "") or ""
        if not name.startswith("Strut_"):
            continue
        # StrutFamily property may contain the tessellation index.
        family = getattr(obj, "StrutFamily", None)
        if family is not None:
            try:
                fam_idx = int(str(family).split("_")[0])
                if fam_idx == strut_index:
                    return obj
            except (ValueError, IndexError):
                pass
    return None


def export_strut_step_files(
    strut_types: List[StrutType],
    doc: Any,
    base_dir: Any,
) -> int:
    """Export one STEP file per unique strut type.

    Uses the representative strut's FreeCAD solid geometry.

    Parameters
    ----------
    strut_types : list of StrutType
        Classified strut types from ``classify_strut_types()``.
    doc : FreeCAD.Document
        The FreeCAD document containing strut geometry.
    base_dir : path-like
        Root directory for CNC export (``cnc_export/``).

    Returns
    -------
    int
        Number of STEP files successfully written.
    """
    struts_dir = Path(base_dir) / "struts"
    struts_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for st in strut_types:
        obj = _find_strut_object_by_index(doc, st.representative_index)
        if obj is None:
            log.warning(
                "No FreeCAD object found for strut type %s (index %d) — skipping STEP",
                st.type_id,
                st.representative_index,
            )
            continue

        shape = getattr(obj, "Shape", None)
        if shape is None or not getattr(shape, "isValid", lambda: False)():
            log.warning("Strut type %s has no valid Shape — skipping STEP", st.type_id)
            continue

        step_path = struts_dir / st.filename
        try:
            shape.exportStep(str(step_path))
            written += 1
            log.debug("Wrote %s", step_path)
        except Exception as exc:
            log.warning("STEP export failed for type %s: %s", st.type_id, exc)

    log.info("Exported %d / %d strut STEP files to %s", written, len(strut_types), struts_dir)
    return written


def export_plate_step_files(
    doc: Any,
    base_dir: Any,
) -> int:
    """Export connector plate geometry as STEP files.

    Searches the FreeCAD document for objects with IfcType "IfcPlate"
    and exports each one.

    Returns the number of files written.
    """
    plates_dir = Path(base_dir) / "plates"
    plates_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    if doc is None:
        return written

    for obj in getattr(doc, "Objects", []):
        if getattr(obj, "IfcType", "") != "IfcPlate":
            continue
        shape = getattr(obj, "Shape", None)
        if shape is None or not getattr(shape, "isValid", lambda: False)():
            continue
        name = getattr(obj, "Name", f"Plate_{written:03d}")
        step_path = plates_dir / f"{name}.step"
        try:
            shape.exportStep(str(step_path))
            written += 1
        except Exception as exc:
            log.warning("STEP export failed for plate %s: %s", name, exc)

    log.info("Exported %d plate STEP files to %s", written, plates_dir)
    return written


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def cnc_export_for_dome(
    dome: TessellatedDome,
    params: DomeParameters,
    out_dir: Any,
    doc: Any = None,
) -> CncExportResult:
    """Run the complete CNC export pipeline.

    1. Classify strut types.
    2. Write cutting table CSV.
    3. Export STEP files (if FreeCAD doc available).
    4. Write CNC manifest JSON.

    Parameters
    ----------
    dome : TessellatedDome
        Dome geometry.
    params : DomeParameters
        Configuration.
    out_dir : path-like
        Output directory (e.g. ``exports/cnc_export``).
    doc : FreeCAD.Document, optional
        FreeCAD document for STEP export. If None, only CSV/JSON are written.

    Returns
    -------
    CncExportResult
    """
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    # 1. Classify.
    strut_types = classify_strut_types(dome, params)

    # 2. Cutting table.
    csv_path = base / "cutting_table.csv"
    write_cutting_table(strut_types, csv_path)

    # 3. STEP files.
    step_count = 0
    plate_count = 0
    if doc is not None:
        step_count = export_strut_step_files(strut_types, doc, base)
        plate_count = export_plate_step_files(doc, base)

    # 4. Manifest.
    result = CncExportResult(
        strut_types=strut_types,
        total_struts=sum(st.quantity for st in strut_types),
        unique_types=len(strut_types),
        step_files_written=step_count,
        plate_files_written=plate_count,
        cutting_table_path=str(csv_path),
    )
    manifest_path = base / "cnc_manifest.json"
    write_cnc_manifest(result, manifest_path)

    log.info(
        "CNC export complete: %d types, %d struts, %d STEP files",
        result.unique_types,
        result.total_struts,
        result.step_files_written,
    )
    return result
