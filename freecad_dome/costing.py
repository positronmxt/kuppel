"""Cost estimation and full material bill for geodesic domes.

Covers:
- Material price catalogue (€/m for timber, €/m² for covering, etc.)
- Hardware BOM: bolts, nuts, washers, angle brackets, gaskets
- Covering cut-list with nesting / waste estimation
- Waste percentage and material cost coefficient
- CSV export of the full cost estimate
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .parameters import DomeParameters
from .tessellation import Strut, TessellatedDome, Vector3

__all__ = [
    "PriceCatalogueEntry",
    "DEFAULT_PRICE_CATALOGUE",
    "BomItem",
    "CoveringSheet",
    "CostEstimate",
    "load_price_catalogue",
    "cost_estimate_for_params",
    "write_cost_report",
    "write_cost_csv",
]


# ---------------------------------------------------------------------------
# Price catalogue
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class PriceCatalogueEntry:
    """One entry in the material price catalogue."""
    name: str
    unit: str           # "m" | "m2" | "pcs" | "kg"
    price_eur: float    # price per unit
    category: str       # "timber" | "covering" | "hardware" | "gasket" | "connector"
    supplier: str = ""  # optional supplier / source


DEFAULT_PRICE_CATALOGUE: Dict[str, PriceCatalogueEntry] = {
    # Timber
    "timber_45x70_pine": PriceCatalogueEntry(
        name="Pine 45×70 mm KD C24",
        unit="m",
        price_eur=3.50,
        category="timber",
    ),
    "timber_45x95_pine": PriceCatalogueEntry(
        name="Pine 45×95 mm KD C24",
        unit="m",
        price_eur=4.80,
        category="timber",
    ),
    "timber_45x120_pine": PriceCatalogueEntry(
        name="Pine 45×120 mm KD C24",
        unit="m",
        price_eur=5.90,
        category="timber",
    ),
    # Covering
    "glass_4mm": PriceCatalogueEntry(
        name="Float glass 4 mm",
        unit="m2",
        price_eur=25.00,
        category="covering",
    ),
    "pc_multiwall_10": PriceCatalogueEntry(
        name="Polycarbonate multiwall 10 mm",
        unit="m2",
        price_eur=15.00,
        category="covering",
    ),
    "pc_multiwall_16": PriceCatalogueEntry(
        name="Polycarbonate multiwall 16 mm",
        unit="m2",
        price_eur=22.00,
        category="covering",
    ),
    # Hardware
    "bolt_m10x80": PriceCatalogueEntry(
        name="Hex bolt M10×80 A2",
        unit="pcs",
        price_eur=0.45,
        category="hardware",
    ),
    "bolt_m12x100": PriceCatalogueEntry(
        name="Hex bolt M12×100 A2",
        unit="pcs",
        price_eur=0.65,
        category="hardware",
    ),
    "nut_m10": PriceCatalogueEntry(
        name="Hex nut M10 A2",
        unit="pcs",
        price_eur=0.10,
        category="hardware",
    ),
    "nut_m12": PriceCatalogueEntry(
        name="Hex nut M12 A2",
        unit="pcs",
        price_eur=0.15,
        category="hardware",
    ),
    "washer_m10": PriceCatalogueEntry(
        name="Flat washer M10 A2",
        unit="pcs",
        price_eur=0.05,
        category="hardware",
    ),
    "washer_m12": PriceCatalogueEntry(
        name="Flat washer M12 A2",
        unit="pcs",
        price_eur=0.08,
        category="hardware",
    ),
    "angle_bracket_90x90": PriceCatalogueEntry(
        name="Angle bracket 90×90×2.5 galv.",
        unit="pcs",
        price_eur=0.85,
        category="hardware",
    ),
    # Connector plates
    "plate_steel_3mm": PriceCatalogueEntry(
        name="Steel plate 3 mm (laser-cut)",
        unit="m2",
        price_eur=120.00,
        category="connector",
    ),
    # Gasket
    "gasket_epdm": PriceCatalogueEntry(
        name="EPDM gasket (D-profile)",
        unit="m",
        price_eur=1.20,
        category="gasket",
    ),
}


# ---------------------------------------------------------------------------
# External catalogue loading
# ---------------------------------------------------------------------------

# Currency conversion rates relative to EUR
_CURRENCY_RATES: Dict[str, float] = {
    "EUR": 1.0,
    "USD": 1.08,
    "GBP": 0.86,
}

SUPPORTED_CURRENCIES = tuple(_CURRENCY_RATES.keys())


def load_price_catalogue(
    json_path: str | Path,
    base: Dict[str, PriceCatalogueEntry] | None = None,
) -> Dict[str, PriceCatalogueEntry]:
    """Load an external JSON price catalogue and merge onto *base*.

    JSON format — a dict of ``{ key: { name, unit, price_eur, category, supplier? } }``::

        {
            "timber_45x70_pine": {
                "name": "Pine 45×70 mm KD C24",
                "unit": "m",
                "price_eur": 4.10,
                "category": "timber",
                "supplier": "Puumerkki OÜ"
            }
        }

    Entries in the JSON override entries in *base* with the same key;
    extra keys are added.
    """
    cat = dict(base or DEFAULT_PRICE_CATALOGUE)
    p = Path(json_path)
    if not p.is_file():
        logging.warning("Price catalogue not found: %s — using defaults", p)
        return cat
    try:
        with open(p, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)
        for key, vals in raw.items():
            cat[key] = PriceCatalogueEntry(
                name=vals.get("name", key),
                unit=vals.get("unit", "pcs"),
                price_eur=float(vals.get("price_eur", 0.0)),
                category=vals.get("category", "other"),
                supplier=vals.get("supplier", ""),
            )
        logging.info("Loaded %d price entries from %s", len(raw), p)
    except Exception as exc:
        logging.warning("Failed to load price catalogue %s: %s", p, exc)
    return cat


# ---------------------------------------------------------------------------
# BOM item
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BomItem:
    """One line in the Bill of Materials."""
    category: str
    description: str
    unit: str
    quantity: float
    unit_price_eur: float
    total_eur: float
    waste_pct: float = 0.0
    supplier: str = ""


@dataclass(slots=True)
class CoveringSheet:
    """One panel's covering material requirement."""
    panel_index: int
    area_m2: float
    perimeter_m: float
    sheet_area_m2: float   # including waste
    waste_pct: float


@dataclass
class CostEstimate:
    """Complete cost estimate."""
    bom: List[BomItem]
    covering_sheets: List[CoveringSheet]
    total_material_eur: float
    total_hardware_eur: float
    total_labour_eur: float
    total_overhead_eur: float
    total_eur: float
    waste_coefficient: float
    timber_total_m: float
    covering_total_m2: float
    bolt_count: int
    gasket_total_m: float
    currency: str = "EUR"


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _dist(a: Vector3, b: Vector3) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _polygon_area_3d(points: List[Vector3]) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    sx = sy = sz = 0.0
    for i in range(n):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[(i + 1) % n]
        sx += (y1 - y2) * (z1 + z2)
        sy += (z1 - z2) * (x1 + x2)
        sz += (x1 - x2) * (y1 + y2)
    return 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)


def _panel_perimeter(nodes: List[Vector3], node_indices: Tuple[int, ...]) -> float:
    total = 0.0
    n = len(node_indices)
    for i in range(n):
        a = nodes[node_indices[i]]
        b = nodes[node_indices[(i + 1) % n]]
        total += _dist(a, b)
    return total


# ---------------------------------------------------------------------------
# BOM computation
# ---------------------------------------------------------------------------

def _timber_key_for_stock(params: DomeParameters) -> str:
    """Match stock dimensions to a catalogue entry."""
    w = params.stock_width_m * 1000
    h = params.stock_height_m * 1000
    if w <= 50 and h <= 75:
        return "timber_45x70_pine"
    elif w <= 50 and h <= 100:
        return "timber_45x95_pine"
    else:
        return "timber_45x120_pine"


def _bolt_key_for_connector(params: DomeParameters) -> str:
    """Match bolt size to a catalogue entry."""
    d_mm = params.node_connector_bolt_diameter_m * 1000
    if d_mm <= 10:
        return "bolt_m10x80"
    return "bolt_m12x100"


def _nut_key(params: DomeParameters) -> str:
    d_mm = params.node_connector_bolt_diameter_m * 1000
    return "nut_m10" if d_mm <= 10 else "nut_m12"


def _washer_key(params: DomeParameters) -> str:
    d_mm = params.node_connector_bolt_diameter_m * 1000
    return "washer_m10" if d_mm <= 10 else "washer_m12"


def _covering_key(params: DomeParameters) -> str:
    ct = getattr(params, "covering_type", "glass")
    if ct.startswith("pc_multiwall_16") or ct.startswith("pc_solid_6"):
        return "pc_multiwall_16"
    elif ct.startswith("pc_"):
        return "pc_multiwall_10"
    return "glass_4mm"


def _effective_price(
    entry: PriceCatalogueEntry | None,
    override: float,
    currency_rate: float,
) -> float:
    """Return unit price: use override if > 0, else catalogue entry, scaled by currency."""
    if override > 0:
        return override
    if entry is not None:
        return entry.price_eur * currency_rate
    return 0.0


def _build_catalogue(params: DomeParameters) -> Dict[str, PriceCatalogueEntry]:
    """Build effective catalogue: defaults → external JSON → price overrides."""
    costing = params.costing
    cat = dict(DEFAULT_PRICE_CATALOGUE)

    # Layer 1: external JSON if provided
    if costing.price_catalogue_path:
        cat = load_price_catalogue(costing.price_catalogue_path, cat)

    return cat


def cost_estimate_for_params(
    dome: TessellatedDome,
    params: DomeParameters,
    catalogue: Dict[str, PriceCatalogueEntry] | None = None,
) -> CostEstimate:
    """Compute the full cost estimate from dome geometry and parameters.

    Uses ``params.costing`` for waste percentages, unit-price overrides,
    currency, labour rates, and overhead.  An explicit *catalogue*
    argument takes precedence over the external-JSON mechanism.
    """
    costing = params.costing
    cat = catalogue or _build_catalogue(params)

    currency = costing.currency if costing.currency in _CURRENCY_RATES else "EUR"
    cur_rate = _CURRENCY_RATES[currency]
    cur_sym = {"EUR": "€", "USD": "$", "GBP": "£"}.get(currency, currency)

    timber_waste = costing.waste_timber_pct / 100.0
    covering_waste = costing.waste_covering_pct / 100.0
    bom: List[BomItem] = []

    # ---- Timber (struts) ----
    timber_key = _timber_key_for_stock(params)
    timber_entry = cat.get(timber_key)
    total_timber_m = sum(s.length for s in dome.struts)
    timber_with_waste = total_timber_m * (1 + timber_waste)

    timber_unit = _effective_price(timber_entry, costing.timber_price_per_m, cur_rate)
    if timber_unit > 0:
        timber_total = timber_with_waste * timber_unit
        bom.append(BomItem(
            category="timber",
            description=timber_entry.name if timber_entry else f"Timber ({timber_key})",
            unit="m",
            quantity=round(timber_with_waste, 2),
            unit_price_eur=round(timber_unit, 2),
            total_eur=round(timber_total, 2),
            waste_pct=costing.waste_timber_pct,
            supplier=timber_entry.supplier if timber_entry else "",
        ))

    # ---- Covering ----
    covering_key = _covering_key(params)
    covering_entry = cat.get(covering_key)
    covering_sheets: List[CoveringSheet] = []
    total_covering_m2 = 0.0

    for panel in dome.panels:
        pts = [dome.nodes[ni] for ni in panel.node_indices]
        area = _polygon_area_3d(pts)
        perim = _panel_perimeter(dome.nodes, panel.node_indices)
        sheet_area = area * (1 + covering_waste)
        total_covering_m2 += sheet_area
        covering_sheets.append(CoveringSheet(
            panel_index=panel.index,
            area_m2=round(area, 4),
            perimeter_m=round(perim, 4),
            sheet_area_m2=round(sheet_area, 4),
            waste_pct=costing.waste_covering_pct,
        ))

    covering_unit = _effective_price(covering_entry, costing.covering_price_per_m2, cur_rate)
    if covering_unit > 0:
        covering_total = total_covering_m2 * covering_unit
        bom.append(BomItem(
            category="covering",
            description=covering_entry.name if covering_entry else "Covering",
            unit="m2",
            quantity=round(total_covering_m2, 2),
            unit_price_eur=round(covering_unit, 2),
            total_eur=round(covering_total, 2),
            waste_pct=costing.waste_covering_pct,
            supplier=covering_entry.supplier if covering_entry else "",
        ))

    # ---- Hardware: bolts, nuts, washers per strut end ----
    # Each strut has 2 ends, each needs 1 bolt + 1 nut + 2 washers
    n_struts = len(dome.struts)
    bolt_count = n_struts * 2
    nut_count = bolt_count
    washer_count = bolt_count * 2

    bolt_key = _bolt_key_for_connector(params)
    nut_k = _nut_key(params)
    washer_k = _washer_key(params)

    bolt_override = costing.bolt_price_each

    for hw_key, hw_qty, hw_desc_suffix in [
        (bolt_key, bolt_count, "bolts"),
        (nut_k, nut_count, "nuts"),
        (washer_k, washer_count, "washers"),
    ]:
        entry = cat.get(hw_key)
        # bolt_price_each override applies only to bolts
        hw_over = bolt_override if hw_desc_suffix == "bolts" else 0.0
        unit_p = _effective_price(entry, hw_over, cur_rate)
        if entry or unit_p > 0:
            bom.append(BomItem(
                category="hardware",
                description=entry.name if entry else hw_key,
                unit="pcs",
                quantity=hw_qty,
                unit_price_eur=round(unit_p, 2),
                total_eur=round(hw_qty * unit_p, 2),
                supplier=entry.supplier if entry else "",
            ))

    # ---- Angle brackets: 1 per node (belt nodes get 2) ----
    belt_height = params.radius_m * (1.0 - 2.0 * params.hemisphere_ratio)
    eps = max(1e-6, params.radius_m * 1e-5)
    bracket_count = 0
    for node in dome.nodes:
        if abs(node[2] - belt_height) <= eps:
            bracket_count += 2  # belt nodes need extra bracing
        else:
            bracket_count += 1

    bracket_entry = cat.get("angle_bracket_90x90")
    if bracket_entry:
        bp = bracket_entry.price_eur * cur_rate
        bom.append(BomItem(
            category="hardware",
            description=bracket_entry.name,
            unit="pcs",
            quantity=bracket_count,
            unit_price_eur=round(bp, 2),
            total_eur=round(bracket_count * bp, 2),
            supplier=bracket_entry.supplier,
        ))

    # ---- Connector plates ----
    plate_entry = cat.get("plate_steel_3mm")
    plate_unit = _effective_price(plate_entry, costing.connector_plate_price_per_m2, cur_rate)
    if plate_unit > 0:
        plate_r_m = (params.node_connector_bolt_offset_m +
                     params.node_connector_bolt_diameter_m * 1.5) / 1000  # to m... wait the offset is already in m
        plate_r_m = params.node_connector_bolt_offset_m + params.node_connector_bolt_diameter_m * 1.5
        plate_area_per = math.pi * plate_r_m ** 2
        total_plate_area = plate_area_per * len(dome.nodes)
        bom.append(BomItem(
            category="connector",
            description=plate_entry.name if plate_entry else "Connector plate",
            unit="m2",
            quantity=round(total_plate_area, 4),
            unit_price_eur=round(plate_unit, 2),
            total_eur=round(total_plate_area * plate_unit, 2),
            supplier=plate_entry.supplier if plate_entry else "",
        ))

    # ---- Gasket ----
    gasket_entry = cat.get("gasket_epdm")
    total_gasket_m = sum(s.length * 2.0 for s in dome.struts)
    gasket_with_waste = total_gasket_m * 1.05  # 5% gasket waste

    gasket_unit = _effective_price(gasket_entry, costing.gasket_price_per_m, cur_rate)
    if gasket_unit > 0:
        bom.append(BomItem(
            category="gasket",
            description=gasket_entry.name if gasket_entry else "Gasket",
            unit="m",
            quantity=round(gasket_with_waste, 2),
            unit_price_eur=round(gasket_unit, 2),
            total_eur=round(gasket_with_waste * gasket_unit, 2),
            waste_pct=5.0,
            supplier=gasket_entry.supplier if gasket_entry else "",
        ))

    # ---- Labour ----
    total_labour = 0.0
    if costing.labor_install_eur_h > 0 and costing.estimated_install_hours > 0:
        install_cost = costing.labor_install_eur_h * costing.estimated_install_hours
        total_labour += install_cost
        bom.append(BomItem(
            category="labour",
            description="Installation labour",
            unit="h",
            quantity=costing.estimated_install_hours,
            unit_price_eur=costing.labor_install_eur_h,
            total_eur=round(install_cost, 2),
        ))
    if costing.labor_cnc_eur_h > 0 and costing.estimated_cnc_hours > 0:
        cnc_cost = costing.labor_cnc_eur_h * costing.estimated_cnc_hours
        total_labour += cnc_cost
        bom.append(BomItem(
            category="labour",
            description="CNC processing",
            unit="h",
            quantity=costing.estimated_cnc_hours,
            unit_price_eur=costing.labor_cnc_eur_h,
            total_eur=round(cnc_cost, 2),
        ))

    # ---- Totals ----
    total_material = sum(
        b.total_eur for b in bom if b.category in ("timber", "covering", "connector")
    )
    total_hardware = sum(
        b.total_eur for b in bom if b.category in ("hardware", "gasket")
    )
    subtotal = sum(b.total_eur for b in bom)

    # Overhead markup
    overhead_pct = max(0.0, costing.overhead_pct)
    total_overhead = subtotal * overhead_pct / 100.0 if overhead_pct > 0 else 0.0
    grand_total = subtotal + total_overhead

    net_area = sum(cs.area_m2 for cs in covering_sheets)
    waste_coeff = total_covering_m2 / max(net_area, 1e-12) if net_area > 0 else 1.0

    return CostEstimate(
        bom=bom,
        covering_sheets=covering_sheets,
        total_material_eur=round(total_material, 2),
        total_hardware_eur=round(total_hardware, 2),
        total_labour_eur=round(total_labour, 2),
        total_overhead_eur=round(total_overhead, 2),
        total_eur=round(grand_total, 2),
        waste_coefficient=round(waste_coeff, 4),
        timber_total_m=round(timber_with_waste, 3),
        covering_total_m2=round(total_covering_m2, 3),
        bolt_count=bolt_count,
        gasket_total_m=round(gasket_with_waste, 3),
        currency=currency,
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_cost_report(estimate: CostEstimate, params: DomeParameters, path: Path) -> None:
    """Write cost estimate as JSON."""
    report = {
        "summary": {
            "currency": estimate.currency,
            "total_eur": estimate.total_eur,
            "total_material_eur": estimate.total_material_eur,
            "total_hardware_eur": estimate.total_hardware_eur,
            "total_labour_eur": estimate.total_labour_eur,
            "total_overhead_eur": estimate.total_overhead_eur,
            "timber_total_m": estimate.timber_total_m,
            "covering_total_m2": estimate.covering_total_m2,
            "bolt_count": estimate.bolt_count,
            "gasket_total_m": estimate.gasket_total_m,
            "waste_coefficient": estimate.waste_coefficient,
        },
        "bom": [
            {
                "category": item.category,
                "description": item.description,
                "unit": item.unit,
                "quantity": item.quantity,
                "unit_price_eur": item.unit_price_eur,
                "total_eur": item.total_eur,
                "waste_pct": item.waste_pct,
                "supplier": item.supplier,
            }
            for item in estimate.bom
        ],
        "covering_sheets": [
            {
                "panel_index": cs.panel_index,
                "area_m2": cs.area_m2,
                "perimeter_m": cs.perimeter_m,
                "sheet_area_m2": cs.sheet_area_m2,
                "waste_pct": cs.waste_pct,
            }
            for cs in estimate.covering_sheets
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Cost report written to %s", path)


def write_cost_csv(estimate: CostEstimate, path: Path) -> None:
    """Write BOM as CSV."""
    cur = estimate.currency
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Category", "Description", "Supplier", "Unit", "Quantity",
            f"Unit Price ({cur})", f"Total ({cur})", "Waste %",
        ])
        for item in estimate.bom:
            writer.writerow([
                item.category,
                item.description,
                item.supplier,
                item.unit,
                f"{item.quantity:.2f}",
                f"{item.unit_price_eur:.2f}",
                f"{item.total_eur:.2f}",
                f"{item.waste_pct:.1f}",
            ])
        # Totals rows
        writer.writerow([])
        writer.writerow(["", "TOTAL Material", "", "", "", "", f"{estimate.total_material_eur:.2f}", ""])
        writer.writerow(["", "TOTAL Hardware", "", "", "", "", f"{estimate.total_hardware_eur:.2f}", ""])
        if estimate.total_labour_eur > 0:
            writer.writerow(["", "TOTAL Labour", "", "", "", "", f"{estimate.total_labour_eur:.2f}", ""])
        if estimate.total_overhead_eur > 0:
            writer.writerow(["", f"Overhead ({cur})", "", "", "", "", f"{estimate.total_overhead_eur:.2f}", ""])
        writer.writerow(["", "GRAND TOTAL", "", "", "", "", f"{estimate.total_eur:.2f}", ""])
    logging.info("Cost CSV written to %s", path)
