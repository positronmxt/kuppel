# FreeCAD Geodesic Dome Toolkit

Parametric tooling for generating geodesic dome frame members inside FreeCAD 1.0.2. The initial target is a wooden dome based on an icosahedral subdivision with beveled struts and headless automation support.

## Goals

- Generate FreeCAD documents that contain properly oriented struts with bevel-ready end cuts.
- Generate matching polygon panels so every edge is backed by a strut.
- Support configurable dome parameters (radius, frequency, truncation ratio, hemisphere clip).
- Represent each structural element as data suitable for IFC export (`IfcMember`).
- Allow layered configuration: JSON (primary), CLI overrides, optional FreeCAD spreadsheet edits.
- Prepare for later STL/DXF fabrication exports alongside IFC.

## Baseline Parameters

| Property | Default | Notes |
| --- | --- | --- |
| Radius | 3.0 m | Controls dome size. |
| Frequency | V4 | Icosahedral subdivision order. |
| Truncation Ratio | 0.18 | Shortens the base for entrance/stand height control. |
| Dome Segment | 0.625 hemisphere | Percentage of sphere kept. |
| Stock Cross-section | 50 mm × 50 mm | Adjustable per project. |
| Kerf | 2 mm | Used when sizing bevel cuts. |
| Clearance/Tolerance | 3 mm | To offset interference. |

All parameters must remain editable via JSON config, CLI flags, or spreadsheet values.

## Structure

```text
freecad_dome/
  __init__.py
  parameters.py          # configuration loading + VarSet propagation
  icosahedron.py         # base geometry + truncation utilities
  tessellation.py        # frequency subdivision + node/edge data
  struts.py              # Arch/Part pipelines for beveled members
  panels.py              # panel face creation and stats
scripts/
  generate_dome.py       # headless entry point for FreeCAD 1.0.2
```

## Roadmap

1. Implement configuration stack with JSON → CLI → spreadsheet precedence.
2. Add geometric primitives (icosahedron, truncation, subdivision, segment clipping).
3. Generate Arch Structures for each strut, apply Part-based bevels respecting stock/kerf/clearance.
4. Tag each strut with stable identifiers and metadata needed for `IfcMember` export.
5. Implement export helpers (IFC first, then STL/DXF).

## Configuration layers

- **JSON (primary)** — see `configs/base.json` for the canonical defaults that align with the current requirements (3 m radius, V4, truncation 0.18, hemisphere 0.625, 50×50 mm timber, kerf 2 mm, clearance 3 mm).
- **Full icosahedron preset** — use `configs/icosahedron_full.json` to keep frequency 1, disable truncation, and set the hemisphere ratio to `1.0` so the entire sphere is generated before any clipping or truncation tweaks.
- **Testing variant** — `configs/testing_v3.json` lowers the frequency to V3 and disables bevels for lighter-weight experiments.
- **CLI overrides** — `scripts/generate_dome.py` accepts flags such as `--radius`, `--frequency`, `--stock-size WIDTH HEIGHT`, and export controls (`--out-dir`, `--skip-ifc`, `--stl-name`...).
- **Joint style toggles** — pass `--no-bevels` to disable beveled ends (use straight prisms) when boolean ops struggle.
- **Truncation toggle** — pass `--no-truncation` to keep the full icosahedron instead of slicing the base.
- **Panels-only runs** — pass `--panels-only` when you only want panel faces (no strut solids, manifests, or exports). Useful while panel topology is being inspected or when strut booleans misbehave.
- **Interactive dialog** — when PySide is available the script now opens a small GUI form before generation so you can tweak the core parameters. Use `--no-gui` to keep existing headless behaviour.
- **Spreadsheet / VarSet** — `freecad_dome.parameters.write_varset` can push the current parameter set into an `App::VarSet` or Spreadsheet object so FreeCAD GUI edits feed back into the script workflow.

## Diagnostics

Each run now emits a validation report immediately after tessellation: node valence distribution, maximum radius deviation, strut length families, panel area statistics, and warnings when a panel edge has no matching strut. Use this log to confirm every vertex sees the expected number of members before the FreeCAD solids are created.

Example headless run:

```bash
freecadcmd scripts/generate_dome.py \
  --config configs/base.json \
  --radius 3.5 \
  --stock-size 0.045 0.075 \
  --no-gui \
  --out-dir exports/run1
```

Full-sphere sanity check (no truncation, full icosahedron):

```bash
freecadcmd scripts/generate_dome.py \
  --config configs/icosahedron_full.json \
  --no-gui \
  --out-dir exports/icosahedron_full
```

Outputs (when FreeCAD modules are available):

- `exports/run1/dome_manifest.json` — per-strut metadata (length, material, grouping, IFC GUID).
- `exports/run1/dome.ifc` — each strut exported as `IfcMember`.
- `exports/run1/dome.stl` — mesh export of the structural members.
- `exports/run1/dome.dxf` — 2D projection suitable for fabrication drawings.
- FreeCAD document now also contains `Panel_*` objects: polygon faces (pentagons/hexagons) sized to match each tessellated cell, useful for checking sheathing layouts.

Panels-only runs still build the FreeCAD document with `Panel_*` objects but skip the strut manifest and IFC/STL/DXF exports, so you can iterate on surface coverage without waiting for boolean-heavy strut geometry.

## IFC export note

IFC export requires the FreeCAD BIM/IFC workbench. Depending on the FreeCAD distribution, the exporter may be exposed as a legacy top-level module `importIFC`, or under BIM's `importers/`.

Some builds also require IfcOpenShell (`ifcopenshell`) to be installed before IFC export works.

Quick check:

```bash
freecadcmd -c "import importIFC"
```

On the FreeCAD snap the CLI command is `freecad.cmd` and you can install Python deps via `freecad.pip`:

```bash
snap run freecad.cmd -c "import FreeCAD; print(FreeCAD.Version())"
snap run freecad.cmd -c "import importers.exportIFC"
snap run freecad.pip install ifcopenshell
```

If you're using an AppImage, run its bundled CLI explicitly:

```bash
APP=/path/to/FreeCAD*.AppImage
chmod +x "$APP"
"$APP" --appimage-extract-and-run freecadcmd -c "import importIFC"
"$APP" --appimage-extract-and-run freecadcmd scripts/generate_dome.py --config configs/base.json --no-gui
```

All paths and export toggles can be controlled through CLI flags.

## Testing

Pure-Python geometry helpers have unit tests under `tests/`. Run them via:

```bash
python -m pytest
```

The tests cover truncation growth, tessellation density by frequency, and hemisphere clipping behaviour.

## Usage (planned)

```bash
freecadcmd scripts/generate_dome.py \
  --config configs/base.json \
  --radius 3.5 \
  --stock-size 0.045 0.075
```

- **JSON**: primary configuration file describing default parameters and material catalog.
- **CLI flags**: runtime overrides for automation pipelines.
- **Spreadsheet**: optional FreeCAD GUI override sheet for manual tweaks.

Documentation will be updated as modules land.
