# Geodesic Dome Generator — User Documentation

Geodesic dome frame generator for FreeCAD 1.0+.  Generates tessellated
geodesic dome geometry with struts, panels, node connectors, covering,
ventilation, foundation layout, load calculations, production drawings,
weather protection details, and cost estimates.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [First Greenhouse Project](#first-greenhouse-project)
4. [Parameter Reference](#parameter-reference)
5. [CLI Reference](#cli-reference)
6. [FreeCAD Integration](#freecad-integration)
7. [Configuration Files](#configuration-files)
8. [Pipeline Architecture](#pipeline-architecture)
9. [Gallery](#gallery)
10. [FAQ](#faq)

---

## Installation

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **FreeCAD 1.0+** (for solid body generation and IFC export)
- **Git** (to clone the repository)

### Linux (Ubuntu/Debian)

```bash
# Install FreeCAD
sudo apt install freecad

# Clone the repository
git clone https://github.com/<your-org>/geodesic_1.git
cd geodesic_1

# Verify — run unit tests (no FreeCAD needed)
python3 -m pytest tests/test_geometry.py -v
```

### macOS

```bash
# Install FreeCAD via Homebrew
brew install --cask freecad

# Clone and test
git clone https://github.com/<your-org>/geodesic_1.git
cd geodesic_1
python3 -m pytest tests/test_geometry.py -v
```

### Windows

1. Download and install [FreeCAD](https://www.freecad.org/downloads.php)
2. Clone the repository: `git clone https://github.com/<your-org>/geodesic_1.git`
3. Open a terminal in the project directory
4. Run: `python -m pytest tests/test_geometry.py -v`

### Optional Dependencies

```bash
# For IFC validation
pip install ifcopenshell

# For linting
pip install ruff
```

---

## Quick Start

### Generate a dome from the command line

```bash
# Default 3m radius, frequency 4 dome
python3 scripts/generate_dome.py

# Custom dome with ventilation and node connectors
python3 scripts/generate_dome.py \
  --radius 4.0 \
  --frequency 3 \
  --ventilation \
  --node-connectors \
  --out-dir exports/my_dome
```

### Generate from a JSON config file

```bash
python3 scripts/generate_dome.py --config configs/base.json
```

### Run inside FreeCAD

1. Open FreeCAD
2. Go to **Macro → Execute Macro**
3. Browse to `macros/GenerateDomeFromRepo.FCMacro`
4. A parameter dialog will appear — adjust settings and click **Generate**

---

## First Greenhouse Project

This tutorial walks you through designing a 4-metre greenhouse dome.

### Step 1: Define geometry

```bash
python3 scripts/generate_dome.py \
  --radius 4.0 \
  --frequency 3 \
  --segment 0.625
```

- **radius**: 4 metres gives ~50 m² floor area
- **frequency**: 3 gives a good balance of panel size and structural detail
- **segment**: 0.625 keeps a practical wall height at the belt

### Step 2: Add polycarbonate covering

```bash
python3 scripts/generate_dome.py \
  --radius 4.0 --frequency 3 --segment 0.625 \
  --covering-type polycarbonate_twin_10 \
  --covering-profile H_profile_alu
```

### Step 3: Add ventilation

Greenhouses need 15–25 % of floor area as operable vents:

```bash
python3 scripts/generate_dome.py \
  --radius 4.0 --frequency 3 --segment 0.625 \
  --covering-type polycarbonate_twin_10 \
  --ventilation \
  --ventilation-target 0.20
```

### Step 4: Add foundation and entry

```bash
python3 scripts/generate_dome.py \
  --radius 4.0 --frequency 3 --segment 0.625 \
  --covering-type polycarbonate_twin_10 \
  --ventilation --ventilation-target 0.20 \
  --entry-porch --porch-depth 0.5 \
  --foundation --foundation-type strip \
  --node-connectors
```

### Step 5: Get production data and cost estimate

```bash
python3 scripts/generate_dome.py \
  --radius 4.0 --frequency 3 --segment 0.625 \
  --covering-type polycarbonate_twin_10 \
  --ventilation --ventilation-target 0.20 \
  --entry-porch --porch-depth 0.5 \
  --foundation --foundation-type strip \
  --node-connectors \
  --production --costing --weather \
  --loads --snow-zone III --wind-speed 25 \
  --out-dir exports/greenhouse_4m
```

The `exports/greenhouse_4m/` folder will contain:
- `dome_manifest.json` — full strut/panel listing
- Production cut-lists as CSV
- Node connector plate DXF outlines
- Cost estimate CSV
- Weather sealing report
- Load calculation report

---

## Parameter Reference

Parameters are organized into 7 configuration groups.
All parameters are accessible as flat CLI flags or as nested JSON config.

### GeometryConfig — Sphere shape

| Parameter | Default | Description |
|-----------|---------|-------------|
| `radius_m` | 3.0 | Sphere radius in metres |
| `frequency` | 4 | Icosahedral subdivision order (1–6) |
| `truncation_ratio` | 0.18 | Vertex truncation factor (0–1) |
| `hemisphere_ratio` | 0.625 | Portion of sphere kept (0–1] |

### StructureConfig — Materials & construction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stock_width_m` | 0.05 | Strut stock width (m) |
| `stock_height_m` | 0.05 | Strut stock height (m) |
| `kerf_m` | 0.002 | Saw kerf allowance (m) |
| `clearance_m` | 0.003 | Joint clearance (m) |
| `material` | "wood" | Material key: wood / aluminum / steel |
| `use_bevels` | true | Enable beveled end cuts |
| `use_truncation` | true | Enable vertex truncation |
| `panels_only` | false | Skip strut generation |
| `generate_struts` | true | Generate strut solids |
| `generate_belt_cap` | false | Close bottom with planar panel |
| `node_fit_plane_mode` | "radial" | End-cut plane: "radial", "axis", or "miter" |
| `node_fit_use_separation_planes` | true | Use separation planes between struts |
| `node_fit_extension_m` | 0.005 | Extend strut past node for tight fit (m) |
| `node_fit_mode` | "planar" | Node-fit algorithm: "planar", "tapered", or "voronoi" |
| `node_fit_taper_ratio` | 0.6 | End cross-section ratio for tapered mode (0–1) |
| `split_struts_per_panel` | false | Split struts per adjacent panel |
| `min_strut_length_factor` | 0.5 | Min strut length as fraction of stock width |
| `prism_only_length_factor` | 3.0 | Below this × stock width, skip bevels |
| `cap_length_factor` | 2.0 | End-cap length as × stock size |
| `max_cap_ratio` | 0.45 | Max cap length as fraction of strut |
| `split_keep_offset_factor` | 0.35 | Longitudinal split keep-offset factor |
| `min_wedge_angle_deg` | 15.0 | Min angle between separation planes (°) |
| `bevel_fillet_radius_m` | 0.0 | Fillet radius on bevel edges (m, 0=off) |
| `cap_blend_mode` | "sharp" | Cap/body junction: sharp / chamfer / fillet |
| `strut_profile` | "rectangular" | Cross-section: rectangular / round / trapezoidal |
| `connector_strut_inset` | true | Shorten struts for connector plate thickness |
| `generate_node_connectors` | false | Generate node connector hubs |
| `node_connector_type` | \"plate\" | Connector type: plate / ball / pipe / lapjoint |
| `node_connector_thickness_m` | 0.006 | Plate thickness (m) |
| `node_connector_bolt_diameter_m` | 0.010 | Bolt diameter (m, e.g. M10) |
| `node_connector_bolt_length_m` | 0.060 | Bolt length (m) |
| `node_connector_washer_diameter_m` | 0.020 | Washer diameter (m) |
| `node_connector_bolt_offset_m` | 0.025 | Bolt offset from center (m) |
| `node_connector_lap_extension_m` | 0.03 | Lap joint strut extension past node (m) |

### CoveringConfig — Panels & glazing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `generate_panel_faces` | true | Generate panel surface solids |
| `generate_panel_frames` | false | Generate inset panel frames |
| `panel_frame_inset_m` | 0.0 | Frame inset from panel edge (m) |
| `panel_frame_profile_width_m` | 0.04 | Frame profile width (m) |
| `panel_frame_profile_height_m` | 0.015 | Frame profile height (m) |
| `glass_thickness_m` | 0.0 | Glass thickness (m, 0 = no glass) |
| `glass_gap_m` | 0.01 | Glass panel gap (m) |
| `covering_type` | "glass" | Covering material key |
| `covering_thickness_m` | 0.0 | Override covering thickness (m) |
| `covering_gap_m` | 0.0 | Edge gap override (m) |
| `covering_delta_t_k` | 40.0 | Thermal swing for expansion (K) |
| `covering_profile_type` | "none" | Attachment profile key |
| `generate_weather` | false | Generate weather protection data |
| `gasket_type` | "epdm_d_10x8" | Gasket profile type |

### OpeningsConfig — Doors, porches & ventilation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `generate_base_wall` | false | Generate cylindrical knee wall |
| `base_wall_height_m` | 2.1 | Wall height (m) |
| `base_wall_thickness_m` | 0.15 | Wall thickness (m) |
| `door_width_m` | 0.9 | Door opening width (m) |
| `door_height_m` | 2.1 | Door opening height (m) |
| `door_angle_deg` | 0.0 | Door azimuth angle (deg, 0 = +X) |
| `auto_door_angle` | false | Auto-select door angle |
| `door_clearance_m` | 0.01 | Door cutout clearance (m) |
| `generate_entry_porch` | false | Generate framed entry porch |
| `porch_depth_m` | 0.5 | Porch projection depth (m, max 0.5) |
| `porch_width_m` | 1.2 | Porch width (m) |
| `porch_height_m` | 2.1 | Porch height (m) |
| `porch_member_size_m` | 0.045 | Porch frame member size (m) |
| `porch_glass_thickness_m` | 0.006 | Porch glazing thickness (m) |
| `generate_ventilation` | false | Generate ventilation plan |
| `ventilation_mode` | "auto" | Vent mode: auto / apex / ring / manual |
| `ventilation_target_ratio` | 0.20 | Target vent area / floor area |
| `ventilation_apex_count` | 1 | Number of apex vent panels |
| `ventilation_ring_count` | 6 | Number of ring vent panels |
| `ventilation_ring_height_ratio` | 0.5 | Ring height (0=belt, 1=apex) |
| `ventilation_panel_indices` | [] | Manual vent panel indices |

### FoundationConfig — Foundation system

| Parameter | Default | Description |
|-----------|---------|-------------|
| `generate_foundation` | false | Generate foundation layout |
| `foundation_type` | "strip" | Type: strip / point / screw_anchor |
| `foundation_bolt_diameter_m` | 0.016 | Anchor bolt diameter (M16) |
| `foundation_bolt_embed_m` | 0.20 | Bolt embed depth (m) |
| `foundation_bolt_protrusion_m` | 0.10 | Bolt protrusion above concrete (m) |
| `foundation_strip_width_m` | 0.30 | Strip footing width (m) |
| `foundation_strip_depth_m` | 0.40 | Strip depth below grade (m) |
| `foundation_pier_diameter_m` | 0.30 | Point pier diameter (m) |
| `foundation_pier_depth_m` | 0.60 | Point pier depth (m) |

### ExportConfig — Analysis & reporting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `generate_spreadsheets` | false | Generate FreeCAD spreadsheets |
| `generate_production` | false | Generate production data |
| `generate_loads` | false | Compute structural loads |
| `load_wind_speed_ms` | 25.0 | Reference wind speed (m/s) |
| `load_wind_terrain` | "II" | Terrain category: 0/I/II/III/IV |
| `load_wind_direction_deg` | 0.0 | Wind azimuth (deg) |
| `load_snow_zone` | "III" | Snow zone: I through V |
| `load_snow_exposure` | 1.0 | Snow exposure coefficient Ce |
| `load_snow_thermal` | 1.0 | Snow thermal coefficient Ct |

### CostingConfig — Cost estimation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `generate_costing` | false | Generate cost estimate and BOM |

---

## CLI Reference

```
python3 scripts/generate_dome.py [OPTIONS]
```

### General options

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to JSON configuration file |
| `--out-dir DIR` | Export directory (default: exports) |
| `--no-gui` | Skip the FreeCAD parameter dialog |

### Geometry

| Flag | Description |
|------|-------------|
| `--radius FLOAT` | Sphere radius in metres |
| `--frequency INT` | Subdivision order |
| `--truncation FLOAT` | Truncation ratio (0–1) |
| `--segment FLOAT` | Hemisphere ratio (0–1] |

### Structure

| Flag | Description |
|------|-------------|
| `--stock-size W H` | Stock width and height in metres |
| `--kerf FLOAT` | Saw kerf in metres |
| `--clearance FLOAT` | Clearance offset in metres |
| `--material {wood,aluminum,steel}` | Material selection |
| `--no-bevels` | Disable beveled cuts |
| `--no-truncation` | Skip truncation step |

### Panels & covering

| Flag | Description |
|------|-------------|
| `--panels-only` | Generate panels without strut solids |
| `--panel-frames` | Generate inset frames for panels |
| `--glass-thickness FLOAT` | Glass panel thickness in metres |
| `--glass-gap FLOAT` | Glass panel edge gap in metres |
| `--covering-type KEY` | Covering material (e.g. polycarbonate_twin_10) |
| `--covering-profile KEY` | Attachment profile (e.g. H_profile_alu) |

### Openings

| Flag | Description |
|------|-------------|
| `--base-wall` | Generate cylindrical knee wall |
| `--entry-porch` | Generate framed entry porch |
| `--door W H` | Door dimensions in metres |
| `--auto-door-angle` | Auto-select door angle |
| `--ventilation` | Generate ventilation plan |
| `--ventilation-mode {auto,apex,ring,manual}` | Vent placement strategy |
| `--ventilation-target FLOAT` | Target vent ratio (0–1) |

### Analysis & export

| Flag | Description |
|------|-------------|
| `--node-connectors` | Generate node connector hubs |
| `--foundation` | Generate foundation layout |
| `--foundation-type {strip,point,screw_anchor}` | Foundation type |
| `--loads` | Compute structural loads |
| `--wind-speed FLOAT` | Reference wind speed (m/s) |
| `--snow-zone {I,II,III,IV,V}` | Snow load zone |
| `--production` | Generate cut-lists and assembly plans |
| `--costing` | Generate cost estimate and BOM |
| `--weather` | Generate weather protection data |
| `--spreadsheets` | Generate FreeCAD spreadsheets |

---

## FreeCAD Integration

### Using the FreeCAD Macro

1. Open FreeCAD 1.0+
2. Go to **Macro → Macros...** or press **Alt+F8**
3. Navigate to the project's `macros/` directory
4. Select `GenerateDomeFromRepo.FCMacro` and click **Execute**
5. A dialog appears with all parameter groups
6. Adjust parameters and click **Generate**
7. The dome model appears in the active document

### Headless generation with freecadcmd

```bash
freecadcmd scripts/generate_dome.py \
  --config configs/base.json \
  --no-gui \
  --out-dir exports/batch_output
```

### Exporting to IFC

The pipeline automatically exports IFC when FreeCAD is available:

```bash
freecadcmd scripts/generate_dome.py \
  --config configs/base.json \
  --no-gui \
  --ifc-name my_dome.ifc
```

Use `--skip-ifc`, `--skip-stl`, or `--skip-dxf` to disable specific exports.

---

## Configuration Files

### Flat format (legacy)

```json
{
  "radius_m": 4.0,
  "frequency": 3,
  "hemisphere_ratio": 0.625,
  "material": "wood",
  "stock_width_m": 0.05,
  "stock_height_m": 0.05,
  "covering_type": "polycarbonate_twin_10",
  "generate_ventilation": true
}
```

### Nested format (recommended)

```json
{
  "geometry": {
    "radius_m": 4.0,
    "frequency": 3,
    "hemisphere_ratio": 0.625
  },
  "structure": {
    "material": "wood",
    "stock_width_m": 0.05,
    "stock_height_m": 0.05
  },
  "covering": {
    "covering_type": "polycarbonate_twin_10"
  },
  "openings": {
    "generate_ventilation": true,
    "ventilation_target_ratio": 0.20
  },
  "foundation": {
    "generate_foundation": true,
    "foundation_type": "strip"
  }
}
```

Both formats are fully supported. Flat JSON files are automatically migrated
at load time. CLI overrides layer on top of any JSON configuration.

### Precedence chain

1. **JSON file** (lowest) — project configuration
2. **CLI overrides** — runtime tweaks
3. **Spreadsheet overrides** — GUI editing inside FreeCAD (highest)

---

## Pipeline Architecture

The dome generator uses a composable pipeline of 21 steps:

```
TessellationStep          → Build geodesic mesh
AutoDoorAngleStep         → Compute optimal door angle
EntryPorchStep            → Generate entry porch frame
BaseWallStep              → Generate cylindrical knee wall
StrutGenerationStep       → Create strut solids + node-fit data
NodeConnectorStep         → Generate node hub connectors
StrutBoltHoleStep         → Drill bolt holes in strut ends
PanelGenerationStep       → Create triangular panels
GlassPanelStep            → Add glass/covering to panels
DoorOpeningStep           → Cut door opening in wall/panels
VentilationStep           → Mark vent panels + metadata
CoveringReportStep        → Material report for covering
FoundationStep            → Anchor bolt layout + pour plan
LoadCalculationStep       → Dead/snow/wind loads (Eurocode)
ProductionDrawingsStep    → Cut-lists, saw table, plates
WeatherProtectionStep     → Gaskets, drainage, eave details
CostEstimationStep        → Full BOM and cost estimate
SpreadsheetStep           → FreeCAD spreadsheet population
PanelAccuracyReportStep   → Validate panel geometry
ManifestExportStep        → Write JSON manifest
ModelExportStep           → Export IFC/STL/DXF
```

### Customizing the pipeline

```python
from freecad_dome.pipeline import DomePipeline, default_steps

pipe = DomePipeline()
pipe.remove("glass_panels")           # Skip glass generation
pipe.insert_after("struts", MyStep())  # Add a custom step
pipe.run(ctx)
```

---

## Gallery

### Example outputs

| Configuration | Description |
|---------------|-------------|
| `configs/base.json` | Default 3m dome with wood struts |
| `configs/icosahedron_full.json` | Full icosahedron (no truncation) |
| `--radius 4.0 --frequency 3 --covering-type polycarbonate_twin_10` | 4m greenhouse |

### Exported files

| File | Format | Content |
|------|--------|---------|
| `dome_manifest.json` | JSON | Complete strut/panel listing with dimensions |
| `dome.ifc` | IFC 2x3 | Full 3D BIM model |
| `dome.stl` | STL | Triangulated mesh for 3D printing |
| `dome.dxf` | DXF | 2D projection for CNC/laser cutting |

---

## FAQ

### Q: Can I use polycarbonate instead of glass?

Yes. Set `--covering-type polycarbonate_twin_10` (or `_twin_8`, `_twin_16`,
`_solid_4`, `_solid_6`). The generator automatically adjusts thermal expansion
gaps, attachment profiles, and condensation drainage specs.

### Q: How do I calculate the right number of vent panels?

Use `--ventilation --ventilation-target 0.20` for a 20% vent ratio (recommended
for greenhouses). The auto mode distributes vent panels between apex (exhaust)
and mid-ring (intake) positions for natural stack-effect ventilation.

### Q: What foundation type should I use?

- **strip**: Continuous strip footing — best for permanent structures
- **point**: Isolated piers — good for lighter structures or uneven ground
- **screw_anchor**: Screw piles — minimal excavation, easy to install/remove

### Q: Can I change the pipeline step order?

Yes. The pipeline is fully composable:
```python
pipe = DomePipeline()
pipe.remove("step_name")                    # Remove a step
pipe.insert_before("reference", new_step)   # Insert before
pipe.insert_after("reference", new_step)    # Insert after
pipe.replace("old_step", new_step)          # Replace
```

### Q: How do I run tests?

```bash
# Unit tests (no FreeCAD needed)
python3 -m pytest tests/test_geometry.py -v

# All tests including integration (FreeCAD/IFC tests auto-skip if unavailable)
python3 -m pytest tests/ -v

# With coverage
python3 -m pytest tests/ --cov=freecad_dome --cov-report=html
```

### Q: What Eurocode standards are used for load calculations?

- **Wind**: EN 1991-1-4 (wind actions on structures)
- **Snow**: EN 1991-1-3 (snow loads) with Estonian national annex zones I–V
- Terrain categories follow the standard 0/I/II/III/IV classification

### Q: Is the IFC export compliant?

The generator produces IFC 2x3 files using FreeCAD's built-in IFC exporter.
Materials are tagged with proper IfcMaterial entities. For strict BIM compliance,
install `ifcopenshell` and run the validation tests:
```bash
pip install ifcopenshell
python3 -m pytest tests/test_integration.py::TestIFCValidation -v
```
