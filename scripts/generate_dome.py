#!/usr/bin/env python3
"""Headless entry point for the geodesic dome generator.

This script is a thin CLI wrapper around the ``freecad_dome.pipeline``
module.  All generation logic lives in composable pipeline steps.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import importlib
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freecad_dome import parameters

# Reload modules only inside FreeCAD's persistent interpreter where stale
# bytecode from a previous macro run may linger.
if "FreeCAD" in sys.modules:
    importlib.reload(parameters)
    # Reload all sub-modules that may have been cached.
    from freecad_dome import (
        icosahedron, tessellation, struts, panels,
        spreadsheets, base_wall, entry_porch, door_opening,
        export, pipeline,
    )
    for _mod in (
        icosahedron, tessellation, struts, panels,
        spreadsheets, base_wall, entry_porch, door_opening,
        export, pipeline,
    ):
        importlib.reload(_mod)

from freecad_dome.pipeline import DomePipeline, PipelineContext

load_parameters = parameters.load_parameters
parse_cli_overrides = parameters.parse_cli_overrides
prompt_parameters_dialog = parameters.prompt_parameters_dialog


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main() -> None:
    configure_logging()
    overrides, cli = parse_cli_overrides(_sanitized_args())
    config_path = _resolve_config_path(cli.config)
    params = load_parameters(config_path, overrides)
    params = _maybe_prompt_for_parameters(params, cli)
    logging.info(
        "Parameters: radius=%.3fm freq=%d material=%s",
        params.radius_m,
        params.frequency,
        params.material,
    )

    ctx = PipelineContext(
        params=params,
        out_dir=Path(cli.out_dir),
        skip_ifc=cli.skip_ifc,
        skip_stl=cli.skip_stl,
        skip_dxf=cli.skip_dxf,
        manifest_name=cli.manifest_name,
        ifc_name=cli.ifc_name,
        stl_name=cli.stl_name,
        dxf_name=cli.dxf_name,
    )

    pipeline = DomePipeline()
    pipeline.run(ctx)


def _sanitized_args() -> List[str]:
    raw = sys.argv[1:]
    filtered: List[str] = []
    for arg in raw:
        if arg in {"--single-instance", "--"}:
            continue
        if arg == "-":
            continue
        filtered.append(arg)
    return filtered


def _default_config_path() -> str | None:
    candidate = REPO_ROOT / "configs" / "base.json"
    if candidate.exists():
        return str(candidate)
    return None


def _resolve_config_path(cli_config: str | None) -> str:
    if cli_config:
        path = Path(cli_config)
        if path.exists():
            return str(path)
        logging.warning("Config file %s not found; trying project default", path)
    default = _default_config_path()
    if default is not None:
        return default
    logging.error("No configuration file available")
    sys.exit(1)


def _maybe_prompt_for_parameters(params, cli):
    if getattr(cli, "no_gui", False):
        return params
    updated = prompt_parameters_dialog(params)
    if updated is None:
        logging.info("Parameter dialog canceled; aborting generation")
        sys.exit(0)
    return updated


if __name__ == "__main__":
    main()
