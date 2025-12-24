#!/usr/bin/env python3
"""Generate TechDraw pages for representative parts in the active FreeCAD document.

Usage (inside FreeCAD Python / FreeCADCmd):

  import runpy
  runpy.run_path('scripts/generate_techdraw.py', run_name='__main__')

Env vars (optional):
- TD_TEMPLATE: SVG template path
- TD_OUT_DIR: export directory
- TD_EXPORT_PDF: 1/0
- TD_EXPORT_DXF: 1/0
"""

from __future__ import annotations

import argparse
import os
import sys

from freecad_dome.techdraw import TechDrawOptions, generate_representative_techdraw


def _truthy(text: str | None) -> bool:
    if text is None:
        return False
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--template", default=os.environ.get("TD_TEMPLATE", ""))
    p.add_argument("--out-dir", default=os.environ.get("TD_OUT_DIR", ""))
    p.add_argument("--page-prefix", default="TD_")
    p.add_argument("--scale", type=float, default=0.5, help="View scale (1:2 => 0.5)")

    p.add_argument("--struts", dest="include_struts", action="store_true", default=True)
    p.add_argument("--no-struts", dest="include_struts", action="store_false")

    p.add_argument("--panel-frames", dest="include_panel_frames", action="store_true", default=True)
    p.add_argument("--no-panel-frames", dest="include_panel_frames", action="store_false")

    p.add_argument("--glass", dest="include_glass_panels", action="store_true", default=True)
    p.add_argument("--no-glass", dest="include_glass_panels", action="store_false")

    p.add_argument("--panel-faces", dest="include_panel_faces", action="store_true", default=False)
    p.add_argument("--no-panel-faces", dest="include_panel_faces", action="store_false")

    p.add_argument(
        "--export-pdf",
        dest="export_pdf",
        action="store_true",
        default=_truthy(os.environ.get("TD_EXPORT_PDF")),
    )
    p.add_argument("--no-export-pdf", dest="export_pdf", action="store_false")

    p.add_argument(
        "--export-dxf",
        dest="export_dxf",
        action="store_true",
        default=_truthy(os.environ.get("TD_EXPORT_DXF")),
    )
    p.add_argument("--no-export-dxf", dest="export_dxf", action="store_false")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])

    try:
        import FreeCAD  # type: ignore
    except Exception as exc:
        print(f"FreeCAD import failed: {exc}")
        return 2

    doc = FreeCAD.ActiveDocument
    if doc is None:
        print("No ActiveDocument; open or generate a dome first")
        return 2

    opts = TechDrawOptions(
        template_path=str(args.template or ""),
        page_prefix=str(args.page_prefix or "TD_"),
        view_scale=float(args.scale),
        include_struts=bool(args.include_struts),
        include_panel_frames=bool(args.include_panel_frames),
        include_glass_panels=bool(args.include_glass_panels),
        include_panel_faces=bool(args.include_panel_faces),
        export_pdf=bool(args.export_pdf),
        export_dxf=bool(args.export_dxf),
        export_dir=str(args.out_dir or ""),
    )

    res = generate_representative_techdraw(doc, opts)

    try:
        FreeCAD.Console.PrintMessage(
            "[generate_techdraw] "
            f"pages={res.get('created_pages')} pdf={res.get('exported_pdf')} dxf={res.get('exported_dxf')}\n"
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
