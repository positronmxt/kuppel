#!/usr/bin/env python3
"""Debug TechDraw view directions in the active FreeCAD document.

Run inside FreeCAD (GUI or FreeCADCmd):

  import runpy
  runpy.run_path('scripts/debug_techdraw_views.py', run_name='__main__')

This prints each TechDraw page and its views' Direction/XDirection (and other
handy properties) to the FreeCAD console / stdout.
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--page-prefix", default="TD_", help="Only pages whose Name/Label starts with this")
    p.add_argument("--show-properties", action="store_true", help="Also print PropertiesList for each view")
    return p.parse_args(argv)


def _fmt_vec(v) -> str:
    try:
        return f"({float(v.x):.6g}, {float(v.y):.6g}, {float(v.z):.6g})"
    except Exception:
        return repr(v)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])

    try:
        import FreeCAD  # type: ignore
    except Exception as exc:
        print(f"FreeCAD import failed: {exc}")
        return 2

    doc = FreeCAD.ActiveDocument
    if doc is None:
        print("No ActiveDocument")
        return 2

    prefix = str(args.page_prefix or "")

    def pmsg(text: str) -> None:
        try:
            FreeCAD.Console.PrintMessage(text)
        except Exception:
            print(text, end="")

    pages = []
    for o in list(getattr(doc, "Objects", []) or []):
        tid = str(getattr(o, "TypeId", ""))
        if tid != "TechDraw::DrawPage":
            continue
        name = str(getattr(o, "Name", ""))
        label = str(getattr(o, "Label", ""))
        key = name or label
        if prefix and not (key.startswith(prefix) or label.startswith(prefix) or name.startswith(prefix)):
            continue
        pages.append(o)

    pmsg(f"[debug_techdraw_views] doc={doc.Name} pages={len(pages)} prefix={prefix!r}\n")

    for page in pages:
        page_name = str(getattr(page, "Name", ""))
        page_label = str(getattr(page, "Label", ""))
        views = list(getattr(page, "Views", []) or [])
        pmsg(f"\n[page] {page_name} label={page_label!r} views={len(views)}\n")

        for v in views:
            v_name = str(getattr(v, "Name", ""))
            v_label = str(getattr(v, "Label", ""))
            v_tid = str(getattr(v, "TypeId", ""))

            direction = getattr(v, "Direction", None)
            xdirection = getattr(v, "XDirection", None)
            scale = getattr(v, "Scale", None)
            scaletype = getattr(v, "ScaleType", None)

            # Optional markers we set in freecad_dome.techdraw
            tgt_d = getattr(v, "_geodesic_td_direction", None)
            tgt_xd = getattr(v, "_geodesic_td_xdirection", None)

            pmsg(
                f"  [view] {v_name} label={v_label!r} type={v_tid}\n"
                f"    Direction={_fmt_vec(direction)} XDirection={_fmt_vec(xdirection)}\n"
                f"    Scale={scale!r} ScaleType={scaletype!r}\n"
            )
            if tgt_d is not None or tgt_xd is not None:
                pmsg(
                    f"    TargetDirection={_fmt_vec(tgt_d)} TargetXDirection={_fmt_vec(tgt_xd)}\n"
                )

            if bool(args.show_properties):
                props = list(getattr(v, "PropertiesList", []) or [])
                props_s = ", ".join(props)
                pmsg(f"    PropertiesList=[{props_s}]\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
