#!/usr/bin/env python3
"""Fix IFC length unit to metres.

If an IFC file declares LENGTHUNIT as millimetres
(IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.)) while coordinates are authored in
metres, importers like BlenderBIM will scale the model down by 1000.

This script rewrites IFCSIUNIT(...,.LENGTHUNIT.,<PREFIX>,.METRE.) to use no
prefix ($), i.e. metres.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def fix_ifc_length_units_to_meters(text: str) -> tuple[str, int]:
    pattern = re.compile(
        r"(IFCSIUNIT\(\s*(?:\*|\$)\s*,\s*\.LENGTHUNIT\.\s*,\s*)\.[A-Z]+\.\s*,\s*(\.METRE\.\s*\))"
    )
    return pattern.subn(r"\1$,\2", text)


def fix_ifc_geometry_scale_if_needed(text: str, *, scale: float | None = None) -> tuple[str, int]:
    """Scale IFCCARTESIANPOINT coordinates when IFC geometry is 1000x too small.

    If scale is None, attempts auto-detection by comparing embedded FreeCAD Vector
    metadata (stored as IFCTEXT strings) to IFC geometry points. Falls back to a
    simple threshold check.
    """

    pt_pat = re.compile(r"IFCCARTESIANPOINT\(\(([^)]*)\)\)")
    vec_pat = re.compile(
        r"Vector \(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)"
    )

    pt_max = 0.0
    pt_count = 0
    for m in pt_pat.finditer(text):
        coords = m.group(1)
        for tok in coords.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = abs(float(tok))
            except Exception:
                continue
            if v > pt_max:
                pt_max = v
        pt_count += 1
        if pt_count >= 800:
            break

    if pt_max <= 0:
        return text, 0

    if scale is None:
        vec_max = None
        vm = vec_pat.search(text)
        if vm:
            try:
                vec_max = max(
                    abs(float(vm.group(1))),
                    abs(float(vm.group(2))),
                    abs(float(vm.group(3))),
                )
            except Exception:
                vec_max = None

        if vec_max is not None and vec_max > 0.5:
            ratio = vec_max / pt_max
            if 900.0 <= ratio <= 1100.0:
                scale = 1000.0

        if scale is None and pt_max < 0.05:
            scale = 1000.0

    if scale is None:
        return text, 0

    def _fmt(x: float) -> str:
        return format(x, ".15g")

    def repl(m: re.Match) -> str:
        coords = m.group(1)
        parts: list[str] = []
        for tok in coords.split(","):
            t = tok.strip()
            if not t:
                continue
            try:
                v = float(t)
                parts.append(_fmt(v * float(scale)))
            except Exception:
                parts.append(t)
        return f"IFCCARTESIANPOINT(({','.join(parts)}))"

    fixed, n = pt_pat.subn(repl, text)
    return fixed, n


def main() -> int:
    ap = argparse.ArgumentParser(description="Rewrite IFC LENGTHUNIT to metres")
    ap.add_argument("input", type=Path, help="Input IFC (.ifc)")
    ap.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output IFC (.ifc). If omitted, edits input in-place.",
    )
    ap.add_argument(
        "--fix-scale",
        action="store_true",
        help="Also auto-fix common 1000x geometry scale issue (scales IFCCARTESIANPOINTs)",
    )
    args = ap.parse_args()

    inp: Path = args.input
    out: Path = args.output if args.output is not None else inp

    text = inp.read_text(encoding="utf-8", errors="ignore")
    fixed, n_units = fix_ifc_length_units_to_meters(text)
    n_scale = 0
    if args.fix_scale:
        fixed, n_scale = fix_ifc_geometry_scale_if_needed(fixed)

    if n_units or n_scale or out != inp:
        out.write_text(fixed, encoding="utf-8")
    if n_units:
        print(f"Updated {n_units} LENGTHUNIT definition(s) -> metres: {out}")
    else:
        print(f"No prefixed LENGTHUNIT found; unit header unchanged: {out}")
    if args.fix_scale:
        if n_scale:
            print(f"Scaled {n_scale} IFCCARTESIANPOINT(s) (auto-detected): {out}")
        else:
            print(f"No scale fix applied: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
