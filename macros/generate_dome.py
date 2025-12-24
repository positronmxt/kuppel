"""Generate dome (FreeCAD macro).

This macro runs the existing generator entrypoint from `scripts/generate_dome.py`
inside FreeCAD GUI.

- Prompts for config JSON and output directory (optional; defaults to repo values)
- Then runs `scripts.generate_dome.main()` which can show the parameter dialog.

Add/run in FreeCAD:
- Macro -> Macros... -> Add -> select this file
- Execute
"""

from __future__ import annotations

import os
import sys
import traceback
import importlib.util
from pathlib import Path

try:
    import FreeCADGui  # noqa: F401
    from PySide import QtWidgets
except Exception as exc:
    raise SystemExit(f"This macro must be run inside FreeCAD GUI. Error: {exc}")

import FreeCAD  # type: ignore


def _repo_root() -> Path:
    macro_path = Path(globals().get("__file__", "")).resolve()
    if macro_path.is_file():
        return macro_path.parents[1]
    return Path.cwd()


def _pick_config(default_path: Path) -> Path:
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select config JSON (Cancel for default)",
        str(default_path.parent),
        "JSON (*.json);;All Files (*)",
    )
    if not path:
        return default_path
    return Path(path)


def _pick_out_dir(default_dir: Path) -> Path:
    path = QtWidgets.QFileDialog.getExistingDirectory(
        None,
        "Select output directory (Cancel for default)",
        str(default_dir),
    )
    if not path:
        return default_dir
    return Path(path)


def main() -> None:
    repo_root = _repo_root().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Avoid accidentally reusing stale modules from a previous run/session.
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts.") or key == "freecad_dome" or key.startswith("freecad_dome."):
            sys.modules.pop(key, None)

    default_config = repo_root / "configs" / "base.json"
    config_path = _pick_config(default_config) if default_config.exists() else default_config
    out_dir = _pick_out_dir(repo_root / "exports")

    old_argv = list(sys.argv)
    try:
        # Make relative paths and exports predictable.
        os.chdir(str(repo_root))
        sys.argv = [
            "generate_dome.py",
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
        ]

        # Load the generator entrypoint from THIS repo path, not from whatever
        # Python happens to resolve first.
        entry_path = (repo_root / "scripts" / "generate_dome.py").resolve()
        if not entry_path.exists():
            QtWidgets.QMessageBox.critical(None, "Dome generation failed", f"Missing entrypoint: {entry_path}")
            return
        spec = importlib.util.spec_from_file_location("geodesic_generate_dome", str(entry_path))
        if spec is None or spec.loader is None:
            QtWidgets.QMessageBox.critical(None, "Dome generation failed", f"Could not load: {entry_path}")
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        try:
            FreeCAD.Console.PrintMessage(f"[macro] repo_root={repo_root}\n")
            FreeCAD.Console.PrintMessage(f"[macro] entrypoint={entry_path}\n")
            FreeCAD.Console.PrintMessage(f"[macro] config={config_path}\n")
            FreeCAD.Console.PrintMessage(f"[macro] out_dir={out_dir}\n")
        except Exception:
            pass

        module.main()

        doc = FreeCAD.ActiveDocument
        if doc is not None:
            try:
                doc.recompute()
            except Exception:
                pass
            try:
                FreeCADGui.ActiveDocument.ActiveView.viewAxonometric()
                FreeCADGui.ActiveDocument.ActiveView.fitAll()
            except Exception:
                pass
            obj_count = len(getattr(doc, "Objects", []) or [])
            QtWidgets.QMessageBox.information(
                None,
                "Dome generation",
                f"Done.\n\nDocument: {doc.Name}\nObjects: {obj_count}\n\nOutputs in:\n{out_dir}",
            )
        else:
            QtWidgets.QMessageBox.information(None, "Dome generation", f"Done. Outputs in:\n{out_dir}")

    except SystemExit as exc:
        code = getattr(exc, "code", None)
        QtWidgets.QMessageBox.information(None, "Dome generation", f"Canceled/Exited (code={code}).")
    except BaseException:
        err = traceback.format_exc()
        QtWidgets.QMessageBox.critical(None, "Dome generation failed", err)
        raise
    finally:
        sys.argv = old_argv


main()
