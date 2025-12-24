#!/usr/bin/env python3
"""Tiny GUI wrapper for `export_joint_variant_batch.py`.

Uses Tkinter (stdlib) and runs the batch in a background thread.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Tkinter is not available in this Python build. "
        "On Debian/Ubuntu try: sudo apt install python3-tk\n"
        f"Original error: {exc}"
    )


def _set_text(text_widget: tk.Text, text: str) -> None:
    text_widget.configure(state="normal")
    text_widget.delete("1.0", "end")
    text_widget.insert("end", text)
    text_widget.configure(state="disabled")


def main() -> int:
    root = tk.Tk()
    root.title("Geodesic: Joint Variant Batch Export")

    config_var = tk.StringVar(value="configs/base.json")
    out_dir_var = tk.StringVar(value="exports/panel_frame_cutplanes")
    q_var = tk.StringVar(value="0.95")
    tol_var = tk.StringVar(value="1e-5")
    max_iter_var = tk.StringVar(value="40")
    also_hub_points_var = tk.BooleanVar(value=False)

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(fill="both", expand=True)

    def browse_config() -> None:
        path = filedialog.askopenfilename(
            title="Select config JSON",
            initialdir=str(Path.cwd()),
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
        if path:
            config_var.set(path)

    def browse_out_dir() -> None:
        path = filedialog.askdirectory(title="Select output dir", initialdir=str(Path.cwd()))
        if path:
            out_dir_var.set(path)

    row = 0
    tk.Label(frame, text="Config").grid(row=row, column=0, sticky="w")
    tk.Entry(frame, textvariable=config_var, width=60).grid(row=row, column=1, sticky="we")
    tk.Button(frame, text="Browse", command=browse_config).grid(row=row, column=2, padx=(6, 0))

    row += 1
    tk.Label(frame, text="Out dir").grid(row=row, column=0, sticky="w", pady=(6, 0))
    tk.Entry(frame, textvariable=out_dir_var, width=60).grid(row=row, column=1, sticky="we", pady=(6, 0))
    tk.Button(frame, text="Browse", command=browse_out_dir).grid(row=row, column=2, padx=(6, 0), pady=(6, 0))

    row += 1
    tk.Label(frame, text="Quantile q").grid(row=row, column=0, sticky="w", pady=(6, 0))
    tk.Entry(frame, textvariable=q_var, width=12).grid(row=row, column=1, sticky="w", pady=(6, 0))

    row += 1
    tk.Label(frame, text="Tolerance (m)").grid(row=row, column=0, sticky="w", pady=(6, 0))
    tk.Entry(frame, textvariable=tol_var, width=12).grid(row=row, column=1, sticky="w", pady=(6, 0))

    row += 1
    tk.Label(frame, text="Max iter").grid(row=row, column=0, sticky="w", pady=(6, 0))
    tk.Entry(frame, textvariable=max_iter_var, width=12).grid(row=row, column=1, sticky="w", pady=(6, 0))

    row += 1
    tk.Checkbutton(frame, text="Also export node-hub points", variable=also_hub_points_var).grid(
        row=row, column=1, sticky="w", pady=(6, 0)
    )

    row += 1
    log = tk.Text(frame, height=14, width=90)
    log.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
    log.configure(state="disabled")

    frame.grid_columnconfigure(1, weight=1)
    frame.grid_rowconfigure(row, weight=1)

    def run_clicked() -> None:
        _set_text(log, "Running...\n")

        def work() -> None:
            try:
                from scripts.export_joint_variant_batch import run_batch

                manifest = run_batch(
                    config=str(config_var.get()).strip(),
                    out_dir=str(out_dir_var.get()).strip(),
                    q=float(q_var.get()),
                    tol=float(tol_var.get()),
                    max_iter=int(float(max_iter_var.get())),
                    also_node_hub_points=bool(also_hub_points_var.get()),
                )
                root.after(0, lambda: _set_text(log, f"OK\nmanifest={manifest}\n"))
            except Exception:
                err = traceback.format_exc()
                root.after(0, lambda: _set_text(log, err))
                root.after(0, lambda: messagebox.showerror("Batch export failed", err))

        threading.Thread(target=work, daemon=True).start()

    btns = tk.Frame(frame)
    btns.grid(row=0, column=3, rowspan=2, sticky="ne", padx=(10, 0))
    tk.Button(btns, text="Run", command=run_clicked, width=10).pack(pady=(0, 6))
    tk.Button(btns, text="Quit", command=root.destroy, width=10).pack()

    root.minsize(760, 360)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
