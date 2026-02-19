"""Qt parameter dialog for interactive dome configuration.

This module separates the GUI layer (PySide2/PySide6) from the pure-Python
parameter logic in :mod:`parameters`, keeping the latter testable without
any Qt dependency.

The dialog is organized into tabs matching the configuration hierarchy:

1. **Geometry** — sphere shape, frequency, truncation
2. **Structure** — strut stock, material, bevels, node-fit
3. **Node connectors** — plate/ball/pipe hubs with bolt specs
4. **Covering & Weather** — covering material, gaskets, panel frames
5. **Openings** — base wall, entry porch, door, ventilation
6. **Foundation** — strip / point / screw-anchor layout
7. **Analysis & Export** — load calc, production, costing, spreadsheets
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .parameters import DomeParameters

__all__ = ["prompt_parameters_dialog"]

# ---------------------------------------------------------------------------
# Lazy Qt import
# ---------------------------------------------------------------------------

QtWidgets = None


def _import_qt() -> tuple[Any, Any] | tuple[None, None]:
    global QtWidgets
    if QtWidgets is not None:
        return QtWidgets, None
    try:
        from PySide2 import QtWidgets as _QtWidgets  # type: ignore

        QtWidgets = _QtWidgets
        return QtWidgets, None
    except ImportError:  # pragma: no cover - GUI optional dependency
        try:
            from PySide6 import QtWidgets as _QtWidgets  # type: ignore

            QtWidgets = _QtWidgets
            return QtWidgets, None
        except ImportError:  # pragma: no cover - GUI optional dependency
            return None, None


def _has_display() -> bool:
    if sys.platform.startswith("win") or sys.platform == "darwin":  # pragma: no cover - platform guards
        return True
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False


# ---------------------------------------------------------------------------
# Public dialog entry point
# ---------------------------------------------------------------------------


def prompt_parameters_dialog(initial: DomeParameters) -> DomeParameters | None:
    """Show a Qt dialog to let the user tweak parameters."""

    widgets, _ = _import_qt()
    if widgets is None:
        logging.info("PySide is not available; skipping parameter dialog")
        return initial
    if not _has_display():
        logging.info("No graphical display detected; skipping parameter dialog")
        return initial

    app = widgets.QApplication.instance()
    created_app = False
    if app is None:
        app = widgets.QApplication([])
        created_app = True

    class _ParameterDialog(widgets.QDialog):
        """Tab-based dialog covering all DomeParameters groups."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Geodesic Dome — Parameters")
            self.setMinimumWidth(540)
            root = widgets.QVBoxLayout(self)

            tabs = widgets.QTabWidget()
            root.addWidget(tabs)

            # ---------------------------------------------------------------
            # Tab 1: Geometry
            # ---------------------------------------------------------------
            geo_page = widgets.QWidget()
            geo_form = widgets.QFormLayout(geo_page)

            self.radius = self._double_spin(0.5, 30.0, 0.1, initial.radius_m)
            geo_form.addRow("Raadius (m)", self.radius)

            self.frequency = widgets.QSpinBox()
            self.frequency.setRange(1, 10)
            self.frequency.setValue(initial.frequency)
            geo_form.addRow("Sagedus (frequency)", self.frequency)

            self.truncation = self._double_spin(0.01, 0.9, 0.01, initial.truncation_ratio)
            geo_form.addRow("Trunkeerimise suhe", self.truncation)

            self.segment = self._double_spin(0.1, 1.0, 0.01, initial.hemisphere_ratio)
            geo_form.addRow("Poolsfääri suhe", self.segment)

            self.use_truncation = widgets.QCheckBox("Rakenda trunkeerimist")
            self.use_truncation.setChecked(initial.use_truncation)
            geo_form.addRow(self.use_truncation)

            tabs.addTab(geo_page, "Geomeetria")

            # ---------------------------------------------------------------
            # Tab 2: Structure
            # ---------------------------------------------------------------
            struct_page = widgets.QWidget()
            struct_lay = widgets.QVBoxLayout(struct_page)

            # -- Strut generation toggle
            self.mode_struts = widgets.QCheckBox("Genereeri latid (konstruktiivne mudel)")
            self.mode_struts.setChecked(initial.generate_struts)
            struct_lay.addWidget(self.mode_struts)

            # -- Material + stock
            mat_group = widgets.QGroupBox("Materjal ja latt")
            mat_form = widgets.QFormLayout(mat_group)

            self.material = widgets.QComboBox()
            for key, spec in initial.materials.items():
                label = spec.name if spec.name else key.title()
                self.material.addItem(label, key)
            idx = self.material.findData(initial.material)
            if idx >= 0:
                self.material.setCurrentIndex(idx)
            mat_form.addRow("Materjal", self.material)

            self.stock_width = self._double_spin(0.005, 0.5, 0.001, initial.stock_width_m)
            mat_form.addRow("Lati laius (m)", self.stock_width)
            self.stock_height = self._double_spin(0.005, 0.5, 0.001, initial.stock_height_m)
            mat_form.addRow("Lati kõrgus (m)", self.stock_height)
            self.kerf = self._double_spin(0.0, 0.02, 0.0005, initial.kerf_m)
            mat_form.addRow("Saetera (m)", self.kerf)
            self.clearance = self._double_spin(0.0, 0.02, 0.0005, initial.clearance_m)
            mat_form.addRow("Kliirens (m)", self.clearance)

            self.use_bevels = widgets.QCheckBox("Kaldlõiked (beveled)")
            self.use_bevels.setChecked(initial.use_bevels)
            mat_form.addRow(self.use_bevels)
            struct_lay.addWidget(mat_group)

            # -- Strut node-fit
            self.strut_group = widgets.QGroupBox("Lati otsa sobitamine (node-fit)")
            strut_form = widgets.QFormLayout(self.strut_group)

            self.node_fit_mode = widgets.QComboBox()
            self.node_fit_mode.addItem("Tasapinnaline (planar) — lõiketasandid", "planar")
            self.node_fit_mode.addItem("Kooniline (tapered) — CNC ahenemine otstes", "tapered")
            self.node_fit_mode.addItem("Voronoi — täpne nurga-jaotus", "voronoi")
            idx_mode = self.node_fit_mode.findData(initial.node_fit_mode)
            if idx_mode >= 0:
                self.node_fit_mode.setCurrentIndex(idx_mode)
            strut_form.addRow("Sobitamise režiim", self.node_fit_mode)

            self.node_fit_taper_ratio = self._double_spin(0.1, 0.9, 0.05, initial.node_fit_taper_ratio)
            strut_form.addRow("Ahenemise suhe (tapered)", self.node_fit_taper_ratio)

            self.node_fit_plane = widgets.QComboBox()
            self.node_fit_plane.addItem("Radiaalne (sfääri puutuja)", "radial")
            self.node_fit_plane.addItem("Telje suunaline (sirgelt lõigatud)", "axis")
            self.node_fit_plane.addItem("Miiter (tihe liit, ilma tühimikuta)", "miter")
            idx_plane = self.node_fit_plane.findData(initial.node_fit_plane_mode)
            if idx_plane >= 0:
                self.node_fit_plane.setCurrentIndex(idx_plane)
            strut_form.addRow("Otsatasand (planar/tapered)", self.node_fit_plane)

            self.node_fit_extension = self._double_spin(0.0, 0.030, 0.001, initial.node_fit_extension_m)
            strut_form.addRow("Pikendus üle sõlme (m)", self.node_fit_extension)

            self.node_fit_separation = widgets.QCheckBox("Eraldustasandid lattide vahel")
            self.node_fit_separation.setChecked(initial.node_fit_use_separation_planes)
            strut_form.addRow(self.node_fit_separation)

            self.split_struts_per_panel = widgets.QCheckBox("Poolita latid paneeliserva kaupa")
            self.split_struts_per_panel.setChecked(initial.split_struts_per_panel)
            strut_form.addRow(self.split_struts_per_panel)
            struct_lay.addWidget(self.strut_group)

            # -- Advanced strut settings (collapsible)
            self.advanced_strut_group = widgets.QGroupBox("Täpsemad seaded")
            self.advanced_strut_group.setCheckable(True)
            self.advanced_strut_group.setChecked(False)
            adv_form = widgets.QFormLayout(self.advanced_strut_group)

            self.strut_profile = widgets.QComboBox()
            self.strut_profile.addItem("Ristkülik", "rectangular")
            self.strut_profile.addItem("Ümar", "round")
            self.strut_profile.addItem("Trapets", "trapezoidal")
            idx_sp = self.strut_profile.findData(initial.strut_profile)
            if idx_sp >= 0:
                self.strut_profile.setCurrentIndex(idx_sp)
            adv_form.addRow("Lati profiil", self.strut_profile)

            self.cap_blend_mode = widgets.QComboBox()
            self.cap_blend_mode.addItem("Terav (sharp)", "sharp")
            self.cap_blend_mode.addItem("Faas (chamfer)", "chamfer")
            self.cap_blend_mode.addItem("Filee (fillet)", "fillet")
            idx_cb = self.cap_blend_mode.findData(initial.cap_blend_mode)
            if idx_cb >= 0:
                self.cap_blend_mode.setCurrentIndex(idx_cb)
            adv_form.addRow("Otsa blend-režiim", self.cap_blend_mode)

            self.bevel_fillet_radius = self._double_spin(0.0, 0.050, 0.001, initial.bevel_fillet_radius_m)
            adv_form.addRow("Kaldlõike filee raadius (m)", self.bevel_fillet_radius)

            self.min_wedge_angle = self._double_spin(0.0, 90.0, 1.0, initial.min_wedge_angle_deg)
            adv_form.addRow("Min kiilunurk (°)", self.min_wedge_angle)

            self.cap_length_factor = self._double_spin(0.5, 10.0, 0.5, initial.cap_length_factor)
            adv_form.addRow("Otsa pikkuse koefitsient", self.cap_length_factor)

            self.max_cap_ratio = self._double_spin(0.05, 0.95, 0.05, initial.max_cap_ratio)
            adv_form.addRow("Otsa max suhe", self.max_cap_ratio)

            self.generate_belt_cap = widgets.QCheckBox("Genereeri vöö-otsad")
            self.generate_belt_cap.setChecked(initial.generate_belt_cap)
            adv_form.addRow(self.generate_belt_cap)

            struct_lay.addWidget(self.advanced_strut_group)

            struct_lay.addStretch(1)
            tabs.addTab(struct_page, "Konstruktsioon")

            # ---------------------------------------------------------------
            # Tab 3: Node Connectors
            # ---------------------------------------------------------------
            conn_page = widgets.QWidget()
            conn_lay = widgets.QVBoxLayout(conn_page)

            self.generate_node_connectors = widgets.QCheckBox("Genereeri sõlmekonnektorid")
            self.generate_node_connectors.setChecked(initial.generate_node_connectors)
            conn_lay.addWidget(self.generate_node_connectors)

            self.conn_detail_group = widgets.QGroupBox("Konnektori tüüp ja mõõdud")
            conn_form = widgets.QFormLayout(self.conn_detail_group)

            self.node_connector_type = widgets.QComboBox()
            self.node_connector_type.addItem("Plaat (plate) — keevitatud/lõigatud terasplaat", "plate")
            self.node_connector_type.addItem("Kera (ball) — massivne sfääriline ühendus", "ball")
            self.node_connector_type.addItem("Toru (pipe) — torukujuline liides", "pipe")
            self.node_connector_type.addItem("Lapjoint — latid katavad teineteist sõlmes", "lapjoint")
            idx_ct = self.node_connector_type.findData(initial.node_connector_type)
            if idx_ct >= 0:
                self.node_connector_type.setCurrentIndex(idx_ct)
            conn_form.addRow("Tüüp", self.node_connector_type)

            self.node_connector_thickness = self._double_spin(0.001, 0.030, 0.001, initial.node_connector_thickness_m)
            conn_form.addRow("Plaadi paksus (m)", self.node_connector_thickness)

            self.node_connector_bolt_diameter = self._double_spin(0.004, 0.030, 0.001, initial.node_connector_bolt_diameter_m)
            conn_form.addRow("Poldi läbimõõt (m)", self.node_connector_bolt_diameter)

            self.node_connector_bolt_length = self._double_spin(0.020, 0.200, 0.005, initial.node_connector_bolt_length_m)
            conn_form.addRow("Poldi pikkus (m)", self.node_connector_bolt_length)

            self.node_connector_washer_diameter = self._double_spin(0.008, 0.050, 0.001, initial.node_connector_washer_diameter_m)
            conn_form.addRow("Seib läbimõõt (m)", self.node_connector_washer_diameter)

            self.node_connector_bolt_offset = self._double_spin(0.010, 0.100, 0.001, initial.node_connector_bolt_offset_m)
            conn_form.addRow("Poldi nihe keskpunktist (m)", self.node_connector_bolt_offset)

            self.node_connector_lap_extension = self._double_spin(0.005, 0.100, 0.005, initial.node_connector_lap_extension_m)
            conn_form.addRow("Lap-joint pikendus (m)", self.node_connector_lap_extension)

            conn_lay.addWidget(self.conn_detail_group)

            # Info label
            conn_info = widgets.QLabel(
                "ℹ Plaat — iga sõlme jaoks lõigatakse terasplaat, kuhu poldiga kinnituvad latid.\n"
                "  Kera — kasutatakse massiivseid sfäärilisi ühendusi (sobib suurematele kuplitele).\n"
                "  Toru — toruliides, kuhu latid lükkuvad sisse.\n"
                "  Lapjoint — latid ulatuvad sõlmest mööda ja katavad naaberlatti (ilma hub-ita)."
            )
            conn_info.setWordWrap(True)
            conn_info.setStyleSheet("color: #666; font-size: 11px; margin-top: 6px;")
            conn_lay.addWidget(conn_info)

            conn_lay.addStretch(1)
            tabs.addTab(conn_page, "Sõlmed")

            # ---------------------------------------------------------------
            # Tab 4: Panels & Covering
            # ---------------------------------------------------------------
            cover_page = widgets.QWidget()
            cover_lay = widgets.QVBoxLayout(cover_page)

            # -- Panel generation toggles (top of tab for clarity)
            panel_gen_group = widgets.QGroupBox("Paneelide genereerimine")
            panel_gen_layout = widgets.QVBoxLayout(panel_gen_group)
            self.mode_panels = widgets.QCheckBox("Genereeri paneeli pinnad")
            self.mode_panels.setChecked(initial.generate_panel_faces)
            panel_gen_layout.addWidget(self.mode_panels)
            self.mode_frames = widgets.QCheckBox("Genereeri paneeliraamid")
            self.mode_frames.setChecked(initial.generate_panel_frames)
            panel_gen_layout.addWidget(self.mode_frames)
            cover_lay.addWidget(panel_gen_group)

            # -- Covering material
            cover_mat_group = widgets.QGroupBox("Kattematerjal")
            cover_mat_form = widgets.QFormLayout(cover_mat_group)

            self.covering_type = widgets.QComboBox()
            _covering_items = [
                ("Klaas (glass)", "glass"),
                ("Polükarbonaat kaksikseina 8mm", "polycarbonate_twin_8"),
                ("Polükarbonaat kaksikseina 10mm", "polycarbonate_twin_10"),
                ("Polükarbonaat kaksikseina 16mm", "polycarbonate_twin_16"),
                ("Polükarbonaat massiiv 4mm", "polycarbonate_solid_4"),
                ("Polükarbonaat massiiv 6mm", "polycarbonate_solid_6"),
            ]
            for label, key in _covering_items:
                self.covering_type.addItem(label, key)
            idx_cov = self.covering_type.findData(initial.covering_type)
            if idx_cov >= 0:
                self.covering_type.setCurrentIndex(idx_cov)
            cover_mat_form.addRow("Kattematerjal", self.covering_type)

            self.covering_thickness = self._double_spin(0.0, 0.05, 0.0005, initial.covering_thickness_m)
            cover_mat_form.addRow("Paksus override (m, 0=kataloog)", self.covering_thickness)

            self.covering_gap = self._double_spin(0.0, 0.1, 0.001, initial.covering_gap_m)
            cover_mat_form.addRow("Servapilude (m, 0=vaikimisi)", self.covering_gap)

            self.covering_delta_t = self._double_spin(0.0, 100.0, 1.0, initial.covering_delta_t_k)
            cover_mat_form.addRow("Temperatuurikõikumine ΔT (K)", self.covering_delta_t)

            self.covering_profile = widgets.QComboBox()
            _profile_items = [
                ("Ei kasuta", "none"),
                ("H-profiil alumiinium", "H_profile_alu"),
                ("H-profiil polükarbonaat", "H_profile_pc"),
                ("U-profiil alumiinium", "U_profile_alu"),
                ("Snap-cap alumiinium", "snap_cap_alu"),
            ]
            for label, key in _profile_items:
                self.covering_profile.addItem(label, key)
            idx_prof = self.covering_profile.findData(initial.covering_profile_type)
            if idx_prof >= 0:
                self.covering_profile.setCurrentIndex(idx_prof)
            cover_mat_form.addRow("Kinnitusprofiil", self.covering_profile)

            cover_lay.addWidget(cover_mat_group)

            # -- Panel frame dimensions
            frame_group = widgets.QGroupBox("Paneeliraami mõõdud")
            frame_form = widgets.QFormLayout(frame_group)
            self.frame_inset = self._double_spin(0.0, 0.5, 0.001, initial.panel_frame_inset_m)
            frame_form.addRow("Raami inset (m)", self.frame_inset)
            self.frame_width = self._double_spin(0.005, 0.5, 0.001, initial.panel_frame_profile_width_m)
            frame_form.addRow("Profiili laius (m)", self.frame_width)
            self.frame_height = self._double_spin(0.005, 0.5, 0.001, initial.panel_frame_profile_height_m)
            frame_form.addRow("Profiili kõrgus (m)", self.frame_height)
            cover_lay.addWidget(frame_group)

            # -- Glass overrides
            glass_group = widgets.QGroupBox("Klaasi seaded (override)")
            glass_form = widgets.QFormLayout(glass_group)
            self.glass_thickness = self._double_spin(0.0, 0.05, 0.0005, initial.glass_thickness_m)
            self.glass_gap = self._double_spin(0.0, 0.1, 0.001, initial.glass_gap_m)
            glass_form.addRow("Klaasi paksus (m)", self.glass_thickness)
            glass_form.addRow("Klaasi vahe (m)", self.glass_gap)
            cover_lay.addWidget(glass_group)

            # -- Weather protection
            weather_group = widgets.QGroupBox("Ilmastikukaitse")
            weather_form = widgets.QFormLayout(weather_group)

            self.generate_weather = widgets.QCheckBox("Genereeri ilmastikukaitse andmed")
            self.generate_weather.setChecked(initial.generate_weather)
            weather_form.addRow(self.generate_weather)

            self.gasket_type = widgets.QComboBox()
            _gasket_items = [
                ("EPDM D-profiil 10×8mm", "epdm_d_10x8"),
                ("EPDM P-profiil 9×5mm", "epdm_p_9x5"),
                ("Silikoon 12×8mm", "silicone_12x8"),
                ("Neopreen tasapindne 15×3mm", "neoprene_flat_15x3"),
                ("Butüülteip 20×1mm", "butyl_tape_20x1"),
            ]
            for label, key in _gasket_items:
                self.gasket_type.addItem(label, key)
            idx_g = self.gasket_type.findData(initial.gasket_type)
            if idx_g >= 0:
                self.gasket_type.setCurrentIndex(idx_g)
            weather_form.addRow("Tihendi tüüp", self.gasket_type)
            cover_lay.addWidget(weather_group)

            # -- Skylights / windows
            skylight_group = widgets.QGroupBox("Katuseluugid / aknad")
            skylight_form = widgets.QFormLayout(skylight_group)

            self.generate_skylights = widgets.QCheckBox("Genereeri katuseluugid")
            self.generate_skylights.setChecked(initial.generate_skylights)
            skylight_form.addRow(self.generate_skylights)

            self.skylight_count = self._int_spin(1, 20, initial.skylight_count)
            skylight_form.addRow("Luukide arv", self.skylight_count)

            self.skylight_position = widgets.QComboBox()
            _sky_pos_items = [
                ("Tipust (apex)", "apex"),
                ("Ringist (ring)", "ring"),
                ("Käsitsi (manual)", "manual"),
            ]
            for label, key in _sky_pos_items:
                self.skylight_position.addItem(label, key)
            idx_sp = self.skylight_position.findData(initial.skylight_position)
            if idx_sp >= 0:
                self.skylight_position.setCurrentIndex(idx_sp)
            skylight_form.addRow("Paigutus", self.skylight_position)

            self.skylight_glass_thickness = self._double_spin(0.001, 0.05, 0.001, initial.skylight_glass_thickness_m)
            skylight_form.addRow("Klaasi paksus (m)", self.skylight_glass_thickness)

            self.skylight_frame_width = self._double_spin(0.01, 0.2, 0.005, initial.skylight_frame_width_m)
            skylight_form.addRow("Raami laius (m)", self.skylight_frame_width)

            self.skylight_hinge_side = widgets.QComboBox()
            _hinge_items = [
                ("Ülemine (top)", "top"),
                ("Alumine (bottom)", "bottom"),
                ("Vasak (left)", "left"),
                ("Parem (right)", "right"),
            ]
            for label, key in _hinge_items:
                self.skylight_hinge_side.addItem(label, key)
            idx_hs = self.skylight_hinge_side.findData(initial.skylight_hinge_side)
            if idx_hs >= 0:
                self.skylight_hinge_side.setCurrentIndex(idx_hs)
            skylight_form.addRow("Hinge külg", self.skylight_hinge_side)

            self.skylight_material = widgets.QComboBox()
            _sky_mat_items = [
                ("Klaas (glass)", "glass"),
                ("Polükarbonaat (polycarbonate)", "polycarbonate"),
            ]
            for label, key in _sky_mat_items:
                self.skylight_material.addItem(label, key)
            idx_sm = self.skylight_material.findData(initial.skylight_material)
            if idx_sm >= 0:
                self.skylight_material.setCurrentIndex(idx_sm)
            skylight_form.addRow("Materjal", self.skylight_material)

            cover_lay.addWidget(skylight_group)

            cover_lay.addStretch(1)
            tabs.addTab(cover_page, "Paneelid ja kate")

            # ---------------------------------------------------------------
            # Tab 5: Openings
            # ---------------------------------------------------------------
            open_page = widgets.QWidget()
            open_lay = widgets.QVBoxLayout(open_page)

            # -- Base wall
            base_group = widgets.QGroupBox("Tsokkelseina + uks")
            base_form = widgets.QFormLayout(base_group)

            self.generate_base_wall = widgets.QCheckBox("Genereeri tsokkelseina")
            self.generate_base_wall.setChecked(initial.generate_base_wall)
            base_form.addRow(self.generate_base_wall)

            self.base_wall_height = self._double_spin(0.1, 5.0, 0.01, initial.base_wall_height_m)
            base_form.addRow("Seina kõrgus (m)", self.base_wall_height)
            self.base_wall_thickness = self._double_spin(0.02, 1.0, 0.01, initial.base_wall_thickness_m)
            base_form.addRow("Seina paksus (m)", self.base_wall_thickness)
            open_lay.addWidget(base_group)

            # -- Entry porch
            porch_group = widgets.QGroupBox("Sissepääsu veranda")
            porch_form = widgets.QFormLayout(porch_group)

            self.generate_entry_porch = widgets.QCheckBox("Genereeri veranda")
            self.generate_entry_porch.setChecked(initial.generate_entry_porch)
            porch_form.addRow(self.generate_entry_porch)

            self.porch_depth = self._double_spin(0.05, 0.5, 0.01, initial.porch_depth_m)
            porch_form.addRow("Sügavus (m)", self.porch_depth)
            self.porch_width = self._double_spin(0.6, 3.0, 0.01, initial.porch_width_m)
            porch_form.addRow("Laius (m)", self.porch_width)
            self.porch_height = self._double_spin(0.6, 3.0, 0.01, initial.porch_height_m)
            porch_form.addRow("Kõrgus (m)", self.porch_height)
            self.porch_member = self._double_spin(0.02, 0.12, 0.001, initial.porch_member_size_m)
            porch_form.addRow("Raami mõõt (m)", self.porch_member)
            self.porch_glass = self._double_spin(0.0, 0.03, 0.0005, initial.porch_glass_thickness_m)
            porch_form.addRow("Klaasi paksus (m)", self.porch_glass)
            open_lay.addWidget(porch_group)

            # -- Door
            door_group = widgets.QGroupBox("Ukseava")
            door_form = widgets.QFormLayout(door_group)

            self.door_width = self._double_spin(0.3, 2.0, 0.01, initial.door_width_m)
            door_form.addRow("Laius (m)", self.door_width)
            self.door_height = self._double_spin(0.3, 3.0, 0.01, initial.door_height_m)
            door_form.addRow("Kõrgus (m)", self.door_height)
            self.auto_door_angle = widgets.QCheckBox("Automaatne nurk")
            self.auto_door_angle.setChecked(initial.auto_door_angle)
            door_form.addRow(self.auto_door_angle)
            self.door_angle = self._double_spin(-360.0, 360.0, 1.0, initial.door_angle_deg)
            door_form.addRow("Käsitsi nurk (°)", self.door_angle)
            self.door_clearance = self._double_spin(0.0, 0.2, 0.001, initial.door_clearance_m)
            door_form.addRow("Kliirens (m)", self.door_clearance)
            open_lay.addWidget(door_group)

            # -- Ventilation
            vent_group = widgets.QGroupBox("Ventilatsioon")
            vent_form = widgets.QFormLayout(vent_group)

            self.generate_ventilation = widgets.QCheckBox("Genereeri ventilatsiooni plaan")
            self.generate_ventilation.setChecked(initial.generate_ventilation)
            vent_form.addRow(self.generate_ventilation)

            self.ventilation_mode = widgets.QComboBox()
            _vent_modes = [
                ("Automaatne (apex + ring mix)", "auto"),
                ("Ainult tipp (apex)", "apex"),
                ("Ainult vöönd (ring)", "ring"),
                ("Käsitsi (manual) — vali paneelid", "manual"),
            ]
            for label, key in _vent_modes:
                self.ventilation_mode.addItem(label, key)
            idx_vm = self.ventilation_mode.findData(initial.ventilation_mode)
            if idx_vm >= 0:
                self.ventilation_mode.setCurrentIndex(idx_vm)
            vent_form.addRow("Režiim", self.ventilation_mode)

            self.ventilation_target = self._double_spin(0.0, 1.0, 0.01, initial.ventilation_target_ratio)
            vent_form.addRow("Sihtvahe (vent/põrand)", self.ventilation_target)

            self.ventilation_apex_count = widgets.QSpinBox()
            self.ventilation_apex_count.setRange(0, 20)
            self.ventilation_apex_count.setValue(initial.ventilation_apex_count)
            vent_form.addRow("Tipp-paneelid (apex)", self.ventilation_apex_count)

            self.ventilation_ring_count = widgets.QSpinBox()
            self.ventilation_ring_count.setRange(0, 30)
            self.ventilation_ring_count.setValue(initial.ventilation_ring_count)
            vent_form.addRow("Vöönd-paneelid (ring)", self.ventilation_ring_count)

            self.ventilation_ring_height = self._double_spin(0.0, 1.0, 0.05, initial.ventilation_ring_height_ratio)
            vent_form.addRow("Vööndi kõrgus (0=vöö, 1=tipp)", self.ventilation_ring_height)

            open_lay.addWidget(vent_group)

            # -- Riser wall (pikendusring)
            riser_group = widgets.QGroupBox("Pikendusring (riser wall)")
            riser_form = widgets.QFormLayout(riser_group)

            self.generate_riser_wall = widgets.QCheckBox("Genereeri pikendusring")
            self.generate_riser_wall.setChecked(initial.generate_riser_wall)
            riser_form.addRow(self.generate_riser_wall)

            self.riser_height = self._double_spin(0.1, 5.0, 0.05, initial.riser_height_m)
            riser_form.addRow("Kõrgus (m)", self.riser_height)

            self.riser_thickness = self._double_spin(0.05, 1.0, 0.01, initial.riser_thickness_m)
            riser_form.addRow("Paksus (m)", self.riser_thickness)

            self.riser_material = widgets.QComboBox()
            _riser_mat_items = [
                ("Betoon (concrete)", "concrete"),
                ("Puit (wood)", "wood"),
                ("Teras (steel)", "steel"),
            ]
            for label, key in _riser_mat_items:
                self.riser_material.addItem(label, key)
            idx_rm = self.riser_material.findData(initial.riser_material)
            if idx_rm >= 0:
                self.riser_material.setCurrentIndex(idx_rm)
            riser_form.addRow("Materjal", self.riser_material)

            self.riser_connection_type = widgets.QComboBox()
            _riser_conn_items = [
                ("Äärik (flange)", "flange"),
                ("Sisseehitatud (embed)", "embed"),
                ("Poltühendus (bolted)", "bolted"),
            ]
            for label, key in _riser_conn_items:
                self.riser_connection_type.addItem(label, key)
            idx_rc = self.riser_connection_type.findData(initial.riser_connection_type)
            if idx_rc >= 0:
                self.riser_connection_type.setCurrentIndex(idx_rc)
            riser_form.addRow("Ühenduse tüüp", self.riser_connection_type)

            self.riser_door_integration = widgets.QCheckBox("Lõika uks ka riser seinast")
            self.riser_door_integration.setChecked(initial.riser_door_integration)
            riser_form.addRow(self.riser_door_integration)

            self.riser_stud_spacing = self._double_spin(0.2, 2.0, 0.05, initial.riser_stud_spacing_m)
            riser_form.addRow("Postide samm (m, puit)", self.riser_stud_spacing)

            self.riser_segments = self._int_spin(8, 120, initial.riser_segments)
            riser_form.addRow("Segmendid (silinder)", self.riser_segments)

            open_lay.addWidget(riser_group)

            open_lay.addStretch(1)
            tabs.addTab(open_page, "Avad ja vent")

            # ---------------------------------------------------------------
            # Tab 6: Foundation
            # ---------------------------------------------------------------
            found_page = widgets.QWidget()
            found_lay = widgets.QVBoxLayout(found_page)

            self.generate_foundation = widgets.QCheckBox("Genereeri vundamendi plaan")
            self.generate_foundation.setChecked(initial.generate_foundation)
            found_lay.addWidget(self.generate_foundation)

            self.found_detail_group = widgets.QGroupBox("Vundamendi tüüp ja mõõdud")
            found_form = widgets.QFormLayout(self.found_detail_group)

            self.foundation_type = widgets.QComboBox()
            _found_items = [
                ("Vundamendilint (strip) — pidev lintvundament", "strip"),
                ("Punktvundament (point) — isoleeritud sambad", "point"),
                ("Kruviankur (screw) — kruvitav vaiaankur", "screw_anchor"),
            ]
            for label, key in _found_items:
                self.foundation_type.addItem(label, key)
            idx_ft = self.foundation_type.findData(initial.foundation_type)
            if idx_ft >= 0:
                self.foundation_type.setCurrentIndex(idx_ft)
            found_form.addRow("Tüüp", self.foundation_type)

            self.foundation_bolt_diameter = self._double_spin(0.008, 0.030, 0.001, initial.foundation_bolt_diameter_m)
            found_form.addRow("Ankrupoldi Ø (m)", self.foundation_bolt_diameter)

            self.foundation_bolt_embed = self._double_spin(0.05, 0.50, 0.01, initial.foundation_bolt_embed_m)
            found_form.addRow("Poldi süvistus (m)", self.foundation_bolt_embed)

            self.foundation_bolt_protrusion = self._double_spin(0.02, 0.30, 0.01, initial.foundation_bolt_protrusion_m)
            found_form.addRow("Poldi väljaulatumine (m)", self.foundation_bolt_protrusion)

            self.foundation_strip_width = self._double_spin(0.10, 1.0, 0.01, initial.foundation_strip_width_m)
            found_form.addRow("Lindi laius (m)", self.foundation_strip_width)
            self.foundation_strip_depth = self._double_spin(0.10, 1.5, 0.01, initial.foundation_strip_depth_m)
            found_form.addRow("Lindi sügavus (m)", self.foundation_strip_depth)

            self.foundation_pier_diameter = self._double_spin(0.10, 1.0, 0.01, initial.foundation_pier_diameter_m)
            found_form.addRow("Samba Ø (m)", self.foundation_pier_diameter)
            self.foundation_pier_depth = self._double_spin(0.20, 2.0, 0.01, initial.foundation_pier_depth_m)
            found_form.addRow("Samba sügavus (m)", self.foundation_pier_depth)

            found_lay.addWidget(self.found_detail_group)
            found_lay.addStretch(1)
            tabs.addTab(found_page, "Vundament")

            # ---------------------------------------------------------------
            # Tab 7: Analysis & Export
            # ---------------------------------------------------------------
            anal_page = widgets.QWidget()
            anal_lay = widgets.QVBoxLayout(anal_page)

            # Loads
            load_group = widgets.QGroupBox("Koormusarvutused (Eurocode)")
            load_form = widgets.QFormLayout(load_group)

            self.generate_loads = widgets.QCheckBox("Arvuta koormused")
            self.generate_loads.setChecked(initial.generate_loads)
            load_form.addRow(self.generate_loads)

            self.load_wind_speed = self._double_spin(0.0, 60.0, 1.0, initial.load_wind_speed_ms)
            load_form.addRow("Tuule kiirus (m/s)", self.load_wind_speed)

            self.load_wind_terrain = widgets.QComboBox()
            for cat in ("0", "I", "II", "III", "IV"):
                label_map = {
                    "0": "0 — avameri",
                    "I": "I — rannik / lagedad",
                    "II": "II — põllumaa (tavaline)",
                    "III": "III — eeslinn / mets",
                    "IV": "IV — linnakeskus",
                }
                self.load_wind_terrain.addItem(label_map.get(cat, cat), cat)
            idx_wt = self.load_wind_terrain.findData(initial.load_wind_terrain)
            if idx_wt >= 0:
                self.load_wind_terrain.setCurrentIndex(idx_wt)
            load_form.addRow("Maastikukategooria", self.load_wind_terrain)

            self.load_wind_direction = self._double_spin(-360.0, 360.0, 1.0, initial.load_wind_direction_deg)
            load_form.addRow("Tuule suund (°)", self.load_wind_direction)

            self.load_snow_zone = widgets.QComboBox()
            for z in ("I", "II", "III", "IV", "V"):
                kn_map = {"I": "1.0", "II": "1.5", "III": "2.0", "IV": "2.5", "V": "3.0"}
                self.load_snow_zone.addItem(f"{z} — {kn_map[z]} kN/m²", z)
            idx_sz = self.load_snow_zone.findData(initial.load_snow_zone)
            if idx_sz >= 0:
                self.load_snow_zone.setCurrentIndex(idx_sz)
            load_form.addRow("Lumetsooon", self.load_snow_zone)

            self.load_snow_exposure = self._double_spin(0.0, 2.0, 0.1, initial.load_snow_exposure)
            load_form.addRow("Lume Ce (avatus)", self.load_snow_exposure)

            self.load_snow_thermal = self._double_spin(0.0, 2.0, 0.1, initial.load_snow_thermal)
            load_form.addRow("Lume Ct (soojus)", self.load_snow_thermal)

            self.generate_structural_check = widgets.QCheckBox("Konstruktsiooni kandevõime kontroll")
            self.generate_structural_check.setChecked(
                getattr(initial, "generate_structural_check", False)
            )
            load_form.addRow(self.generate_structural_check)

            anal_lay.addWidget(load_group)

            # Production / Costing / Spreadsheets
            output_group = widgets.QGroupBox("Väljundid")
            output_form = widgets.QFormLayout(output_group)

            self.generate_production = widgets.QCheckBox("Tootmisandmed (lõikelehted, saetabel, plaadid)")
            self.generate_production.setChecked(initial.generate_production)
            output_form.addRow(self.generate_production)

            self.generate_cnc_export = widgets.QCheckBox("CNC/STEP eksport (STEP failid + lõiketabel)")
            self.generate_cnc_export.setChecked(
                getattr(initial, "generate_cnc_export", False)
            )
            output_form.addRow(self.generate_cnc_export)

            self.generate_techdraw = widgets.QCheckBox("TechDraw joonised (PDF)")
            self.generate_techdraw.setChecked(
                getattr(initial, "generate_techdraw", False)
            )
            output_form.addRow(self.generate_techdraw)

            # --- TechDraw detail panel (visible when techdraw enabled) ---
            techdraw_group = widgets.QGroupBox("TechDraw seaded")
            techdraw_form = widgets.QFormLayout(techdraw_group)

            self.techdraw_page_format = widgets.QComboBox()
            for fmt in ("A2", "A3", "A4"):
                self.techdraw_page_format.addItem(fmt, fmt)
            td_fmt_idx = self.techdraw_page_format.findData(
                getattr(initial, "techdraw_page_format", "A3")
            )
            if td_fmt_idx >= 0:
                self.techdraw_page_format.setCurrentIndex(td_fmt_idx)
            techdraw_form.addRow("Lehe formaat", self.techdraw_page_format)

            self.techdraw_scale = self._double_spin(0.001, 1.0, 0.005,
                getattr(initial, "techdraw_scale", 0.02))
            self.techdraw_scale.setDecimals(3)
            techdraw_form.addRow("Mõõtkava (nt 0.02 = 1:50)", self.techdraw_scale)

            self.techdraw_views = widgets.QComboBox()
            for mode, label in (
                ("all", "Kõik (ülevaade + osad + sõlmed)"),
                ("overview", "Ülevaade"),
                ("parts", "Osade joonised"),
                ("nodes", "Sõlmede detailid"),
            ):
                self.techdraw_views.addItem(label, mode)
            td_view_idx = self.techdraw_views.findData(
                getattr(initial, "techdraw_views", "all")
            )
            if td_view_idx >= 0:
                self.techdraw_views.setCurrentIndex(td_view_idx)
            techdraw_form.addRow("Vaated", self.techdraw_views)

            self.techdraw_project_name = widgets.QLineEdit(
                getattr(initial, "techdraw_project_name", ""))
            self.techdraw_project_name.setPlaceholderText("Projekti nimi (tiitelplokk)")
            techdraw_form.addRow("Projekti nimi", self.techdraw_project_name)

            self.techdraw_version = widgets.QLineEdit(
                getattr(initial, "techdraw_version", ""))
            self.techdraw_version.setPlaceholderText("Versioon / revisjon")
            techdraw_form.addRow("Versioon", self.techdraw_version)

            techdraw_group.setVisible(
                getattr(initial, "generate_techdraw", False)
            )
            output_form.addRow(techdraw_group)

            self.generate_assembly_guide = widgets.QCheckBox(
                "Montaažijuhised (SVG joonised + ajahinnang)")
            self.generate_assembly_guide.setChecked(
                getattr(initial, "generate_assembly_guide", False)
            )
            output_form.addRow(self.generate_assembly_guide)

            # --- Assembly detail panel (visible when assembly enabled) ---
            assembly_group = widgets.QGroupBox("Montaaži seaded")
            assembly_form = widgets.QFormLayout(assembly_group)

            self.assembly_time_per_strut = self._double_spin(
                1.0, 120.0, 1.0,
                getattr(initial, "assembly_time_per_strut_min", 15.0))
            assembly_form.addRow("Aeg lati kohta (min)", self.assembly_time_per_strut)

            self.assembly_time_per_node = self._double_spin(
                1.0, 120.0, 1.0,
                getattr(initial, "assembly_time_per_node_min", 10.0))
            assembly_form.addRow("Aeg sõlme kohta (min)", self.assembly_time_per_node)

            self.assembly_time_per_panel = self._double_spin(
                1.0, 120.0, 1.0,
                getattr(initial, "assembly_time_per_panel_min", 20.0))
            assembly_form.addRow("Aeg paneeli kohta (min)", self.assembly_time_per_panel)

            self.assembly_workers = widgets.QSpinBox()
            self.assembly_workers.setRange(1, 20)
            self.assembly_workers.setValue(
                int(getattr(initial, "assembly_workers", 2)))
            assembly_form.addRow("Meeskonna suurus", self.assembly_workers)

            assembly_group.setVisible(
                getattr(initial, "generate_assembly_guide", False)
            )
            output_form.addRow(assembly_group)

            self.generate_costing = widgets.QCheckBox("Kuluarvestus ja BOM")
            self.generate_costing.setChecked(initial.generate_costing)
            output_form.addRow(self.generate_costing)

            # --- Costing detail panel (visible when costing enabled) ---
            costing_group = widgets.QGroupBox("Kuluarvestuse seaded")
            costing_form = widgets.QFormLayout(costing_group)

            self.costing_currency = widgets.QComboBox()
            for cur_code in ("EUR", "USD", "GBP"):
                self.costing_currency.addItem(cur_code, cur_code)
            cur_idx = self.costing_currency.findData(
                getattr(initial.costing, "currency", "EUR")
            )
            if cur_idx >= 0:
                self.costing_currency.setCurrentIndex(cur_idx)
            costing_form.addRow("Valuuta", self.costing_currency)

            self.waste_timber_pct = self._double_spin(0.0, 50.0, 1.0,
                getattr(initial.costing, "waste_timber_pct", 10.0))
            costing_form.addRow("Puidu praak %", self.waste_timber_pct)

            self.waste_covering_pct = self._double_spin(0.0, 50.0, 1.0,
                getattr(initial.costing, "waste_covering_pct", 8.0))
            costing_form.addRow("Katte praak %", self.waste_covering_pct)

            self.overhead_pct = self._double_spin(0.0, 100.0, 1.0,
                getattr(initial.costing, "overhead_pct", 0.0))
            costing_form.addRow("Üldkulude juurdehindlus %", self.overhead_pct)

            self.timber_price_per_m = self._double_spin(0.0, 999.0, 0.5,
                getattr(initial.costing, "timber_price_per_m", 0.0))
            self.timber_price_per_m.setSpecialValueText("kataloogist")
            costing_form.addRow("Puidu hind €/jm (0=kataloog)", self.timber_price_per_m)

            self.covering_price_per_m2 = self._double_spin(0.0, 999.0, 1.0,
                getattr(initial.costing, "covering_price_per_m2", 0.0))
            self.covering_price_per_m2.setSpecialValueText("kataloogist")
            costing_form.addRow("Katte hind €/m² (0=kataloog)", self.covering_price_per_m2)

            self.labor_install_eur_h = self._double_spin(0.0, 999.0, 5.0,
                getattr(initial.costing, "labor_install_eur_h", 0.0))
            costing_form.addRow("Paigaldus €/h", self.labor_install_eur_h)

            self.estimated_install_hours = self._double_spin(0.0, 9999.0, 10.0,
                getattr(initial.costing, "estimated_install_hours", 0.0))
            costing_form.addRow("Paigaldusaeg h", self.estimated_install_hours)

            self.labor_cnc_eur_h = self._double_spin(0.0, 999.0, 5.0,
                getattr(initial.costing, "labor_cnc_eur_h", 0.0))
            costing_form.addRow("CNC tööaeg €/h", self.labor_cnc_eur_h)

            self.estimated_cnc_hours = self._double_spin(0.0, 9999.0, 1.0,
                getattr(initial.costing, "estimated_cnc_hours", 0.0))
            costing_form.addRow("CNC aeg h", self.estimated_cnc_hours)

            self.price_catalogue_path = widgets.QLineEdit(
                getattr(initial.costing, "price_catalogue_path", ""))
            self.price_catalogue_path.setPlaceholderText("hinnakataloogi JSON faili tee (valikuline)")
            costing_form.addRow("Hinnakataloog", self.price_catalogue_path)

            costing_group.setVisible(initial.generate_costing)
            output_form.addRow(costing_group)

            self.generate_spreadsheets = widgets.QCheckBox("FreeCAD tabelid (parameetrid + osaloend)")
            self.generate_spreadsheets.setChecked(initial.generate_spreadsheets)
            output_form.addRow(self.generate_spreadsheets)

            anal_lay.addWidget(output_group)
            anal_lay.addStretch(1)
            tabs.addTab(anal_page, "Analüüs")

            # ---------------------------------------------------------------
            # Tab 8: Multi-dome (Mitmikkuppel)
            # ---------------------------------------------------------------
            mdome_page = widgets.QWidget()
            mdome_lay = widgets.QVBoxLayout(mdome_page)

            self.multi_dome_enabled = widgets.QCheckBox("Mitmikkuppel projekt")
            self.multi_dome_enabled.setChecked(
                getattr(initial, "multi_dome_enabled", False)
            )
            mdome_lay.addWidget(self.multi_dome_enabled)

            mdome_group = widgets.QGroupBox("Kuplite paigutus")
            mdome_form = widgets.QFormLayout(mdome_group)

            self.dome_instances_json = widgets.QTextEdit()
            self.dome_instances_json.setPlaceholderText(
                '[{"label": "Anneks", "offset_x_m": 8.0, "offset_y_m": 0.0, '
                '"overrides": {"radius_m": 3.0}}]'
            )
            self.dome_instances_json.setMaximumHeight(100)
            self.dome_instances_json.setText(
                getattr(initial, "dome_instances_json", "[]")
            )
            mdome_form.addRow("Kuplid (JSON)", self.dome_instances_json)

            mdome_lay.addWidget(mdome_group)

            # Corridors
            corr_group = widgets.QGroupBox("Ühenduskäigud")
            corr_form = widgets.QFormLayout(corr_group)

            self.corridor_width = self._double_spin(
                0.5, 5.0, 0.1,
                getattr(initial, "corridor_width_m", 1.2))
            corr_form.addRow("Laius (m)", self.corridor_width)

            self.corridor_height = self._double_spin(
                1.5, 4.0, 0.1,
                getattr(initial, "corridor_height_m", 2.1))
            corr_form.addRow("Kõrgus (m)", self.corridor_height)

            self.corridor_wall_thickness = self._double_spin(
                0.05, 0.5, 0.01,
                getattr(initial, "corridor_wall_thickness_m", 0.15))
            corr_form.addRow("Seina paksus (m)", self.corridor_wall_thickness)

            self.corridor_material = widgets.QComboBox()
            for mat, lbl in (
                ("wood", "Puit"),
                ("steel", "Teras"),
                ("glass", "Klaas"),
            ):
                self.corridor_material.addItem(lbl, mat)
            cm_idx = self.corridor_material.findData(
                getattr(initial, "corridor_material", "wood")
            )
            if cm_idx >= 0:
                self.corridor_material.setCurrentIndex(cm_idx)
            corr_form.addRow("Materjal", self.corridor_material)

            self.corridor_definitions_json = widgets.QTextEdit()
            self.corridor_definitions_json.setPlaceholderText(
                '[{"from_dome": 0, "to_dome": 1}]'
            )
            self.corridor_definitions_json.setMaximumHeight(80)
            self.corridor_definitions_json.setText(
                getattr(initial, "corridor_definitions_json", "[]")
            )
            corr_form.addRow("Käigud (JSON)", self.corridor_definitions_json)

            mdome_lay.addWidget(corr_group)

            # Merge options
            merge_group = widgets.QGroupBox("Ühendamine")
            merge_form = widgets.QFormLayout(merge_group)

            self.merge_foundation = widgets.QCheckBox("Ühine vundamendiplaan")
            self.merge_foundation.setChecked(
                getattr(initial, "merge_foundation", True)
            )
            merge_form.addRow(self.merge_foundation)

            self.merge_bom = widgets.QCheckBox("Ühine BOM")
            self.merge_bom.setChecked(
                getattr(initial, "merge_bom", True)
            )
            merge_form.addRow(self.merge_bom)

            mdome_lay.addWidget(merge_group)

            mdome_lay.addStretch(1)
            tabs.addTab(mdome_page, "Mitmikkuppel")

            # ---------------------------------------------------------------
            # Sync logic
            # ---------------------------------------------------------------
            def _sync_state() -> None:
                # Base wall ↔ entry porch are mutually exclusive.
                bw = bool(self.generate_base_wall.isChecked())
                ep = bool(self.generate_entry_porch.isChecked())
                if bw and ep:
                    self.generate_base_wall.blockSignals(True)
                    self.generate_base_wall.setChecked(False)
                    self.generate_base_wall.blockSignals(False)
                    bw = False
                self.generate_base_wall.setEnabled(not ep)
                self.generate_entry_porch.setEnabled(not bw)

                # Wall detail fields
                for w in (self.base_wall_height, self.base_wall_thickness):
                    w.setEnabled(bw)

                # Porch detail fields
                for w in (self.porch_depth, self.porch_width, self.porch_height,
                          self.porch_member, self.porch_glass):
                    w.setEnabled(ep)

                # Door fields relevant when either wall or porch is active
                door_on = bw or ep
                for w in (self.door_width, self.door_height,
                          self.auto_door_angle, self.door_clearance):
                    w.setEnabled(door_on)
                self.door_angle.setEnabled(door_on and not self.auto_door_angle.isChecked())

                # Mode-dependent controls
                struts_mode = bool(self.mode_struts.isChecked())
                frames_mode = bool(self.mode_frames.isChecked())
                self.strut_group.setEnabled(struts_mode)

                # Node-fit mode sub-controls
                nf_mode = str(self.node_fit_mode.currentData() or "planar")
                self.node_fit_taper_ratio.setEnabled(nf_mode == "tapered")
                # Plane mode and separation planes only relevant for planar/tapered
                planar_ish = nf_mode in ("planar", "tapered")
                self.node_fit_plane.setEnabled(planar_ish)
                self.node_fit_separation.setEnabled(planar_ish)

                # Advanced strut settings — enable only when struts mode active
                self.advanced_strut_group.setEnabled(struts_mode)

                frame_on = frames_mode
                for w in (self.frame_inset, self.frame_width, self.frame_height):
                    w.setEnabled(frame_on)

                # Node connector details
                nc = bool(self.generate_node_connectors.isChecked())
                self.conn_detail_group.setEnabled(nc)

                # Foundation details
                fd = bool(self.generate_foundation.isChecked())
                self.found_detail_group.setEnabled(fd)

                # Foundation type-dependent fields
                ft = str(self.foundation_type.currentData() or "strip")
                is_strip = (ft == "strip")
                is_point = (ft == "point")
                self.foundation_strip_width.setEnabled(fd and is_strip)
                self.foundation_strip_depth.setEnabled(fd and is_strip)
                self.foundation_pier_diameter.setEnabled(fd and is_point)
                self.foundation_pier_depth.setEnabled(fd and is_point)

                # Weather details
                ww = bool(self.generate_weather.isChecked())
                self.gasket_type.setEnabled(ww)

                # Ventilation details
                vent = bool(self.generate_ventilation.isChecked())
                for w in (self.ventilation_mode, self.ventilation_target,
                          self.ventilation_apex_count, self.ventilation_ring_count,
                          self.ventilation_ring_height):
                    w.setEnabled(vent)

                # Load calc details
                lc = bool(self.generate_loads.isChecked())
                for w in (self.load_wind_speed, self.load_wind_terrain,
                          self.load_wind_direction, self.load_snow_zone,
                          self.load_snow_exposure, self.load_snow_thermal,
                          self.generate_structural_check):
                    w.setEnabled(lc)

                # Costing details
                cc = bool(self.generate_costing.isChecked())
                costing_group.setVisible(cc)

                # TechDraw details
                td = bool(self.generate_techdraw.isChecked())
                techdraw_group.setVisible(td)

                # Assembly guide details
                ag = bool(self.generate_assembly_guide.isChecked())
                assembly_group.setVisible(ag)

                # Skylight details
                sk = bool(self.generate_skylights.isChecked())
                for w in (self.skylight_count, self.skylight_position,
                          self.skylight_glass_thickness, self.skylight_frame_width,
                          self.skylight_hinge_side, self.skylight_material):
                    w.setEnabled(sk)

                # Riser wall details
                rw = bool(self.generate_riser_wall.isChecked())
                for w in (self.riser_height, self.riser_thickness,
                          self.riser_material, self.riser_connection_type,
                          self.riser_door_integration, self.riser_stud_spacing,
                          self.riser_segments):
                    w.setEnabled(rw)

                # Multi-dome details
                md = bool(self.multi_dome_enabled.isChecked())
                mdome_group.setEnabled(md)
                corr_group.setEnabled(md)
                merge_group.setEnabled(md)

            # Connect signals
            for sig in (
                self.mode_struts.toggled,
                self.mode_panels.toggled,
                self.mode_frames.toggled,
                self.generate_base_wall.toggled,
                self.generate_entry_porch.toggled,
                self.auto_door_angle.toggled,
                self.generate_node_connectors.toggled,
                self.generate_foundation.toggled,
                self.generate_weather.toggled,
                self.generate_ventilation.toggled,
                self.generate_loads.toggled,
                self.generate_costing.toggled,
                self.generate_techdraw.toggled,
                self.generate_assembly_guide.toggled,
                self.generate_skylights.toggled,
                self.generate_riser_wall.toggled,
                self.multi_dome_enabled.toggled,
            ):
                sig.connect(_sync_state)
            self.foundation_type.currentIndexChanged.connect(_sync_state)
            self.node_fit_mode.currentIndexChanged.connect(_sync_state)

            _sync_state()

            # ---------------------------------------------------------------
            # Preset save / load
            # ---------------------------------------------------------------
            preset_row = widgets.QHBoxLayout()
            btn_save_preset = widgets.QPushButton("Salvesta preset…")
            btn_load_preset = widgets.QPushButton("Lae preset…")
            preset_row.addWidget(btn_save_preset)
            preset_row.addWidget(btn_load_preset)
            preset_row.addStretch(1)
            root.addLayout(preset_row)

            def _save_preset() -> None:
                path, _ = widgets.QFileDialog.getSaveFileName(
                    self, "Salvesta preset", "", "JSON failid (*.json)"
                )
                if not path:
                    return
                try:
                    p = self.to_params()
                    data = p.to_dict()
                    # Remove non-serialisable entries
                    data.pop("materials", None)
                    Path(path).write_text(json.dumps(data, indent=2, default=str))
                    logging.info("Preset saved to %s", path)
                except Exception as exc:
                    logging.warning("Preset save failed: %s", exc)
                    widgets.QMessageBox.warning(
                        self, "Viga", f"Preseti salvestamine ebaõnnestus:\n{exc}"
                    )

            def _load_preset() -> None:
                path, _ = widgets.QFileDialog.getOpenFileName(
                    self, "Lae preset", "", "JSON failid (*.json)"
                )
                if not path:
                    return
                try:
                    raw = json.loads(Path(path).read_text(encoding="utf-8"))
                    # Apply loaded values to the widgets
                    _apply_preset(raw)
                    logging.info("Preset loaded from %s", path)
                except Exception as exc:
                    logging.warning("Preset load failed: %s", exc)
                    widgets.QMessageBox.warning(
                        self, "Viga", f"Preseti laadimine ebaõnnestus:\n{exc}"
                    )

            def _apply_preset(data: dict) -> None:
                """Push a dict of parameter values into the dialog widgets."""
                _set_spin = lambda w, k: w.setValue(float(data[k])) if k in data else None
                _set_check = lambda w, k: w.setChecked(bool(data[k])) if k in data else None
                _set_combo = lambda w, k: (
                    w.setCurrentIndex(max(0, w.findData(data[k])))
                    if k in data else None
                )
                # Geometry
                _set_spin(self.radius, "radius_m")
                _set_spin(self.frequency, "frequency")
                _set_spin(self.truncation, "truncation_ratio")
                _set_spin(self.segment, "hemisphere_ratio")
                # Structure
                _set_spin(self.stock_width, "stock_width_m")
                _set_spin(self.stock_height, "stock_height_m")
                _set_spin(self.kerf, "kerf_m")
                _set_spin(self.clearance, "clearance_m")
                _set_combo(self.material, "material")
                _set_check(self.use_bevels, "use_bevels")
                _set_check(self.use_truncation, "use_truncation")
                # Node-fit
                _set_combo(self.node_fit_mode, "node_fit_mode")
                _set_spin(self.node_fit_taper_ratio, "node_fit_taper_ratio")
                _set_combo(self.node_fit_plane, "node_fit_plane_mode")
                _set_spin(self.node_fit_extension, "node_fit_extension_m")
                _set_check(self.node_fit_separation, "node_fit_use_separation_planes")
                _set_check(self.split_struts_per_panel, "split_struts_per_panel")
                # Advanced strut
                _set_combo(self.strut_profile, "strut_profile")
                _set_combo(self.cap_blend_mode, "cap_blend_mode")
                _set_spin(self.bevel_fillet_radius, "bevel_fillet_radius_m")
                _set_spin(self.min_wedge_angle, "min_wedge_angle_deg")
                _set_spin(self.cap_length_factor, "cap_length_factor")
                _set_spin(self.max_cap_ratio, "max_cap_ratio")
                _set_check(self.generate_belt_cap, "generate_belt_cap")
                # Covering
                _set_combo(self.covering_type, "covering_type")
                _set_spin(self.covering_delta_t, "covering_delta_t_k")
                _set_combo(self.covering_profile, "covering_profile_type")
                _set_spin(self.frame_inset, "panel_frame_inset_m")
                _set_spin(self.frame_width, "panel_frame_profile_width_m")
                _set_spin(self.frame_height, "panel_frame_profile_height_m")
                _set_spin(self.glass_thickness, "glass_thickness_m")
                _set_spin(self.glass_gap, "glass_gap_m")
                # Skylights
                _set_check(self.generate_skylights, "generate_skylights")
                _set_spin(self.skylight_count, "skylight_count")
                _set_combo(self.skylight_position, "skylight_position")
                _set_spin(self.skylight_glass_thickness, "skylight_glass_thickness_m")
                _set_spin(self.skylight_frame_width, "skylight_frame_width_m")
                _set_combo(self.skylight_hinge_side, "skylight_hinge_side")
                _set_combo(self.skylight_material, "skylight_material")
                # Riser wall
                _set_check(self.generate_riser_wall, "generate_riser_wall")
                _set_spin(self.riser_height, "riser_height_m")
                _set_spin(self.riser_thickness, "riser_thickness_m")
                _set_combo(self.riser_material, "riser_material")
                _set_combo(self.riser_connection_type, "riser_connection_type")
                _set_check(self.riser_door_integration, "riser_door_integration")
                _set_spin(self.riser_stud_spacing, "riser_stud_spacing_m")
                _set_spin(self.riser_segments, "riser_segments")
                # Costing
                _set_check(self.generate_costing, "generate_costing")
                _set_combo(self.costing_currency, "currency")
                _set_spin(self.waste_timber_pct, "waste_timber_pct")
                _set_spin(self.waste_covering_pct, "waste_covering_pct")
                _set_spin(self.overhead_pct, "overhead_pct")
                _set_spin(self.timber_price_per_m, "timber_price_per_m")
                _set_spin(self.covering_price_per_m2, "covering_price_per_m2")
                _set_spin(self.labor_install_eur_h, "labor_install_eur_h")
                _set_spin(self.estimated_install_hours, "estimated_install_hours")
                _set_spin(self.labor_cnc_eur_h, "labor_cnc_eur_h")
                _set_spin(self.estimated_cnc_hours, "estimated_cnc_hours")
                if "price_catalogue_path" in data:
                    self.price_catalogue_path.setText(str(data["price_catalogue_path"]))
                # Multi-dome
                _set_check(self.multi_dome_enabled, "multi_dome_enabled")
                if "dome_instances_json" in data:
                    self.dome_instances_json.setText(str(data["dome_instances_json"]))
                _set_spin(self.corridor_width, "corridor_width_m")
                _set_spin(self.corridor_height, "corridor_height_m")
                _set_spin(self.corridor_wall_thickness, "corridor_wall_thickness_m")
                _set_combo(self.corridor_material, "corridor_material")
                if "corridor_definitions_json" in data:
                    self.corridor_definitions_json.setText(str(data["corridor_definitions_json"]))
                _set_check(self.merge_foundation, "merge_foundation")
                _set_check(self.merge_bom, "merge_bom")
                # Re-sync
                _sync_state()

            btn_save_preset.clicked.connect(_save_preset)
            btn_load_preset.clicked.connect(_load_preset)

            # ---------------------------------------------------------------
            # Buttons
            # ---------------------------------------------------------------
            buttons = widgets.QDialogButtonBox(
                widgets.QDialogButtonBox.Ok | widgets.QDialogButtonBox.Cancel
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            root.addWidget(buttons)

        # -- Helpers ---------------------------------------------------------

        def _double_spin(self, minimum: float, maximum: float, step: float, value: float):
            spin = widgets.QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(4)
            spin.setSingleStep(step)
            spin.setValue(value)
            return spin

        def _int_spin(self, minimum: int, maximum: int, value: int):
            spin = widgets.QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            return spin

        def to_params(self) -> DomeParameters:
            generate_struts = bool(self.mode_struts.isChecked())
            generate_panel_frames = bool(self.mode_frames.isChecked())
            generate_panel_faces = bool(self.mode_panels.isChecked())
            panels_only = not generate_struts and (generate_panel_faces or generate_panel_frames)

            params = DomeParameters(
                # Geometry
                radius_m=float(self.radius.value()),
                frequency=int(self.frequency.value()),
                truncation_ratio=float(self.truncation.value()),
                hemisphere_ratio=float(self.segment.value()),
                # Structure
                stock_width_m=float(self.stock_width.value()),
                stock_height_m=float(self.stock_height.value()),
                kerf_m=float(self.kerf.value()),
                clearance_m=float(self.clearance.value()),
                material=self.material.currentData(),
                use_bevels=self.use_bevels.isChecked(),
                use_truncation=self.use_truncation.isChecked(),
                panels_only=panels_only,
                generate_struts=generate_struts,
                node_fit_plane_mode=str(self.node_fit_plane.currentData()),
                node_fit_use_separation_planes=bool(self.node_fit_separation.isChecked()),
                node_fit_extension_m=float(self.node_fit_extension.value()),
                node_fit_mode=str(self.node_fit_mode.currentData()),
                node_fit_taper_ratio=float(self.node_fit_taper_ratio.value()),
                split_struts_per_panel=bool(self.split_struts_per_panel.isChecked()),
                # Advanced strut settings
                strut_profile=str(self.strut_profile.currentData()),
                cap_blend_mode=str(self.cap_blend_mode.currentData()),
                bevel_fillet_radius_m=float(self.bevel_fillet_radius.value()),
                min_wedge_angle_deg=float(self.min_wedge_angle.value()),
                cap_length_factor=float(self.cap_length_factor.value()),
                max_cap_ratio=float(self.max_cap_ratio.value()),
                generate_belt_cap=bool(self.generate_belt_cap.isChecked()),
                # Node connectors
                generate_node_connectors=bool(self.generate_node_connectors.isChecked()),
                node_connector_type=str(self.node_connector_type.currentData()),
                node_connector_thickness_m=float(self.node_connector_thickness.value()),
                node_connector_bolt_diameter_m=float(self.node_connector_bolt_diameter.value()),
                node_connector_bolt_length_m=float(self.node_connector_bolt_length.value()),
                node_connector_washer_diameter_m=float(self.node_connector_washer_diameter.value()),
                node_connector_bolt_offset_m=float(self.node_connector_bolt_offset.value()),
                node_connector_lap_extension_m=float(self.node_connector_lap_extension.value()),
                # Covering
                generate_panel_faces=generate_panel_faces,
                generate_panel_frames=generate_panel_frames,
                panel_frame_inset_m=float(self.frame_inset.value()),
                panel_frame_profile_width_m=float(self.frame_width.value()),
                panel_frame_profile_height_m=float(self.frame_height.value()),
                glass_thickness_m=float(self.glass_thickness.value()),
                glass_gap_m=float(self.glass_gap.value()),
                covering_type=str(self.covering_type.currentData()),
                covering_thickness_m=float(self.covering_thickness.value()),
                covering_gap_m=float(self.covering_gap.value()),
                covering_delta_t_k=float(self.covering_delta_t.value()),
                covering_profile_type=str(self.covering_profile.currentData()),
                # Weather
                generate_weather=bool(self.generate_weather.isChecked()),
                gasket_type=str(self.gasket_type.currentData()),
                # Skylights
                generate_skylights=bool(self.generate_skylights.isChecked()),
                skylight_count=int(self.skylight_count.value()),
                skylight_position=str(self.skylight_position.currentData()),
                skylight_glass_thickness_m=float(self.skylight_glass_thickness.value()),
                skylight_frame_width_m=float(self.skylight_frame_width.value()),
                skylight_hinge_side=str(self.skylight_hinge_side.currentData()),
                skylight_material=str(self.skylight_material.currentData()),
                # Openings
                generate_base_wall=bool(self.generate_base_wall.isChecked()),
                base_wall_height_m=float(self.base_wall_height.value()),
                base_wall_thickness_m=float(self.base_wall_thickness.value()),
                door_width_m=float(self.door_width.value()),
                door_height_m=float(self.door_height.value()),
                door_angle_deg=float(self.door_angle.value()),
                auto_door_angle=bool(self.auto_door_angle.isChecked()),
                door_clearance_m=float(self.door_clearance.value()),
                generate_entry_porch=bool(self.generate_entry_porch.isChecked()),
                porch_depth_m=float(self.porch_depth.value()),
                porch_width_m=float(self.porch_width.value()),
                porch_height_m=float(self.porch_height.value()),
                porch_member_size_m=float(self.porch_member.value()),
                porch_glass_thickness_m=float(self.porch_glass.value()),
                # Ventilation
                generate_ventilation=bool(self.generate_ventilation.isChecked()),
                ventilation_mode=str(self.ventilation_mode.currentData()),
                ventilation_target_ratio=float(self.ventilation_target.value()),
                ventilation_apex_count=int(self.ventilation_apex_count.value()),
                ventilation_ring_count=int(self.ventilation_ring_count.value()),
                ventilation_ring_height_ratio=float(self.ventilation_ring_height.value()),
                # Riser wall
                generate_riser_wall=bool(self.generate_riser_wall.isChecked()),
                riser_height_m=float(self.riser_height.value()),
                riser_thickness_m=float(self.riser_thickness.value()),
                riser_material=str(self.riser_material.currentData()),
                riser_connection_type=str(self.riser_connection_type.currentData()),
                riser_door_integration=bool(self.riser_door_integration.isChecked()),
                riser_stud_spacing_m=float(self.riser_stud_spacing.value()),
                riser_segments=int(self.riser_segments.value()),
                # Foundation
                generate_foundation=bool(self.generate_foundation.isChecked()),
                foundation_type=str(self.foundation_type.currentData()),
                foundation_bolt_diameter_m=float(self.foundation_bolt_diameter.value()),
                foundation_bolt_embed_m=float(self.foundation_bolt_embed.value()),
                foundation_bolt_protrusion_m=float(self.foundation_bolt_protrusion.value()),
                foundation_strip_width_m=float(self.foundation_strip_width.value()),
                foundation_strip_depth_m=float(self.foundation_strip_depth.value()),
                foundation_pier_diameter_m=float(self.foundation_pier_diameter.value()),
                foundation_pier_depth_m=float(self.foundation_pier_depth.value()),
                # Analysis / export
                generate_loads=bool(self.generate_loads.isChecked()),
                generate_structural_check=bool(self.generate_structural_check.isChecked()),
                load_wind_speed_ms=float(self.load_wind_speed.value()),
                load_wind_terrain=str(self.load_wind_terrain.currentData()),
                load_wind_direction_deg=float(self.load_wind_direction.value()),
                load_snow_zone=str(self.load_snow_zone.currentData()),
                load_snow_exposure=float(self.load_snow_exposure.value()),
                load_snow_thermal=float(self.load_snow_thermal.value()),
                generate_production=bool(self.generate_production.isChecked()),
                generate_cnc_export=bool(self.generate_cnc_export.isChecked()),
                generate_techdraw=bool(self.generate_techdraw.isChecked()),
                techdraw_page_format=str(self.techdraw_page_format.currentData()),
                techdraw_scale=float(self.techdraw_scale.value()),
                techdraw_views=str(self.techdraw_views.currentData()),
                techdraw_project_name=str(self.techdraw_project_name.text()).strip(),
                techdraw_version=str(self.techdraw_version.text()).strip(),
                generate_assembly_guide=bool(self.generate_assembly_guide.isChecked()),
                assembly_time_per_strut_min=float(self.assembly_time_per_strut.value()),
                assembly_time_per_node_min=float(self.assembly_time_per_node.value()),
                assembly_time_per_panel_min=float(self.assembly_time_per_panel.value()),
                assembly_workers=int(self.assembly_workers.value()),
                generate_costing=bool(self.generate_costing.isChecked()),
                currency=str(self.costing_currency.currentData()),
                waste_timber_pct=float(self.waste_timber_pct.value()),
                waste_covering_pct=float(self.waste_covering_pct.value()),
                overhead_pct=float(self.overhead_pct.value()),
                timber_price_per_m=float(self.timber_price_per_m.value()),
                covering_price_per_m2=float(self.covering_price_per_m2.value()),
                labor_install_eur_h=float(self.labor_install_eur_h.value()),
                estimated_install_hours=float(self.estimated_install_hours.value()),
                labor_cnc_eur_h=float(self.labor_cnc_eur_h.value()),
                estimated_cnc_hours=float(self.estimated_cnc_hours.value()),
                price_catalogue_path=str(self.price_catalogue_path.text()).strip(),
                generate_spreadsheets=bool(self.generate_spreadsheets.isChecked()),
                # Multi-dome
                multi_dome_enabled=bool(self.multi_dome_enabled.isChecked()),
                dome_instances_json=self.dome_instances_json.toPlainText().strip() or "[]",
                corridor_width_m=float(self.corridor_width.value()),
                corridor_height_m=float(self.corridor_height.value()),
                corridor_wall_thickness_m=float(self.corridor_wall_thickness.value()),
                corridor_material=str(self.corridor_material.currentData()),
                corridor_definitions_json=self.corridor_definitions_json.toPlainText().strip() or "[]",
                merge_foundation=bool(self.merge_foundation.isChecked()),
                merge_bom=bool(self.merge_bom.isChecked()),
                # Materials passthrough
                materials=initial.materials,
            )
            params.validate()
            return params

    dialog = _ParameterDialog()
    exec_fn = getattr(dialog, "exec_", dialog.exec)
    result = exec_fn()
    if created_app:
        app.quit()
    if result != widgets.QDialog.Accepted:
        return None
    return dialog.to_params()
