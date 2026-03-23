"""FighterCAD – Tkinter-based parametric supersonic fighter design GUI."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from fightercad.parameters import (
    AircraftParams,
    MetaParams,
    FuselageParams,
    WingParams,
    BlendingParams,
    VerticalStabilizerParams,
    IntakeParams,
    ExhaustParams,
    AreaRuleParams,
)
from fightercad.config import load_config, save_config
from fightercad.aircraft import AircraftAssembler
from fightercad.aero.drag import compute_drag, DragResult
from fightercad.aero.lift import compute_CL_for_level_flight, compute_lift_curve_slope
from fightercad.aero.stability import compute_stability, StabilityResult
from fightercad.visualization.viewer3d import plot_aircraft_3d
from fightercad.visualization.plot2d import (
    plot_drag_polar,
    plot_area_distribution,
    plot_mach_sweep,
    plot_drag_breakdown,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "default_fighter.yaml"


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------
class ScrollableFrame(ttk.Frame):
    """Vertically scrollable frame using Canvas + Scrollbar."""

    def __init__(self, parent: tk.Widget, width: int = 320):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Mouse-wheel support
        self.inner.bind("<Enter>", self._bind_wheel)
        self.inner.bind("<Leave>", self._unbind_wheel)

    def _bind_wheel(self, _event: Any) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_wheel(self, _event: Any) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: Any) -> None:
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class ParamSlider(ttk.Frame):
    """Combined slider + entry for a single numeric parameter."""

    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        resolution: float = 0.01,
    ):
        super().__init__(parent)
        self._var = tk.DoubleVar(value=default)
        self._updating = False

        ttk.Label(self, text=label, width=22, anchor="w").pack(side="left")
        self.scale = ttk.Scale(
            self, from_=min_val, to=max_val, variable=self._var,
            orient="horizontal", length=120, command=self._on_scale,
        )
        self.scale.pack(side="left", padx=2)
        self.entry = ttk.Entry(self, width=8)
        self.entry.pack(side="left", padx=2)
        self.entry.insert(0, f"{default:.4g}")
        self.entry.bind("<Return>", self._on_entry)
        self.entry.bind("<FocusOut>", self._on_entry)
        self._resolution = resolution

    def _on_scale(self, _val: str) -> None:
        if self._updating:
            return
        self._updating = True
        v = round(self._var.get() / self._resolution) * self._resolution
        self.entry.delete(0, "end")
        self.entry.insert(0, f"{v:.4g}")
        self._updating = False

    def _on_entry(self, _event: Any) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            v = float(self.entry.get())
            self._var.set(v)
        except ValueError:
            pass
        self._updating = False

    def get(self) -> float:
        try:
            return float(self.entry.get())
        except ValueError:
            return self._var.get()

    def set(self, value: float) -> None:
        self._var.set(value)
        self.entry.delete(0, "end")
        self.entry.insert(0, f"{value:.4g}")


class ParamCombo(ttk.Frame):
    """Label + Combobox for categorical parameters."""

    def __init__(self, parent: tk.Widget, label: str, values: list[str], default: str):
        super().__init__(parent)
        ttk.Label(self, text=label, width=22, anchor="w").pack(side="left")
        self._var = tk.StringVar(value=default)
        self.combo = ttk.Combobox(self, textvariable=self._var, values=values, width=14, state="readonly")
        self.combo.pack(side="left", padx=2)

    def get(self) -> str:
        return self._var.get()

    def set(self, value: str) -> None:
        self._var.set(value)


class ParamCheck(ttk.Frame):
    """Label + Checkbutton for boolean parameters."""

    def __init__(self, parent: tk.Widget, label: str, default: bool):
        super().__init__(parent)
        self._var = tk.BooleanVar(value=default)
        ttk.Label(self, text=label, width=22, anchor="w").pack(side="left")
        ttk.Checkbutton(self, variable=self._var).pack(side="left")

    def get(self) -> bool:
        return self._var.get()

    def set(self, value: bool) -> None:
        self._var.set(value)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class FighterCADApp(tk.Tk):
    """Parametric supersonic fighter design application."""

    def __init__(self) -> None:
        super().__init__()
        self.title("FighterCAD – Parametric Supersonic Fighter Designer")
        self.geometry("1280x800")
        self.minsize(960, 600)

        self._widgets: dict[str, ParamSlider | ParamCombo | ParamCheck] = {}
        self._assembler: AircraftAssembler | None = None
        self._components: dict | None = None
        self._drag_result: DragResult | None = None
        self._stability_result: StabilityResult | None = None

        self._create_menu()
        self._create_layout()
        self._create_status_bar()

        # Load defaults
        if _DEFAULT_CONFIG.exists():
            try:
                self._apply_params(load_config(_DEFAULT_CONFIG))
            except Exception:
                pass

    # -- Menu --
    def _create_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="YAML Aç...", command=self._on_load_config)
        file_menu.add_command(label="YAML Kaydet...", command=self._on_save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Çıkış", command=self.quit)
        menubar.add_cascade(label="Dosya", menu=file_menu)

        export_menu = tk.Menu(menubar, tearoff=0)
        export_menu.add_command(label="STEP Export...", command=self._on_export_step)
        export_menu.add_command(label="STL Export...", command=self._on_export_stl)
        menubar.add_cascade(label="Export", menu=export_menu)

        catia_menu = tk.Menu(menubar, tearoff=0)
        catia_menu.add_command(label="CATIA Bağlan", command=self._on_catia_connect)
        catia_menu.add_command(label="CATIA Gönder", command=self._on_catia_send)
        menubar.add_cascade(label="CATIA", menu=catia_menu)

        self.config(menu=menubar)

    # -- Main layout --
    def _create_layout(self) -> None:
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        # Left: parameter panel
        self._scroll = ScrollableFrame(paned, width=340)
        paned.add(self._scroll, weight=0)
        self._create_param_panel(self._scroll.inner)

        # Right: notebook
        self._notebook = ttk.Notebook(paned)
        paned.add(self._notebook, weight=1)

        # Tab 1: 3D view
        self._tab_3d = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_3d, text="3D Görünüm")
        self._fig_3d: plt.Figure | None = None
        self._canvas_3d: FigureCanvasTkAgg | None = None

        # Tab 2: Aero analysis
        self._tab_aero = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_aero, text="Aerodinamik Analiz")
        self._aero_label = ttk.Label(self._tab_aero, text="Henüz analiz yapılmadı.", font=("TkDefaultFont", 10))
        self._aero_label.pack(pady=10)
        self._aero_plots_frame = ttk.Frame(self._tab_aero)
        self._aero_plots_frame.pack(fill="both", expand=True)

        # Tab 3: Mach sweep
        self._tab_mach = ttk.Frame(self._notebook)
        self._notebook.add(self._tab_mach, text="Mach Taraması")

    # -- Parameter panel --
    def _create_param_panel(self, parent: ttk.Frame) -> None:
        def section(title: str) -> ttk.LabelFrame:
            lf = ttk.LabelFrame(parent, text=title, padding=4)
            lf.pack(fill="x", padx=4, pady=3)
            return lf

        def slider(frame: ttk.LabelFrame, key: str, label: str, lo: float, hi: float, default: float, res: float = 0.01) -> ParamSlider:
            w = ParamSlider(frame, label, lo, hi, default, res)
            w.pack(fill="x")
            self._widgets[key] = w
            return w

        def combo(frame: ttk.LabelFrame, key: str, label: str, values: list[str], default: str) -> ParamCombo:
            w = ParamCombo(frame, label, values, default)
            w.pack(fill="x")
            self._widgets[key] = w
            return w

        def check(frame: ttk.LabelFrame, key: str, label: str, default: bool) -> ParamCheck:
            w = ParamCheck(frame, label, default)
            w.pack(fill="x")
            self._widgets[key] = w
            return w

        # --- Meta ---
        f = section("Uçuş Koşulu")
        slider(f, "meta.design_mach", "Tasarım Mach", 0.6, 3.0, 1.6, 0.05)
        slider(f, "meta.design_altitude_m", "İrtifa (m)", 0, 25000, 10000, 100)

        # --- Fuselage ---
        f = section("Gövde (Fuselage)")
        slider(f, "fuselage.length_m", "Uzunluk (m)", 8, 25, 15.0, 0.1)
        slider(f, "fuselage.max_diameter_m", "Maks Çap (m)", 0.8, 3.0, 1.8, 0.05)
        slider(f, "fuselage.nose_fineness_ratio", "Burun Fineness", 2.0, 6.0, 3.5, 0.1)
        combo(f, "fuselage.nose_profile", "Burun Profili", ["haack", "conical", "ogive", "power_series"], "haack")
        slider(f, "fuselage.haack_C", "Haack C", 0.0, 1.0, 0.0, 0.01)
        slider(f, "fuselage.aft_taper_ratio", "Arka Taper", 0.2, 1.0, 0.6, 0.05)
        combo(f, "fuselage.cross_section", "Kesit Tipi", ["circular", "elliptical", "rounded_rect"], "elliptical")
        slider(f, "fuselage.cross_section_aspect", "Kesit Oranı", 0.8, 1.5, 1.15, 0.05)

        # --- Wing ---
        f = section("Kanat (Wing)")
        slider(f, "wing.area_m2", "Alan (m²)", 15, 60, 28.0, 0.5)
        slider(f, "wing.aspect_ratio", "Aspect Ratio", 1.0, 5.0, 2.2, 0.1)
        slider(f, "wing.taper_ratio", "Taper Ratio", 0.0, 0.6, 0.10, 0.01)
        slider(f, "wing.leading_edge_sweep_deg", "LE Süpürme (°)", 30, 75, 55.0, 1.0)
        slider(f, "wing.trailing_edge_sweep_deg", "TE Süpürme (°)", -20, 20, -5.0, 1.0)
        slider(f, "wing.dihedral_deg", "Dihedral (°)", -10, 10, -3.0, 0.5)
        slider(f, "wing.twist_deg", "Twist (°)", -8, 4, -2.0, 0.5)
        airfoils = ["biconvex_3", "biconvex_4", "biconvex_5", "diamond_3", "diamond_4", "diamond_5", "naca64a004"]
        combo(f, "wing.root_airfoil", "Kök Airfoil", airfoils, "biconvex_5")
        combo(f, "wing.tip_airfoil", "Uç Airfoil", airfoils, "biconvex_3")
        slider(f, "wing.leading_edge_radius_mm", "LE Yarıçap (mm)", 0.5, 5.0, 1.5, 0.1)
        slider(f, "wing.wing_station_pct", "Kanat Konum (%L)", 0.3, 0.6, 0.45, 0.01)
        slider(f, "wing.incidence_deg", "İncidence (°)", -2, 5, 1.0, 0.5)

        # --- Blending ---
        f = section("Birleşim Hatları")
        slider(f, "blending.root_fillet_radius_mm", "Kök Fillet (mm)", 5, 60, 20.0, 1)
        slider(f, "blending.leading_edge_fillet_mm", "LE Fillet (mm)", 2, 30, 10.0, 1)
        slider(f, "blending.trailing_edge_fillet_mm", "TE Fillet (mm)", 2, 30, 10.0, 1)
        combo(f, "blending.blending_mode", "Mod", ["sharp", "smooth", "strake"], "smooth")
        slider(f, "blending.strake_length_m", "Strake Uzunluk (m)", 0, 3, 0.0, 0.1)
        slider(f, "blending.strake_sweep_deg", "Strake Süpürme (°)", 50, 85, 70.0, 1)
        slider(f, "blending.fairing_width_mm", "Fairing Gen. (mm)", 30, 300, 100.0, 5)
        slider(f, "blending.stabilizer_root_fillet_mm", "Stab Fillet (mm)", 5, 60, 30.0, 1)
        slider(f, "blending.intake_fuselage_fillet_mm", "Alık Fillet (mm)", 5, 40, 15.0, 1)

        # --- Vertical stabilizer ---
        f = section("Dikey Stabilizatör")
        slider(f, "vertical_stabilizer.area_m2", "Alan (m²)", 2, 10, 5.5, 0.1)
        slider(f, "vertical_stabilizer.aspect_ratio", "AR", 0.5, 2.5, 1.2, 0.1)
        slider(f, "vertical_stabilizer.sweep_deg", "Süpürme (°)", 30, 70, 50.0, 1)
        slider(f, "vertical_stabilizer.taper_ratio", "Taper", 0.1, 0.5, 0.25, 0.01)
        slider(f, "vertical_stabilizer.cant_deg", "Cant (°)", -15, 15, 0.0, 1)

        # --- Intake ---
        f = section("Hava Alığı")
        combo(f, "intake.intake_type", "Tip", ["side_mounted", "chin", "dorsal", "dsi"], "side_mounted")
        slider(f, "intake.capture_area_m2", "Yakalama Alanı (m²)", 0.1, 1.0, 0.45, 0.01)
        slider(f, "intake.ramp_angle_deg", "Rampa Açısı (°)", 3, 15, 7.0, 0.5)
        slider(f, "intake.lip_sweep_deg", "Dudak Süpürme (°)", 10, 50, 30.0, 1)
        slider(f, "intake.station_pct", "Konum (%L)", 0.15, 0.5, 0.30, 0.01)
        slider(f, "intake.boundary_layer_diverter_mm", "BL Ayırıcı (mm)", 20, 200, 80.0, 5)

        # --- Exhaust ---
        f = section("Egzoz Nozulu")
        combo(f, "exhaust.nozzle_type", "Tip", ["convergent_divergent", "convergent"], "convergent_divergent")
        slider(f, "exhaust.exit_diameter_m", "Çıkış Çap (m)", 0.4, 1.5, 0.85, 0.01)
        slider(f, "exhaust.throat_diameter_m", "Boğaz Çap (m)", 0.3, 1.2, 0.65, 0.01)
        slider(f, "exhaust.nozzle_length_m", "Nozul Uzunluk (m)", 0.5, 2.5, 1.2, 0.05)

        # --- Area rule ---
        f = section("Alan Kuralı (Area Rule)")
        check(f, "area_rule.enabled", "Aktif", True)
        slider(f, "area_rule.target_mach", "Hedef Mach", 0.8, 2.0, 1.2, 0.05)
        slider(f, "area_rule.waist_station_pct", "İnceltme Konum (%)", 0.3, 0.7, 0.55, 0.01)
        slider(f, "area_rule.smoothing_iterations", "Düzeltme İterasyon", 1, 20, 5, 1)

    # -- Status bar & buttons --
    def _create_status_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="bottom", padx=4, pady=4)

        self._status_var = tk.StringVar(value="Hazır.")
        ttk.Label(bar, textvariable=self._status_var).pack(side="left", padx=4)

        ttk.Button(bar, text="CATIA Gönder", command=self._on_catia_send).pack(side="right", padx=4)
        ttk.Button(bar, text="Analiz", command=self._on_analyze).pack(side="right", padx=4)
        ttk.Button(bar, text="Oluştur", command=self._on_generate).pack(side="right", padx=4)

    # -- Collect params from GUI --
    def _collect_params(self) -> AircraftParams:
        def g(key: str) -> Any:
            return self._widgets[key].get()

        return AircraftParams(
            meta=MetaParams(
                design_mach=g("meta.design_mach"),
                design_altitude_m=g("meta.design_altitude_m"),
            ),
            fuselage=FuselageParams(
                length_m=g("fuselage.length_m"),
                max_diameter_m=g("fuselage.max_diameter_m"),
                nose_fineness_ratio=g("fuselage.nose_fineness_ratio"),
                nose_profile=g("fuselage.nose_profile"),
                haack_C=g("fuselage.haack_C"),
                aft_taper_ratio=g("fuselage.aft_taper_ratio"),
                cross_section=g("fuselage.cross_section"),
                cross_section_aspect=g("fuselage.cross_section_aspect"),
            ),
            wing=WingParams(
                area_m2=g("wing.area_m2"),
                aspect_ratio=g("wing.aspect_ratio"),
                taper_ratio=g("wing.taper_ratio"),
                leading_edge_sweep_deg=g("wing.leading_edge_sweep_deg"),
                trailing_edge_sweep_deg=g("wing.trailing_edge_sweep_deg"),
                dihedral_deg=g("wing.dihedral_deg"),
                twist_deg=g("wing.twist_deg"),
                root_airfoil=g("wing.root_airfoil"),
                tip_airfoil=g("wing.tip_airfoil"),
                leading_edge_radius_mm=g("wing.leading_edge_radius_mm"),
                wing_station_pct=g("wing.wing_station_pct"),
                incidence_deg=g("wing.incidence_deg"),
            ),
            blending=BlendingParams(
                root_fillet_radius_mm=g("blending.root_fillet_radius_mm"),
                leading_edge_fillet_mm=g("blending.leading_edge_fillet_mm"),
                trailing_edge_fillet_mm=g("blending.trailing_edge_fillet_mm"),
                blending_mode=g("blending.blending_mode"),
                strake_length_m=g("blending.strake_length_m"),
                strake_sweep_deg=g("blending.strake_sweep_deg"),
                fairing_width_mm=g("blending.fairing_width_mm"),
                stabilizer_root_fillet_mm=g("blending.stabilizer_root_fillet_mm"),
                intake_fuselage_fillet_mm=g("blending.intake_fuselage_fillet_mm"),
            ),
            vertical_stabilizer=VerticalStabilizerParams(
                area_m2=g("vertical_stabilizer.area_m2"),
                aspect_ratio=g("vertical_stabilizer.aspect_ratio"),
                sweep_deg=g("vertical_stabilizer.sweep_deg"),
                taper_ratio=g("vertical_stabilizer.taper_ratio"),
                cant_deg=g("vertical_stabilizer.cant_deg"),
            ),
            intake=IntakeParams(
                intake_type=g("intake.intake_type"),
                capture_area_m2=g("intake.capture_area_m2"),
                ramp_angle_deg=g("intake.ramp_angle_deg"),
                lip_sweep_deg=g("intake.lip_sweep_deg"),
                station_pct=g("intake.station_pct"),
                boundary_layer_diverter_mm=g("intake.boundary_layer_diverter_mm"),
            ),
            exhaust=ExhaustParams(
                nozzle_type=g("exhaust.nozzle_type"),
                exit_diameter_m=g("exhaust.exit_diameter_m"),
                throat_diameter_m=g("exhaust.throat_diameter_m"),
                nozzle_length_m=g("exhaust.nozzle_length_m"),
            ),
            area_rule=AreaRuleParams(
                enabled=g("area_rule.enabled"),
                target_mach=g("area_rule.target_mach"),
                waist_station_pct=g("area_rule.waist_station_pct"),
                smoothing_iterations=int(g("area_rule.smoothing_iterations")),
            ),
        )

    def _apply_params(self, params: AircraftParams) -> None:
        """Push dataclass values into GUI widgets."""
        import dataclasses
        for section_name, section_obj in [
            ("meta", params.meta),
            ("fuselage", params.fuselage),
            ("wing", params.wing),
            ("blending", params.blending),
            ("vertical_stabilizer", params.vertical_stabilizer),
            ("intake", params.intake),
            ("exhaust", params.exhaust),
            ("area_rule", params.area_rule),
        ]:
            for fld in dataclasses.fields(section_obj):
                key = f"{section_name}.{fld.name}"
                if key in self._widgets:
                    self._widgets[key].set(getattr(section_obj, fld.name))

    # -- Actions --
    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)
        self.update_idletasks()

    def _on_generate(self) -> None:
        self._set_status("Geometri oluşturuluyor...")
        try:
            params = self._collect_params()
            self._assembler = AircraftAssembler(params)
            self._components = self._assembler.build()
            self._update_3d_view()
            self._set_status(f"Oluşturuldu – {len(self._components)} bileşen.")
        except Exception as exc:
            messagebox.showerror("Hata", str(exc))
            self._set_status("Hata oluştu.")

    def _on_analyze(self) -> None:
        self._set_status("Aerodinamik analiz çalışıyor...")
        try:
            params = self._collect_params()
            mach = params.meta.design_mach
            alt = params.meta.design_altitude_m

            CL = compute_CL_for_level_flight(params, mach, alt)
            self._drag_result = compute_drag(params, mach, alt, CL)
            self._stability_result = compute_stability(params, mach)
            CLa = compute_lift_curve_slope(params, mach)

            self._update_analysis_panel(params, CL, CLa)
            self._set_status(f"Analiz tamamlandı – L/D = {self._drag_result.LD_ratio:.2f}")
        except Exception as exc:
            messagebox.showerror("Hata", str(exc))
            self._set_status("Analiz hatası.")

    def _on_export_step(self) -> None:
        from fightercad.export.step_export import is_occ_available, export_step
        if not is_occ_available():
            messagebox.showwarning("Uyarı", "pythonocc-core yüklü değil. STEP export yapılamıyor.\nSTL export kullanabilirsiniz.")
            return
        path = filedialog.asksaveasfilename(
            title="STEP Dosyası Kaydet",
            defaultextension=".step",
            filetypes=[("STEP AP214", "*.step *.stp")],
        )
        if not path:
            return
        if self._assembler is None:
            messagebox.showinfo("Bilgi", "Önce 'Oluştur' butonuna basın.")
            return
        self._set_status("STEP export ediliyor...")
        try:
            export_step(self._assembler.components, path)
            self._set_status(f"STEP kaydedildi: {path}")
        except Exception as exc:
            messagebox.showerror("Export Hatası", str(exc))

    def _on_export_stl(self) -> None:
        if self._assembler is None:
            messagebox.showinfo("Bilgi", "Önce 'Oluştur' butonuna basın.")
            return
        path = filedialog.asksaveasfilename(
            title="STL Dosyası Kaydet",
            defaultextension=".stl",
            filetypes=[("STL Binary", "*.stl")],
        )
        if not path:
            return
        self._set_status("STL export ediliyor...")
        try:
            from fightercad.export.step_export import export_stl_mesh
            verts, faces = self._assembler.get_combined_mesh()
            export_stl_mesh(verts, faces, path)
            self._set_status(f"STL kaydedildi: {path}")
        except Exception as exc:
            messagebox.showerror("Export Hatası", str(exc))

    def _on_load_config(self) -> None:
        path = filedialog.askopenfilename(
            title="YAML Konfigürasyon Aç",
            filetypes=[("YAML", "*.yaml *.yml")],
        )
        if path:
            try:
                params = load_config(path)
                self._apply_params(params)
                self._set_status(f"Yüklendi: {path}")
            except Exception as exc:
                messagebox.showerror("Yükleme Hatası", str(exc))

    def _on_save_config(self) -> None:
        path = filedialog.asksaveasfilename(
            title="YAML Konfigürasyon Kaydet",
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml *.yml")],
        )
        if path:
            try:
                params = self._collect_params()
                save_config(params, path)
                self._set_status(f"Kaydedildi: {path}")
            except Exception as exc:
                messagebox.showerror("Kaydetme Hatası", str(exc))

    def _on_catia_connect(self) -> None:
        try:
            from fightercad.catia.live_update import CATIALiveUpdater
            self._catia = CATIALiveUpdater()
            if self._catia.connect():
                self._set_status("CATIA bağlantısı kuruldu.")
            else:
                messagebox.showwarning("Uyarı", "CATIA bağlantısı kurulamadı.\nCATIA'nın çalıştığından emin olun.")
        except Exception as exc:
            messagebox.showerror("CATIA Hatası", str(exc))

    def _on_catia_send(self) -> None:
        if not hasattr(self, "_catia") or self._catia is None:
            messagebox.showinfo("Bilgi", "Önce CATIA > Bağlan ile bağlantı kurun.")
            return
        if self._assembler is None:
            messagebox.showinfo("Bilgi", "Önce 'Oluştur' butonuna basın.")
            return
        self._set_status("CATIA'ya gönderiliyor...")
        try:
            self._catia.rebuild(self._assembler)
            self._set_status("CATIA modeli güncellendi.")
        except Exception as exc:
            messagebox.showerror("CATIA Hatası", str(exc))

    # -- View updates --
    def _update_3d_view(self) -> None:
        if self._components is None:
            return

        # Clear previous
        for child in self._tab_3d.winfo_children():
            child.destroy()

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        plot_aircraft_3d(self._components, ax=ax)

        self._canvas_3d = FigureCanvasTkAgg(fig, master=self._tab_3d)
        self._canvas_3d.draw()
        self._canvas_3d.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self._canvas_3d, self._tab_3d)
        toolbar.update()
        self._fig_3d = fig

    def _update_analysis_panel(self, params: AircraftParams, CL: float, CLa: float) -> None:
        dr = self._drag_result
        sr = self._stability_result
        if dr is None or sr is None:
            return

        # Clear previous
        for child in self._tab_aero.winfo_children():
            child.destroy()

        # Summary text
        summary = (
            f"Mach: {params.meta.design_mach:.2f}   İrtifa: {params.meta.design_altitude_m:.0f} m\n"
            f"CL: {CL:.4f}   CLα: {CLa:.3f} /rad\n"
            f"CD0: {dr.CD0:.5f}   CDi: {dr.CDi:.5f}   CDw: {dr.CDw:.5f}\n"
            f"CD_total: {dr.CD_total:.5f}   L/D: {dr.LD_ratio:.2f}\n"
            f"Statik Marj: {sr.static_margin_pct:.1f}% MAC   NP: {sr.neutral_point:.2f} m"
        )
        ttk.Label(self._tab_aero, text=summary, font=("Consolas", 10), justify="left").pack(padx=10, pady=5, anchor="w")

        # Plots frame
        plots = ttk.Frame(self._tab_aero)
        plots.pack(fill="both", expand=True)

        # Drag polar
        CL_arr = np.linspace(0, 1.2, 50)
        CD_arr = np.array([compute_drag(params, params.meta.design_mach, params.meta.design_altitude_m, cl).CD_total for cl in CL_arr])
        fig_polar = plt.figure(figsize=(4, 3))
        ax_polar = fig_polar.add_subplot(111)
        plot_drag_polar(CL_arr, CD_arr, CL_design=CL, ax=ax_polar)
        c1 = FigureCanvasTkAgg(fig_polar, master=plots)
        c1.draw()
        c1.get_tk_widget().pack(side="left", fill="both", expand=True)

        # Drag breakdown
        labels = ["Gövde", "Kanat", "V.Stab", "Alık", "Diğer", "İndüklenmiş", "Dalga"]
        values = [dr.CD0_fuselage, dr.CD0_wing, dr.CD0_vstab, dr.CD0_intake, dr.CD0_other, dr.CDi, dr.CDw]
        fig_bk = plt.figure(figsize=(4, 3))
        ax_bk = fig_bk.add_subplot(111)
        plot_drag_breakdown(labels, values, ax=ax_bk)
        c2 = FigureCanvasTkAgg(fig_bk, master=plots)
        c2.draw()
        c2.get_tk_widget().pack(side="left", fill="both", expand=True)

        # Mach sweep on tab 3
        self._update_mach_sweep(params)

    def _update_mach_sweep(self, params: AircraftParams) -> None:
        for child in self._tab_mach.winfo_children():
            child.destroy()

        mach_arr = np.linspace(0.6, 2.5, 40)
        alt = params.meta.design_altitude_m
        LD_arr = []
        CD_arr = []
        for m in mach_arr:
            cl = compute_CL_for_level_flight(params, m, alt)
            dr = compute_drag(params, m, alt, cl)
            LD_arr.append(dr.LD_ratio)
            CD_arr.append(dr.CD_total)

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        plot_mach_sweep(mach_arr, np.array(LD_arr), np.array(CD_arr), mach_design=params.meta.design_mach, ax=ax)
        canvas = FigureCanvasTkAgg(fig, master=self._tab_mach)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


def main() -> None:
    app = FighterCADApp()
    app.mainloop()


if __name__ == "__main__":
    main()
