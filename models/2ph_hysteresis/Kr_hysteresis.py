#!/usr/bin/env python
"""
Single–cell Killough hysteresis test, CO2–H2O system, isothermal P/T
-------------------------------------------------------------------

• One cell, constant P = 30 bar, T = 330 K
• Drainage:    z_CO2 grows   0  →  0.8
• Imbibition:  z_CO2 shrinks 0.8 →  0.1
• Land + Killough (λ = 2) for the gas phase
• Corey for drainage of both phases
• Result: kr_g(Sg) curve with hysteresis
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --- DARTS / dartsflash pieces ---------------------------------------
from dartsflash.components import CompData
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.libflash import NegativeFlash
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import  ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
# from model import ALProperty_container, PhaseCapillaryHysteresis

# ---------------------------------------------------------------------
#  Corey parameters packaged in a tiny dataclass
# ---------------------------------------------------------------------
@dataclass
class Corey:
    krge:   float = 1.0    # end-point kr (gas)
    krwe:   float = 1.0    # end-point kr (water)
    sgc:    float = 0.1   # connate gas sat
    swc:    float = 0.20   # connate water sat
    ng:     float = 1.5    # Corey exp. gas
    nw:     float = 1.5    # Corey exp. water
    sgrmax: float = 0.40   # Land upper bound Sgr(max)
    a:      float = 0.8  # Killough exponent λ
    Pc_drainage_section = 'Pc_drainage'
    Pc_imbibition_section = 'Pc_imbibition'

# ---------------------------------------------------------------------
# 3.  Killough + Land evaluator
# ---------------------------------------------------------------------
class kr_hysteresis:
    """
    Gas or water relative-permeability evaluator with:
      • Corey drainage
      • Land residual trapping
      • Killough imbibition scanning (gas only)

    Parameters
    ----------
    corey : Corey
    phase : str      either 'gas' or 'water'
    """
    def __init__(self, corey: Corey, phase: str):
        self.phase = phase.lower()
        self.c = corey               # keep a copy

        # Land constant
        self.C_land = 1 / corey.sgrmax - 1 / (1 - corey.swc)

    # ------------ helpers -------------------------------------------
    def _kr_corey(self, S):
        """Corey drainage curve for the selected phase"""
        if self.phase == 'water':
            Se = (S - self.c.swc) / (1 - self.c.swc - self.c.sgc)
            Se = np.clip(Se, 0, 1)
            return self.c.krwe * Se**self.c.nw
        else:
            Se = (S - self.c.sgc) / (1 - self.c.swc - self.c.sgc)
            Se = np.clip(Se, 0, 1)
            kr = self.c.krge * Se**self.c.ng
            return kr

    # ------------ main API ------------------------------------------
    def evaluate(self, Sg, Sg_max = 0):
        """
        Parameters
        ----------
        Sg      : current gas saturation   (always pass TOTAL gas sat)
        Sg_max  : historical maximum gas saturation in this cell

        Returns
        -------
        kr      : relative permeability for *self.phase*
        """
        if self.phase == 'water':
            # no hysteresis in CO₂/H₂O; use drainage Corey only

            Sw = Sg
            return self._kr_corey(Sw)
        if Sg >= Sg_max:
            # drainage branch: Corey at Sg
            return self._kr_corey(Sg)

            # imbibition branch: Land + Killough
        Sg_max = min(1 - self.c.swc, Sg_max)
        Sgr = Sg_max / (1 + self.C_land * Sg_max)
        # Sgr = Sg_max/2
        Sg = min(1 - self.c.swc, Sg)
        Sgm = max(Sg - Sgr, 0.0)
        # if Sgm < 1e-5:
        #     Sgm = 0
        denom = max(Sg_max - Sgr, 1e-12)

        kr_inflection = self._kr_corey(Sg_max)
        factor = (Sgm / denom) ** self.c.a
        # factor = (Sgm / denom) ** 2
        return kr_inflection * factor


# ---------------------------------------------------------------------
# 4.  Minimal property container with CO₂–H₂O flash
# ---------------------------------------------------------------------
def make_property_container():
    components = ['H2O', 'CO2']
    comp_data  = CompData(components, setprops=True)

    # cubic PR + AQ for water
    pr  = CubicEoS(comp_data, CubicEoS.PR)
    aq  = AQEoS(
        comp_data,
        {AQEoS.water: AQEoS.Jager2003,
         AQEoS.solute: AQEoS.Ziabakhsh2012})

    flash_params = FlashParams(comp_data)
    flash_params.add_eos("PR", pr)
    flash_params.add_eos("AQ", aq)
    flash_params.eos_order = ["AQ", "PR"]

    prop = ALProperty_container(
        phases_name       = ["Aq", "V"],
        components_name   = components,
        Mw                = comp_data.Mw,
        temperature       = 330,      # placeholder; we pass real T later
        rock_comp         = 0,
        min_z             = 1e-14)

    prop.flash_ev     = NegativeFlash(
        flash_params, ["AQ", "PR"], [InitialGuess.Henry_AV])
    prop.density_ev = {'V': EoSDensity(pr, comp_data.Mw),
                                                  'Aq': Garcia2001(components)}
    prop.viscosity_ev = {'V': ConstFunc(6.4e-2), 'Aq': ConstFunc(0.47)}
    prop.enthalpy_ev = {'V': ConstFunc(0.0), 'Aq': ConstFunc(0.0)}
    # attach kr evaluators (drainage only for water)
    corey = Corey()
    prop.rel_perm_ev = {
        'V' : kr_hysteresis(corey, 'gas'),
        'Aq': kr_hysteresis(corey, 'water')
    }
    prop.capillary_pressure_ev = dict([('V', PhaseCapillaryHysteresis(corey, 'gas')),
                                                     ('Aq', PhaseCapillaryHysteresis(corey, 'water'))])
    return prop


# ---------------------------------------------------------------------
# 5.  Run one drainage–imbibition cycle
# ---------------------------------------------------------------------
# def run_cycle(P_bar=250.0, T_K=338.0):
#     prop = make_property_container()
#
#     # path for overall CO₂ fraction
#     z_segments = []
#     for top in np.arange(0.05, 0.8, 0.1):  # top goes from 0.2 to 0.8
#         z_segments.append(np.linspace(0.01, top, 30, endpoint=False)[1:])
#         z_segments.append(np.linspace(top, 0.01, 30, endpoint=False)[1:])
#
#     z_path = np.concatenate(z_segments)
#     mode_list = []  # 'drainage' or 'imbibition'
#     track_Sg = []
#     track_kr = []
#     Sg_max = 0.0
#     for z in z_path:
#         # Flash to get phase saturations
#         state = np.array([P_bar,1 - z ,T_K ,Sg_max])
#         prop.evaluate(state)
#         Sg = prop.sat[1]      # gas saturation from flash
#
#         # ------- hysteresis branch switching ------------------------
#         # if Sg_max == 1.0 and Sg < Sg_prev:  # start imbibition
#         #
#         #     Sg_max = Sg_prev
#         # elif Sg_max < 1.0:  # currently imbibition
#         #
#         #     if (Sg > Sg_prev) and (Sg >= Sg_max):  # back to drainage
#         #         Sg_max = 1.0
#         #     # Determine mode
#         if Sg >= Sg_max:
#             Sg_max = Sg
#             mode_list.append('drainage')
#         else:
#             mode_list.append('imbibition')
#         # ------- kr evaluation --------------------------------------
#         kr = prop.rel_perm_ev['V'].evaluate(Sg, Sg_max)
#
#         track_Sg.append(Sg)
#         track_kr.append(kr)
#         Sg_prev = Sg
#
#     return np.array(track_Sg), np.array(track_kr),np.array(mode_list)


# ---------------------------------------------------------------------
# 6.  Plotting
# ---------------------------------------------------------------------
def run_cycle(P_bar=250.0, T_K=330.0):
    prop = make_property_container()

    # Build a z_CO2 path with back-and-forth segments
    z_segments = []
    for top in np.arange(0.01, 0.8, 0.1):
        z_segments.append(np.linspace(0.01, top, 30, endpoint=False)[1:])
        z_segments.append(np.linspace(top, 0.01, 30, endpoint=False)[1:])
    z_path = np.concatenate(z_segments)

    mode_list = []         # 'drainage' or 'imbibition'
    track_Sg = []
    track_krg = []
    track_krw = []
    track_pc  = []

    Sg_max = 0.0
    for z in z_path:
        # State = [P, z_H2O, T, Sg_max]; here z_H2O = 1 - z_CO2
        state = np.array([P_bar, 1 - z, T_K, Sg_max])
        prop.evaluate(state)
        Sg = prop.sat[1]  # gas saturation from flash (phase index 1 = V)

        # Decide branch & update Sg_max
        if Sg >= Sg_max:
            Sg_max = Sg
            mode_list.append('drainage')
        else:
            mode_list.append('imbibition')

        # Relative perms
        krg = prop.rel_perm_ev['V'].evaluate(Sg, Sg_max)
        krw = prop.rel_perm_ev['Aq'].evaluate(Sg, Sg_max)  # drainage-only for water

        # Capillary pressure (uses the gas-phase evaluator here)
        pc_val = prop.capillary_pressure_ev['Aq'].evaluate(Sg, Sg_max)

        track_Sg.append(Sg)
        track_krg.append(krg)
        track_krw.append(krw)
        track_pc.append(pc_val)

    return (np.array(track_Sg),
            np.array(track_krg),
            np.array(track_krw),
            np.array(track_pc),
            np.array(mode_list))
# import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_relperm_and_pc(Sg, krg, krw, pc, mode, savepath_pdf="relperm_pc.pdf", savepath_png=None, pc_scale=1.0, pc_unit="bar"):
    # --- global style ---
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    ax_rp, ax_pc = axes

    # Colors
    c_gas   = "#D55E00"  # orange
    c_water = "#0072B2"  # blue
    c_pc    = "#6A3D9A"  # purple

    # --- (a) Relative permeability ---
    for i in range(1, len(Sg)):
        ls = '-' if mode[i] == 'drainage' else ':'
        ax_rp.plot(Sg[i-1:i+1], krg[i-1:i+1], ls, lw=1.8, color=c_gas)
        ax_rp.plot(Sg[i-1:i+1], krw[i-1:i+1], ls, lw=1.8, color=c_water)

    ax_rp.set_xlim(0.0, 1.0)
    ax_rp.set_ylim(0.0, 1.05)
    ax_rp.set_xlabel(r"Gas saturation $S_g$")
    ax_rp.set_ylabel(r"Relative permeability $k_r$")
    ax_rp.minorticks_on()
    ax_rp.grid(True, which="major", ls=":", lw=0.6, alpha=0.8)

    # Legends: (phase colors) + (style key)
    phase_legend = [
        Line2D([0], [0], color=c_gas,   lw=2.0, label=r"$k_{rg}$ (gas)"),
        Line2D([0], [0], color=c_water, lw=2.0, label=r"$k_{rw}$ (water)")
    ]
    style_legend = [
        Line2D([0], [0], color="black", lw=2.0, linestyle='-', label="Drainage"),
        Line2D([0], [0], color="black", lw=2.0, linestyle=':', label="Imbibition")
    ]
    l1 = ax_rp.legend(handles=phase_legend, loc="lower right", frameon=False)
    ax_rp.add_artist(l1)
    ax_rp.legend(handles=style_legend, loc="upper left", frameon=False)

    ax_rp.text(0.02, 0.98, "(a) Relative permeability", transform=ax_rp.transAxes,
               ha="left", va="top", fontweight="bold")

    # --- (b) Capillary pressure ---
    pc_plot = pc * pc_scale  # scale if needed (e.g., Pa -> bar: pc_scale = 1e-5)
    for i in range(1, len(Sg)):
        ls = '-' if mode[i] == 'drainage' else ':'
        ax_pc.plot(Sg[i-1:i+1], pc_plot[i-1:i+1], ls, lw=1.8, color=c_pc)

    ax_pc.set_xlim(0.0, 1.0)
    # autoscale y from data with a little headroom
    ymin, ymax = np.nanmin(pc_plot), np.nanmax(pc_plot)
    pad = 0.05 * max(1e-12, (ymax - ymin))
    ax_pc.set_ylim(ymin - pad, ymax + pad)

    ax_pc.set_xlabel(r"Gas saturation $S_g$")
    ax_pc.set_ylabel(r"Capillary pressure $p_c$ [{:s}]".format(pc_unit))
    ax_pc.minorticks_on()
    ax_pc.grid(True, which="major", ls=":", lw=0.6, alpha=0.8)

    ax_pc.text(0.02, 0.98, "(b) Capillary pressure", transform=ax_pc.transAxes,
               ha="left", va="top", fontweight="bold")

    # aesthetic spines
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

    plt.tight_layout()
    plt.savefig(savepath_pdf, bbox_inches="tight")  # vector for journal
    if savepath_png:
        plt.savefig(savepath_png, dpi=600, bbox_inches="tight")
    plt.show()
if __name__ == "__main__":
    Sg, krg, krw, pc, mode = run_cycle(P_bar=250.0, T_K=330.0)

    # If your pc evaluator returns Pa, use pc_scale=1e-5 and pc_unit="bar"
    plot_relperm_and_pc(
        Sg, krg, krw, pc, mode,
        savepath_pdf="relperm_pc.pdf",
        savepath_png="relperm_pc.png",
        pc_scale=1.0,   # set to 1e-5 if pc is in Pa and you want bar
        pc_unit="bar"
    )

# if __name__ == "__main__":
#     # Sg, kr = run_cycle()
#     #
#     # plt.figure(figsize=(7, 5))
#     # plt.plot(Sg, kr, "o-", lw=1.4, ms=4, color="#e69f00")
#     # plt.xlabel(r"Gas saturation $S_g$")
#     # plt.ylabel(r"$k_{rg}$")
#     # # plt.title("Killough hysteresis – single cell, CO₂/H₂O\n"
#     # #           "P = 250 bar, T = 330 K")
#     # plt.grid(True, ls=":")
#     # plt.show()
#     if __name__ == "__main__":
#         Sg, kr, mode = run_cycle()
#
#         plt.figure(figsize=(7, 5))
#
#         # Plot in segments with different colors and line styles
#         for i in range(1, len(Sg)):
#             if mode[i] == 'drainage':
#                 linestyle = '-'
#                 marker = 's'
#                 color = 'orange'
#             else:
#                 linestyle = ':'
#                 color = 'blue'
#                 marker = 'o'
#             plt.plot(Sg[i - 1:i + 1], kr[i - 1:i + 1], linestyle=linestyle, marker=marker,color=color, lw=1.4)
#
#         plt.xlabel(r"Gas saturation $S_g$")
#         plt.ylabel(r"$k_{rg}$")
#         plt.grid(True, ls=":")
#         plt.tight_layout()
#         plt.show()
