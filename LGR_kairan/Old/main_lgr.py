import numpy as np
import pandas as pd
import sys, os
from LGR_consK import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from darts.physics.base.operators_base import PropertyOperators as props
import matplotlib.tri as mtri

import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as mticker

def plot_well_time_data_2(m, time_data_df,
                                well_names=None,
                                component="CO2",
                                include_rates=("mass_rate",),
                                include_bhp=True,
                                include_bht=True):
    """
    Plot injector and producer on the same figure for selected metrics.

    Parameters
    ----------
    well_names : list[str] or None
        If None -> use first two wells in m.reservoir.wells
    component : str
        Component name for component-based rates (e.g., "CO2")
    include_rates : tuple[str]
        Any of ("mass_rate","molar_rate","volumetric_rate")
    include_bhp/include_bht : bool
        Whether to plot BHP/BHT combined
    """
    out_dir = os.path.join(WELL_DIR, "combined")
    os.makedirs(out_dir, exist_ok=True)

    wells_all = m.reservoir.wells
    if well_names is None:
        wells = wells_all[:2]
    else:
        name_set = set(well_names)
        wells = [w for w in wells_all if w.name in name_set]

    if len(wells) < 2:
        raise RuntimeError("Need at least 2 wells to plot combined (injector + producer).")



    type_candidates = ["by_sum_perfs"]
  
     

    rate_unit = {
        "volumetric_rate": "[m3/day]",
        "mass_rate": "[kg/day]",
        "molar_rate": "[kmol/day]",
        "advective_heat": "[kJ/day]",   
    }

    def find_col(well_name, key_prefix):
        """
        Try to find a column with suffix type in preferred order.
        key_prefix example: f"well_{name}_mass_rate_CO2_"
        """
        for tp in type_candidates:
            col = f"{key_prefix}{tp}"
            if col in time_data_df.columns:
                return col
        return None

    # ---------- component rates (injector+producer together) ----------
    for rate in include_rates:
        plt.figure(figsize=(8, 4.8), dpi=150)

        found_any = False
        for w in wells:
            
            key_prefix = f"well_{w.name}_{rate}_{component}_"
            col = find_col(w.name, key_prefix)
            if col is None:
                continue

            y = time_data_df[col].abs()
            plt.plot(time_data_df["time"], y, label=f"{w.name}")

            found_any = True

        if found_any:
            plt.xlabel("time [days]")
            plt.ylabel(f"{rate} {rate_unit.get(rate, '')}")
            plt.title(f"{component} {rate} (combined wells)")
            plt.legend()
            fpath = os.path.join(out_dir, f"combined_{component}_{rate}.png")
            plt.savefig(fpath, bbox_inches="tight")
        plt.close()

    # ---------- BHP combined ----------
    if include_bhp:
        plt.figure(figsize=(8, 4.8), dpi=150)
        found_any = False
        for w in wells:
            col = f"well_{w.name}_BHP"
            if col in time_data_df.columns:
                plt.plot(time_data_df["time"], time_data_df[col], label=f"{w.name}")
                found_any = True
        if found_any:
            plt.xlabel("time [days]")
            plt.ylabel("BHP [bar]") 
            plt.title("BHP (combined wells)")
            plt.legend()
            fpath = os.path.join(out_dir, "combined_BHP.png")
            plt.savefig(fpath, bbox_inches="tight")
        plt.close()

    # ---------- BHT combined ----------
    if include_bht:
        plt.figure(figsize=(8, 4.8), dpi=150)
        found_any = False
        for w in wells:
            col = f"well_{w.name}_BHT"
            if col in time_data_df.columns:
                plt.plot(time_data_df["time"], time_data_df[col], label=f"{w.name}")
                found_any = True
        if found_any:
            plt.xlabel("time [days]")
            plt.ylabel("BHT [K]")
            plt.title("BHT (combined wells)")
            plt.legend()
            fpath = os.path.join(out_dir, "combined_BHT.png")
            plt.savefig(fpath, bbox_inches="tight")
        plt.close()

    print(f"Saved combined plots to: {out_dir}")

def get_physics_field(model):
    X = np.array(model.physics.engine.X, copy=False)
    n = int(model.reservoir.n)
    nb = len(model.physics.vars)
    X_res = X[:n*nb]
    Xc = X_res.reshape((n,nb), order="C")
    return {str(v): Xc[:,i] for i, v in enumerate(model.physics.vars)}

def plot_xz_section(model, values, y0= None, tol= None, savepath="xz.png", title="", 
                    logscale=False, vmin=None, vmax=None, clip_floor=1e-20):
    res = model.reservoir
    x = res.cell_center_x
    y = res.cell_center_y
    z = res.cell_center_z
    v = np.asarray(values, float)
    x = np.asarray(res.cell_center_x)

    y = np.asarray(res.cell_center_y)
    z = np.asarray(res.cell_center_z)
    v = np.asarray(values)

    print("shapes:", x.shape, y.shape, z.shape, v.shape)

    if y0 is None:
        y0 = float(np.median(y))
    if tol is None:
        # tol = 0.5 * float(np.median(np.asarray(res.dy, float)))
        tol = 0.5 * float(100)
    
    m = np.abs(y-y0) <= tol
    xp, zp, vp = x[m], z[m], v[m]
    # --- choose normalization ---
    if logscale:
        # LogNorm can't handle <=0: clip to small positive
        vp_plot = np.clip(vp, clip_floor, None)

        if vmin is None:
            # robust lower bound: smallest positive in slice
            pos = vp_plot[vp_plot > 0]
            vmin = float(np.min(pos)) if pos.size else clip_floor
        if vmax is None:
            vmax = float(np.max(vp_plot))

        norm = LogNorm(vmin=vmin, vmax=vmax)
        vals_to_plot = vp_plot
    else:
        if vmin is None:
            vmin = float(np.min(vp))
        if vmax is None:
            vmax = float(np.max(vp))
        norm = Normalize(vmin=vmin, vmax=vmax)
        vals_to_plot = vp
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=150)

    sc = ax.scatter(xp, zp, c=vals_to_plot, s=10, norm=norm)

    cbar = fig.colorbar(sc, ax=ax)
    # nicer ticks/labels
    if logscale:
        cbar.formatter = mticker.LogFormatterMathtext()
        cbar.update_ticks()
    else:
        fmt = mticker.ScalarFormatter(useMathText=True)
        fmt.set_useOffset(False)
        cbar.formatter = fmt
        cbar.update_ticks()

    ax.set_xlabel("x [m]")
    ax.set_ylabel("depth [m]")
    ax.invert_yaxis()
    ax.set_title(title or f"XZ @ y={y0:.2f}")

    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)

output_dir = r".\output"
FIG_DIR = os.path.join(output_dir, "figures")
SECTION_DIR = os.path.join(FIG_DIR, "sections")
WELL_DIR = os.path.join(FIG_DIR, "well_time_plots")
os.makedirs(SECTION_DIR, exist_ok=True)
os.makedirs(WELL_DIR, exist_ok=True)
if __name__ == '__main__':
    redirect_darts_output(os.path.join(output_dir, 'run.log'))
    darts_model = Model()
   
    # darts_model.params.linear_type = darts_model.params.linear_solver_t.cpu_superlu
    darts_model.init()
    darts_model.set_output()


    if True:
        darts_model.run(365)
        # darts_model.reservoir.wells[0].control = n.physics.new_bhp_inj(100, 3*[n.zero])
        # darts_model.run_python(300, restart_dt=1e-3)
        darts_model.print_timers()
        darts_model.print_stat()
        save_dict = {
            "X": np.array(darts_model.physics.engine.X, copy=True),
            "vars": [str(v) for v in darts_model.physics.vars],
            "n_cells": int(len(darts_model.reservoir.poro)),
            "n_vars": len(darts_model.physics.vars),
            }

        np.save(os.path.join(output_dir, "solution_final.npy"), save_dict)

        prim = get_physics_field(darts_model)
        print("vars:", list(prim.keys()))
        for k in prim.keys():
            if "pressure" in k.lower() or k.lower() == "p":
                plot_xz_section(darts_model, prim[k],
                                savepath=os.path.join(SECTION_DIR, "coarse_section_P.png"),
                                title=k,
                                logscale=False)
                break

     
        for k in prim.keys():
            if "co2" in k.lower() :  
                plot_xz_section(darts_model, prim[k] - 1e-8,
                                savepath=os.path.join(SECTION_DIR, "coarse_section_CO2.png"),
                                title="CO2_delta",
                                logscale=False,
                                vmin=0, vmax=1)   
                break
        
        for k in prim.keys():
            if "temp" in k.lower() or "temperature" in k.lower():
                plot_xz_section(darts_model, prim[k],
                                savepath=os.path.join(SECTION_DIR, "coarse_section_T.png"),
                                title=k,
                                logscale=False)
                break

        # compute well time data
        time_data_dict = darts_model.output.store_well_time_data(save_output_files=True)
        time_data_df = pd.DataFrame.from_dict(time_data_dict)

        plot_well_time_data_2(darts_model, time_data_df)
    else:
        # darts_model.load_restart_data()
        darts_model.load_restart_data('output/solution.h5')
        time_data = pd.read_pickle("darts_time_data.pkl")

    # if True:
    #     Xn = np.array(darts_model.physics.engine.X, copy=False)
    #     nc = darts_model.physics.nc + darts_model.physics.thermal
    #     nb = darts_model.reservoir.mesh.n_res_blocks

    #     plt.figure(num=1, figsize=(12, 8), dpi=100)
    #     for i in range(nc if nc < 3 else 3):
    #         plt.subplot(330 + (i + 1))
    #         plt.plot(Xn[i:nb*nc:nc])
    #     plt.savefig('out.png')
    # else:
    #     #plot_sol(n)
    #     n.print_and_plot('sim_data')





