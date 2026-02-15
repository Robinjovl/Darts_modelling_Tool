import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm, Normalize
import sys, os



def plot_well_time_data_2(m, time_data_df,
                                save_output_files=True,
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

    out_dir = os.path.join(save_output_files,"combined")
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
