import numpy as np
import pandas as pd
import sys, os
from Eclipse_method import Model
from darts.engines import value_vector, redirect_darts_output
import matplotlib.pyplot as plt
from darts.physics.base.operators_base import PropertyOperators as props

import os
import matplotlib.pyplot as plt

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

    out_root = getattr(m.output, "output_folder", getattr(m, "output_folder", "output"))
    out_dir = os.path.join(out_root, "figures", "well_time_plots", "combined")
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



if __name__ == '__main__':
    darts_model = Model()
    # darts_model.params.linear_type = n.params.linear_solver_t.cpu_superlu
    darts_model.init()
    darts_model.set_output()


    if True:
        darts_model.run(50)
        # darts_model.reservoir.wells[0].control = n.physics.new_bhp_inj(100, 3*[n.zero])
        # darts_model.run_python(300, restart_dt=1e-3)
        darts_model.print_timers()
        darts_model.print_stat()

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
