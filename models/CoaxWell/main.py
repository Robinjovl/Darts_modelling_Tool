from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

m = Model(resolution=10)
m.init()
m.set_output()
m.output.output_to_vtk(ith_step = 0, engine = True)

m.run(365)
m.print_timers()
m.print_stat()
m.output.output_to_vtk(ith_step = 1, engine = False)

# compute and save well time data
time_data_dict = m.output.store_well_time_data(save_output_files=True)

# plot well time data
time_data_df = pd.DataFrame.from_dict(time_data_dict)

def save_logx_plot(df, x, y, outpath, xlabel=None, dpi=100):
    ax = df.plot(x=x, y=y)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel or f"{x} (log scale)")
    fig = ax.get_figure()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# BHT vs log(time)
save_logx_plot(
    time_data_df,
    x='time',
    y=['well_PRD_BHT', 'well_INJ_BHT'],
    outpath=m.output_folder + '/well_temperature.png'
)

# BHP vs log(time)
save_logx_plot(
    time_data_df,
    x='time',
    y=['well_PRD_BHP', 'well_INJ_BHP'],
    outpath=m.output_folder + '/well_BHP.png'
)

# Production rates vs log(time)
save_logx_plot(
    time_data_df,
    x='time',
    y=['well_PRD_volumetric_rate_water_at_wh',
       'well_PRD_volumetric_rate_steam_at_wh'],
    outpath=m.output_folder + '/well_production_rates.png'
)

# Injection rates vs log(time)
save_logx_plot(
    time_data_df,
    x='time',
    y=['well_INJ_volumetric_rate_water_at_wh',
       'well_INJ_volumetric_rate_steam_at_wh'],
    outpath=m.output_folder + '/well_injection_rates.png'
)
