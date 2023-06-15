import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_bhp_darts(well_name, darts_df, style = '-', color = '#00A6D6', ax = None):
    search_str = well_name + ' : BHP'
    ax = darts_df.plot(x='time', y = [col for col in darts_df.columns if search_str in col], style = style, color = color, ax = ax)
          
    plt.show(block=False)
    return ax


def plot_oil_rate_darts(well_name, darts_df, style = '-', color = '#00A6D6', ax = None):
    search_str = well_name + ' : oil rate'
    ax = darts_df.plot(x='time', y = [col for col in darts_df.columns if search_str in col], style = style, color = color, ax = ax) 

    plt.show(block=False)
    return ax

def plot_gas_rate_darts(well_name, darts_df, style = '-', color = '#00A6D6', ax = None):
    search_str = well_name + ' : gas rate'
    ax = darts_df.plot(x='time', y = [col for col in darts_df.columns if search_str in col], style = style, color = color, ax = ax)

    plt.show(block=False)
    return ax

def plot_water_rate_darts(well_name, darts_df, style = '-', color = '#00A6D6', ax = None):
    search_str = well_name + ' : water rate'
    ax = darts_df.plot(x='time', y = [col for col in darts_df.columns if search_str in col], style = style, color = color, ax = ax)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    plt.show(block=False)
    return ax

def plot_temp_darts(well_name, darts_df, style = '-', color = '#00A6D6', ax = None):
    search_str = well_name + ' : T'
    ax = darts_df.plot(x='time', y = [col for col in darts_df.columns if search_str in col], style = style, color = color, ax = ax)

    plt.show(block=False)
    return ax
