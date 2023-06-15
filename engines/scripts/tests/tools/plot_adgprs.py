import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.plot_darts import *

def plot_bhp_adgprs(well_name, adgprs_df, style = '-.', color = '#C41E3A', ax = None):
    
    my_df = pd.DataFrame()
    my_df['Day'] = adgprs_df['Day']
    my_df[well_name] = 0
    col = well_name.upper() + 'BHP'
    if col in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col]

    ax = my_df.plot(x='Day', y=well_name, style = style, color = color, linewidth = 1.5, ax=ax)
    
    return ax  
    
def plot_water_rate_adgprs(well_name, adgprs_df, style = '-.', color = '#C41E3A', ax = None):
    
    my_df = pd.DataFrame()
    my_df['Day'] = adgprs_df['Day']
    my_df[well_name] = 0
    col_p = well_name.upper() + 'WPR'
    col_i = well_name.upper() + 'WIR'
    if col_p in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_p]

    if col_i in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_i]

    my_df[well_name] = -my_df[well_name]
    ax = my_df.plot(x='Day', y=well_name, style = style, color = color, linewidth = 1.5, ax=ax)
    
    return ax

def plot_oil_rate_adgprs(well_name, adgprs_df, style = '-.', color = '#C41E3A', ax = None):
    
    my_df = pd.DataFrame()
    my_df['Day'] = adgprs_df['Day']
    my_df[well_name] = 0
    col_p = well_name.upper() + 'OPR'
    col_i = well_name.upper() + 'OIR'
    if col_p in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_p]

    if col_i in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_i]

    my_df[well_name] = -my_df[well_name]
    ax = my_df.plot(x='Day', y=well_name, style = style, color = color, linewidth = 1.5, ax=ax)
    
    return ax

def plot_gas_rate_adgprs(well_name, adgprs_df, style = '-.', color = '#C41E3A', ax = None):
    
    my_df = pd.DataFrame()
    my_df['Day'] = adgprs_df['Day']
    my_df[well_name] = 0
    col_p = well_name.upper() + 'GPR'
    col_i = well_name.upper() + 'GIR'
    if col_p in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_p]

    if col_i in adgprs_df.columns:
        my_df[well_name] += adgprs_df[col_i]

    my_df[well_name] = -my_df[well_name]
    ax = my_df.plot(x='Day', y=well_name, style = style, color = color, linewidth = 1.5, ax=ax)
    
    return ax


def plot_bhp_vs_adgprs(well_name, darts_df, adgprs_df, ax = None):
    ax = plot_bhp_adgprs (well_name, adgprs_df, ax)
    plot_bhp_darts (well_name, darts_df, ax = ax)
    return ax

def plot_temp_vs_adgprs(well_name, darts_df, adgprs_df, ax = None):
    ax = plot_temp_adgprs (well_name, adgprs_df, ax)
    plot_temp_darts (well_name, darts_df, ax = ax)
    return ax

def plot_oil_rate_vs_adgprs(well_name, darts_df, adgprs_df, ax = None):
    ax = plot_oil_rate_adgprs (well_name, adgprs_df, ax)
    plot_oil_rate_darts (well_name, darts_df, ax = ax)
    return ax

def plot_gas_rate_vs_adgprs(well_name, darts_df, adgprs_df, ax = None):
    ax = plot_gas_rate_adgprs (well_name, adgprs_df, ax)
    plot_gas_rate_darts (well_name, darts_df, ax = ax)
    return ax

def plot_water_rate_vs_adgprs(well_name, darts_df, adgprs_df, ax = None):
    ax = plot_water_rate_adgprs (well_name, adgprs_df, ax)
    plot_water_rate_darts (well_name, darts_df, ax = ax)
    return ax
