import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

matplotlib.use("Agg")
import inspect
from builtins import zip as _zip
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps


def plot_well_multithreaded(out_dirs, time_data_dfs, n_threads=4):
    def _plot_single(df, x, y_cols, xlabel, ylabel, title, path):
        fig, ax = plt.subplots(figsize=(5, 3))
        for col in y_cols:
            ax.plot(df[x].to_numpy(), df[col].to_numpy())
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_title(title)
        if len(y_cols) > 1:
            ax.legend(fontsize=6, loc="best", frameon=False)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=100)
        plt.close(fig)

    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "axes.grid": True,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "lines.linewidth": 2,
            "figure.autolayout": False,
            "savefig.pad_inches": 0.1,
            "text.usetex": False,
            "lines.antialiased": False,
        }
    )

    from builtins import zip as _zip

    key_units = {
        "volumetric": "[m3/day]",
        "mass_rate": "[kg/day]",
        "molar_rate": "[kmol/day]",
        "advective_heat": "[kJ/day]",
        "BHT": "[°C]",
        "BHP": "[bar]",
    }

    for out_dir, time_data_df in _zip(out_dirs, time_data_dfs, strict=False):
        os.makedirs(out_dir, exist_ok=True)

        jobs = []
        for key in time_data_df.keys():
            ylabel = next(
                (unit for name, unit in key_units.items() if name in key), None
            )
            if n_threads:
                jobs.append(
                    (
                        time_data_df[["time", key]].copy(),
                        "time",
                        [key],
                        "time [days]",
                        ylabel,
                        key,
                        os.path.join(out_dir, "fast", f"{key}.png"),
                    )
                )

        if n_threads and jobs:
            os.makedirs(os.path.join(out_dir, "fast"), exist_ok=True)
            nthreads = min(2, os.cpu_count() or 4)
            print(
                f"[plot_well_time_data_fast] Starting {len(jobs)} plots on {nthreads} threads..."
            )

            # Use threads — zero pickling, zero startup delay
            with ThreadPoolExecutor(max_workers=nthreads) as ex:
                futures = [ex.submit(_plot_single, *args) for args in jobs]
                for _ in as_completed(futures):
                    pass  # could add tqdm here for progress


def plot_well_time_data_2(out_dirs, time_data_dfs):
    """
    Make plots out of the time data dataframe.
    """

    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "axes.grid": True,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "lines.linewidth": 2,
            "figure.autolayout": False,
            "savefig.pad_inches": 0.1,
            "text.usetex": False,
            "lines.antialiased": False,
        }
    )

    def _plot_single(y_valid, y_label, title, out_dir):
        ax = time_data_df.plot(
            x="time",
            y=y_valid,
            xlabel="time [days]",
            ylabel=y_label,
            title=title,
            grid=True,
        )
        fig = ax.get_figure()
        fig.savefig(
            os.path.join(out_dir, f"{title}.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)

        return 0

    # import warnings
    from builtins import zip as _zip

    for out_dir, time_data_df in _zip(out_dirs, time_data_dfs, strict=False):
        os.makedirs(out_dir, exist_ok=True)

        # required_attrs = ["wells", "components", "phases", "perfs"]
        # for attr in required_attrs:
        # if attr not in time_data_df.attrs:
        # warnings.warn(
        #     f"DataFrame in {out_dir} is missing attribute '{attr}'. "
        #     f"Plots depending on it may be incomplete.",
        #     UserWarning,
        # )

        wells = time_data_df.attrs.get("wells", [])
        components = time_data_df.attrs.get("components", [])
        phases = time_data_df.attrs.get("phases", [])
        perfs = time_data_df.attrs.get("perfs", [])

        rate_list = {
            'volumetric_rate': ' [m3/day]',
            'mass_rate': ' [kg/day]',
            'molar_rate': ' [kmol/day]',
            'advective_heat': ' [kJ]',
        }
        type_list = ['perf', 'at_wh', 'by_sum_perfs']

        # for the phases
        for typ in type_list:
            os.makedirs(os.path.join(out_dir, 'phases', typ), exist_ok=True)
            dir = os.path.join(out_dir, 'phases', typ)
            if typ == 'perf':
                for phase in phases:
                    for rate in rate_list:
                        for perf in perfs:
                            y = [f"well_{perf}_{rate}_{phase}"]
                            y_valid = [col for col in y if col in time_data_df.columns]
                            if y_valid:
                                _plot_single(
                                    y_valid, rate + rate_list[rate], y_valid[0], dir
                                )
            else:
                for well in wells:
                    for phase in phases:
                        for rate in rate_list:
                            y = [f"well_{well}_{rate}_{phase}_{typ}"]
                            y_valid = [col for col in y if col in time_data_df.columns]
                            if y_valid:
                                _plot_single(
                                    y_valid, rate + rate_list[rate], y_valid[0], dir
                                )

        # for the components
        for typ in type_list:
            os.makedirs(os.path.join(out_dir, 'component', typ), exist_ok=True)
            dir = os.path.join(out_dir, 'component', typ)
            if typ == 'perf':
                for component in components:
                    for rate in {'molar_rate': ' [kmol/day]'}:
                        for perf in perfs:
                            y = [f"well_{perf}_{rate}_{component}"]
                            y_valid = [col for col in y if col in time_data_df.columns]
                            if y_valid:
                                _plot_single(
                                    y_valid, rate + rate_list[rate], y_valid[0], dir
                                )

            else:
                for component in components:
                    for rate in {'molar_rate': ' [kmol/day]'}:
                        y = [f"well_{well}_{rate}_{component}_{typ}"]
                        y_valid = [col for col in y if col in time_data_df.columns]
                        if y_valid:
                            _plot_single(
                                y_valid, rate + rate_list[rate], y_valid[0], dir
                            )

        # BHP and BHT
        bottom_hole_list = {"BHP": " [bar]", "BHT": " [°C]"}
        for well in wells:
            for key, unit in bottom_hole_list.items():
                y = [f"well_{well}_{key}"]
                y_valid = [col for col in y if col in time_data_df.columns]
                if y_valid:
                    ax = time_data_df.plot(
                        x="time",
                        y=y_valid,
                        xlabel="time [days]",
                        ylabel=unit,
                        title=well,
                        grid=True,
                    )
                    fig = ax.get_figure()
                    fig.savefig(
                        os.path.join(out_dir, f"{y_valid[0]}.png"),
                        dpi=100,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

    return 0


def multi_fig_decorator(f):
    df_arg_name = next(iter(inspect.signature(f).parameters))

    @wraps(f)
    def wrapper(*args, **kwargs):
        if args:
            df, *args = args
        else:
            df = kwargs.pop(df_arg_name)

        # Identify x and y columns
        x_col = "time" if "time" in df.columns else None
        y_cols = [c for c in df.columns if c != x_col]

        figs = {}
        for col in y_cols:
            # keep 'time' and current y-column only
            keep_cols = [col]
            if x_col:
                keep_cols.insert(0, x_col)
            sub_df = df[keep_cols].copy()

            # Defensive: skip if column has no data
            if sub_df[col].dropna().empty:
                continue

            kwargs[df_arg_name] = sub_df
            kwargs.setdefault("x_label", "time [days]" if x_col else "index")
            kwargs.setdefault("y_label", col)

            fig = f(*args, **kwargs)
            figs[col] = fig

        return figs

    return wrapper


@multi_fig_decorator
def my_figure_generator_function(df, x_label="time [days]", y_label=None):
    # determine x and y columns
    x = df["time"] if "time" in df.columns else df.index
    y_cols = [c for c in df.columns if c != "time"]
    if not y_cols:
        return None  # safety: skip if no y-column found

    key_units = {
        "volumetric": "Volumetric rate [m3/day]",
        "mass_rate": "Mass rate [kg/day]",
        "molar_rate": "Molar rate [kmol/day]",
        "advective_heat": "Advective heat rate [kJ/day]",
        "BHT": "Temperature [°C]",
        "BHP": "Pressure [bar]",
    }

    col = y_cols[0]
    y_label = next((unit for name, unit in key_units.items() if name in col), None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df[col], name=col))
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        title=f"{col} vs {x_label}",
        margin=dict(l=60, r=20, t=40, b=50),
    )
    fig.update_yaxes(tickformat=".2f")  # show full numbers without M/k suffix
    return fig


def html_plot(out_dirs, time_data_dfs):
    for out_dir, time_data in _zip(out_dirs, time_data_dfs, strict=False):
        figs = my_figure_generator_function(time_data)

        if 1:
            import os

            import plotly.io as pio

            os.makedirs(out_dir, exist_ok=True)
            for name, fig in figs.items():
                pio.write_html(
                    fig,
                    file=os.path.join(out_dir, f"{name}.html"),
                    include_plotlyjs='cdn',
                    auto_open=False,
                )
        else:
            import os
            import zipfile

            import plotly.io as pio

            with zipfile.ZipFile(
                os.path.join(out_dir, "plots.zip"), "w", zipfile.ZIP_DEFLATED
            ) as zf:
                for name, fig in figs.items():
                    html_bytes = pio.to_html(
                        fig, include_plotlyjs="cdn", full_html=False
                    ).encode("utf-8")
                    zf.writestr(f"{name}.html", html_bytes)
            print(f"Packed {len(figs)} figures into plots.zip")

    return 0


def plot_wells_output(out_dirs, time_data_dfs, n_threads=4, format='png'):
    if format == 'png':
        if n_threads:
            plot_well_multithreaded(out_dirs, time_data_dfs, n_threads)
        else:
            plot_well_time_data_2(out_dirs, time_data_dfs)

    elif format == 'html':
        html_plot(out_dirs, time_data_dfs)

    return 0


def plot_phase_rate_darts(
    well_name, darts_df, ph, style='-', color='#00A6D6', ax=None, alpha=1
):
    search_str = well_name + ' : ' + ph + ' rate'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
        alpha=alpha,
    )
    return ax


def plot_bhp_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : BHP'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
    )
    return ax


def plot_oil_rate_darts(
    well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    search_str = well_name + ' : oil rate'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
        alpha=alpha,
    )
    return ax


def plot_oil_rate_darts_2(
    well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    search_str = well_name + ' : oil rate'
    darts_df['time'] = -darts_df['time']
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
        alpha=alpha,
    )
    return ax


def plot_gas_rate_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : gas rate'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
    )
    return ax


def plot_water_rate_darts(
    well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    search_str = well_name + ' : water rate'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
        alpha=alpha,
    )

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_watercut_darts(
    well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1, label=''
):
    wat = well_name + ' : water rate (m3/day)'
    oil = well_name + ' : oil rate (m3/day)'
    wcut = well_name + ' watercut'
    darts_df[wcut] = darts_df[wat] / (darts_df[wat] + darts_df[oil])

    if label == '':
        label = wcut
    ax = darts_df.plot(
        x='time', y=wcut, style=style, color=color, ax=ax, alpha=alpha, label=label
    )

    ax.set_ylim(0, 1)

    return ax


def plot_water_rate_darts_2(
    well_name, darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    search_str = well_name + ' : water rate'
    darts_df['time'] = -darts_df['time']
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
        alpha=alpha,
    )

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_water_rate_vs_obsrate(
    well_name, darts_df, truth_df, style='-', color='#00A6D6', ax=None, marker="o"
):
    search_str = well_name + ' : water rate'
    darts_df = darts_df.set_index('time', drop=False)
    ax.scatter(
        x=abs(
            darts_df[[col for col in truth_df.columns if search_str in col]].loc[
                truth_df.time, :
            ]
        ),
        y=abs(truth_df[[col for col in truth_df.columns if search_str in col]]),
        marker=marker,
    )
    max_rate = max(
        max(abs(darts_df[search_str + ' (m3/day)'].values)),
        max(abs(truth_df[search_str + ' (m3/day)'].values)),
    )
    min_rate = min(
        min(abs(darts_df[search_str + ' (m3/day)'].values)),
        min(abs(truth_df[search_str + ' (m3/day)'].values)),
    )
    plt.plot([max_rate, min_rate], [max_rate, min_rate], color=color, marker='x')
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')
    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)

    return ax


def plot_water_rate_vs_obsrate_time(
    well_name,
    darts_df,
    truth_df,
    style='-',
    color='#00A6D6',
    ax=None,
    marker="o",
    time=0,
):
    search_str = well_name + ' : water rate'
    darts_df = darts_df.set_index('time', drop=False)
    truth_df = truth_df.set_index('time', drop=False)
    ax.scatter(
        x=abs(
            darts_df[[col for col in truth_df.columns if search_str in col]].loc[
                truth_df.time[time], :
            ]
        ),
        y=abs(
            truth_df[[col for col in truth_df.columns if search_str in col]].loc[
                truth_df.time[time], :
            ]
        ),
        marker=marker,
        label=(well_name + ' time: ' + time.__str__()),
    )
    ax.legend(loc=5)
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')
    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)

    return ax


def plot_oil_rate_rate_vs_obsrate(
    well_name, darts_df, truth_df, style='-', color='#00A6D6', ax=None, marker="o"
):
    search_str = well_name + ' : oil rate'
    darts_df = darts_df.set_index('time', drop=False)
    ax.scatter(
        x=abs(
            darts_df[[col for col in truth_df.columns if search_str in col]].loc[
                truth_df.time, :
            ]
        ),
        y=abs(truth_df[[col for col in truth_df.columns if search_str in col]]),
        marker=marker,
    )
    min(abs(darts_df[search_str + ' (m3/day)'].values))
    min_oil_rate_truth = min(abs(truth_df[search_str + ' (m3/day)'].values))
    max(abs(darts_df[search_str + ' (m3/day)'].values))
    max_oil_rate_truth = max(abs(truth_df[search_str + ' (m3/day)'].values))
    plt.plot(
        [max_oil_rate_truth, min_oil_rate_truth],
        [max_oil_rate_truth, min_oil_rate_truth],
        color=color,
        marker='x',
    )
    plt.ylabel('Truth Data')
    plt.xlabel('Simulation Data')

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)
    plt.grid(True)

    return ax


def plot_total_inj_water_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : water rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) > 0:
            if 'I' in col:
                #     acc_df['total'] += darts_df[col]
                for i in range(0, len(darts_df[col])):
                    if darts_df[col][i] >= 0:
                        # acc_df['total'][i] += darts_df[col][i]
                        acc_df.loc[i, 'total'] += darts_df[col][i]

    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_total_inj_gas_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : gas rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) > 0:
            if 'I' in col:
                #     acc_df['total'] += darts_df[col]
                for i in range(0, len(darts_df[col])):
                    if darts_df[col][i] >= 0:
                        # acc_df['total'][i] += darts_df[col][i]
                        acc_df.loc[i, 'total'] += darts_df[col][i]

    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_water_rate_prediction(darts_df, style='-', color='#00A6D6', ax=None, alpha=1):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : water rate'
    for col in darts_df.columns:
        if search_str in col:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_total_prod_water_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : water rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_acc_prod_water_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = 'water  acc'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_acc_prod_oil_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = 'oil  acc'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_total_prod_oil_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : oil rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_total_prod_gas_rate_darts(
    darts_df, style='-', color='#00A6D6', ax=None, alpha=1
):
    acc_df = pd.DataFrame()
    acc_df['time'] = darts_df['time']
    acc_df['total'] = 0
    search_str = ' : gas rate'
    for col in darts_df.columns:
        if search_str in col:
            # if sum(darts_df[col]) < 0:
            if 'I' not in col:
                acc_df['total'] += darts_df[col]

    acc_df['total'] = acc_df['total'].abs()
    ax = acc_df.plot(x='time', y='total', style=style, color=color, ax=ax, alpha=alpha)

    ymin, ymax = ax.get_ylim()
    if ymax < 0:
        ax.set_ylim(ymin * 1.1, 0)

    if ymin > 0:
        ax.set_ylim(0, ymax * 1.1)

    return ax


def plot_temp_darts(well_name, darts_df, style='-', color='#00A6D6', ax=None):
    search_str = well_name + ' : temperature'
    ax = darts_df.plot(
        x='time',
        y=[col for col in darts_df.columns if search_str in col],
        style=style,
        color=color,
        ax=ax,
    )

    return ax


def plot_extracted_energy_darts(darts_df, style='-', color='#00A6D6', ax=None):
    search_str = ' : energy'
    y = [
        col for col in darts_df.columns if search_str in col
    ]  # get columns with 'energy' data
    y = darts_df[y].sum(axis=1)  # sum over the wells
    t = darts_df['time']
    dt = np.append(0, np.ediff1d(t))  # add the first time step
    col_name = 'energy extracted, PJ'
    darts_df[col_name] = -(y * dt).cumsum() * 1e-12  # kJ/day -> PJ
    ax = darts_df.plot(x='time', y=col_name, style=style, color=color, ax=ax)
    return ax


def tersurf(a, b, c, d, line=None, inf_p=None):
    import matplotlib.tri as tri

    """
    :param a: z1
    :param b: z2
    :param c: z3
    :param d: values you want to plot ( e.g. operator values, derivative, hessian,...)
    :param line: in case want to draw trajectory on it
    :param inf_p: inflection point in a given trajectory
    :return:
    """
    z = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    # transfer matrix
    # mt = np.transpose([[1 / 2, 1], [np.sqrt(3) / 2, 0]])
    # plot triangle
    # p = np.matmul(z, mt)
    # plt.figure(figsize=(10, 8), dpi=100)
    # plt.plot(p[:, 0], p[:, 1], 'k', 'linewidth', 1.5)
    x = 0.5 - z[:, 0] * np.cos(np.pi / 3) + z[:, 1] / 2
    y = 0.866 - z[:, 0] * np.sin(np.pi / 3) - z[:, 1] / np.tan(np.pi / 6) / 2
    plt.plot(x, y, 'k', 'linewidth', 1.5)
    # create the grid
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) * 0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    # creating the grid
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=3)

    # plotting the mesh
    plt.triplot(trimesh, color='navajowhite', linestyle='--', linewidth=0.8)
    plt.ylim([0, 1])
    plt.axis('off')

    # translate the data to cords
    x = 0.5 - a * np.cos(np.pi / 3) + b / 2
    y = 0.866 - a * np.sin(np.pi / 3) - b / np.tan(np.pi / 6) / 2

    # create a triangulation out of these points
    T = tri.Triangulation(x, y)
    # plot the contour
    vmin = min(d) - 0.1  # -10.1
    vmax = max(d) + 0.1  # 10.1
    level = np.linspace(vmin, vmax, 101)
    plt.tricontourf(x, y, T.triangles, d, cmap='jet', levels=level)
    plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], linewidth=1)
    plt.rc('font', size=12)
    cax = plt.axes([0.75, 0.55, 0.055, 0.3])
    plt.colorbar(cax=cax, format='%.3f', label='')
    # plt.gcf().text(0.08, 0.1, '$C_1$', fontsize=20, color='black')
    # plt.gcf().text(0.91, 0.1, '$CO_2$', fontsize=20, color='black')
    # plt.gcf().text(0.5, 0.8, '$H_2O$', fontsize=20, color='black')
    if line is not None:
        # in case, want to draw random trajectories on the ternary diagrm
        line = line[:, 1:]
        # traj = np.matmul(line, mt)
        # plt.plot(traj[:,0], traj[:,1], '--')
        x = 0.5 - line[:, 0] * np.cos(np.pi / 3) + line[:, 1] / 2
        y = 0.866 - line[:, 0] * np.sin(np.pi / 3) - line[:, 1] / np.tan(np.pi / 6) / 2
        plt.plot(x, y, '--')
        if inf_p is not None:
            # inf_p = np.matmul(inf_p, mt)
            # plt.scatter(inf_p[:, 0], inf_p[:, 1])
            # translate the data to cords
            x = 0.5 - inf_p[:, 0] * np.cos(np.pi / 3) + inf_p[:, 1] / 2
            y = (
                0.866
                - inf_p[:, 0] * np.sin(np.pi / 3)
                - inf_p[:, 1] / np.tan(np.pi / 6) / 2
            )
            plt.scatter(x, y)
