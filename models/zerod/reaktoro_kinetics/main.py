import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from darts.engines import redirect_darts_output

from model_darts import Model as DartsModel
from model_reaktoro import Model as ReaktoroModel


def _compute_darts_amounts(darts_data):
    amounts = {}
    amounts["H2O(aq)"] = np.squeeze(darts_data['properties']['n_H2O(a)'])
    amounts["Ca+2"] = np.squeeze(darts_data['properties']['n_Ca+2'])
    amounts["Mg+2"] = np.squeeze(darts_data['properties']['n_Mg+2'])
    amounts["CO2(g)"] = np.squeeze(darts_data['properties']['n_CO2(g)'])

    amounts["Calcite"] = np.squeeze(darts_data['properties']['n_CaCO3'])
    amounts["Dolomite"] = np.squeeze(darts_data['properties']['n_CaMg(CO3)2'])
    amounts["Magnesite"] = np.squeeze(darts_data['properties']['n_MgCO3'])

    return amounts

def _load_h5(h5_file_path, group_name=None, flat=False):
    def _decode_array(values):
        if isinstance(values, (bytes, bytearray)):
            return values.decode("utf-8")

        arr = np.asarray(values)
        if arr.dtype.kind == "S":
            return arr.astype(str)
        if arr.dtype == object:
            decoded = np.array(
                [
                    item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else item
                    for item in arr.ravel()
                ],
                dtype=object,
            )
            return decoded.reshape(arr.shape)
        return arr

    def _to_dict(node):
        if isinstance(node, h5py.Dataset):
            return _decode_array(node[()])
        return {key: _to_dict(value) for key, value in node.items()}

    def _flatten_dict(data, prefix=""):
        out = {}
        for key, value in data.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                out.update(_flatten_dict(value, full_key))
            else:
                out[full_key] = value
        return out

    with h5py.File(h5_file_path, "r") as h5file:
        node = h5file[group_name] if group_name and group_name in h5file else h5file
        data = _to_dict(node)

    return _flatten_dict(data) if flat and isinstance(data, dict) else data


def run_darts_simulation(n_obl_mult: int = 1):
    output_folder = "output_darts"
    os.makedirs(output_folder, exist_ok=True)
    redirect_darts_output(os.path.join(output_folder, "log.txt"))

    model = DartsModel(runtime=1500.0, max_ts=10.0 / 3.0, n_obl_mult=n_obl_mult)
    model.init(output_folder=output_folder)

    ok = model.run(days=1500.0, method="radau")
    if not ok:
        raise RuntimeError("Zerod run failed to reach final time.")

    model.plot_state_history(
        use_log_p=False,
        output_path=os.path.join(output_folder, "state_history.png"),
    )
    return model, output_folder


def plot_darts_properties(model, output_folder):
    times, props = model.extract_property_history()
    if times.size == 0:
        return

    prop_keys = list(props.keys())
    amounts = _compute_darts_amounts(props)
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 8))

    ca_key = _find_key(prop_keys, "x_Ca+2")
    mg_key = _find_key(prop_keys, "x_Mg+2")
    if ca_key is not None:
        ax[0].plot(times, props[ca_key], color="b", label=ca_key)
    if mg_key is not None:
        ax[0].plot(times, props[mg_key], color="r", label=mg_key)

    amount_specs = [
        ("Calcite", "tab:blue"),
        ("Dolomite", "tab:red"),
        ("Magnesite", "tab:green"),
        ("CO2(g)", "tab:orange"),
        ("H2O(aq)", "tab:cyan"),
    ]

    for label, color in amount_specs:
        amount = amounts.get(label)
        if amount is None:
            continue
        ax[1].plot(times, amount, color=color, label=label)

    ax[0].set_ylabel("xCa2+, xMg2+")
    ax[1].set_ylabel("Amount [mol]")
    ax[1].set_xlabel("Time, days")
    if ax[0].lines:
        ax[0].legend()
    if ax[1].lines:
        ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "properties.png"))
    plt.close(fig)


def run_reaktoro_simulation():
    model = ReaktoroModel()
    model.run()
    return model


def plot_reaktoro_properties(model):
    data = _load_h5(
        h5_file_path=model.output_h5_file_path,
        group_name=model.output_h5_group,
    )
    time = data["Time"]

    fig, ax = plt.subplots(figsize=(6, 4))
    if "Ca+2" in data:
        ax.plot(time, data["Ca+2"], label="Ca+2")
    if "Mg+2" in data:
        ax.plot(time, data["Mg+2"], label="Mg+2")
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("Amount [mol]")
    ax.set_title("Aqueous Species Amounts")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(model.output_folder, "reaktoro_properties.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("Calcite", "Dolomite", "Magnesite"):
        if name in data:
            ax.plot(time, data[name], label=name)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("Amount [mol]")
    ax.set_title("Mineral Amounts")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(model.output_folder, "reaktoro_minerals.png"))
    plt.close(fig)

    if "pH" in data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time, data["pH"], label="pH")
        ax.set_xlabel("Time [day]")
        ax.set_ylabel("pH")
        ax.set_title("pH Over Time")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(model.output_folder, "reaktoro_pH.png"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("RateCalcite", "RateDolomite", "RateMagnesite"):
        if name in data:
            ax.plot(time, data[name], label=name)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("Reaction rate [mol/s]")
    ax.set_title("Reaction Rates")
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(model.output_folder, "reaktoro_reaction_rates.png"))
    plt.close(fig)


def plot_darts_reaktoro_comparison(darts_h5, reaktoro_h5, output_folder, fig_name: str = "darts_vs_reaktoro_comparison.png"):
    os.makedirs(output_folder, exist_ok=True)
    font_size = 14
    legend_font_size = 12

    darts_data = _load_h5(darts_h5)
    reaktoro_data = _load_h5(reaktoro_h5, group_name="results")
    times_reaktoro = np.squeeze(reaktoro_data['Time'])
    times_darts, darts_amounts = np.squeeze(darts_data['dynamic']['time']), _compute_darts_amounts(darts_data)
    darts_properties = darts_data.get("properties", {})
    darts_ref_amount = 1.0

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(12, 8))
    ax_minerals = ax[0, 0]
    ax_aqueous = ax[0, 1]
    ax_rates = ax[1, 0]
    ax_saturation = ax[1, 1]

    aqueous_specs = [("Ca+2", "tab:blue"), ("Mg+2", "tab:red")]
    mineral_specs = [
        ("Calcite", "tab:green"),
        ("Dolomite", "tab:orange"),
        ("Magnesite", "tab:brown"),
    ]
    reaction_rate_specs = [
        ("Calcite", "rate_CaCO3", "RateCalcite", "tab:green"),
        ("Dolomite", "rate_CaMg(CO3)2", "RateDolomite", "tab:orange"),
        ("Magnesite", "rate_MgCO3", "RateMagnesite", "tab:brown"),
    ]
    saturation_ratio_specs = [
        ("Calcite", "SR_CaCO3", "OmegaCalcite", "tab:green"),
        ("Dolomite", "SR_CaMg(CO3)2", "OmegaDolomite", "tab:orange"),
        ("Magnesite", "SR_MgCO3", "OmegaMagnesite", "tab:brown"),
    ]

    for name, color in mineral_specs:
        darts_series = darts_amounts.get(name)
        reaktoro_series = reaktoro_data.get(name)
        if darts_series is None or reaktoro_series is None:
            continue
        ax_minerals.plot(times_darts, darts_series / darts_ref_amount, color=color, linestyle="-", label=f"open-darts: {name}")
        ax_minerals.plot(times_reaktoro, reaktoro_series, color=color, linestyle="--", label=f"Reaktoro: {name}")

    for name, color in aqueous_specs:
        darts_series = darts_amounts.get(name)
        reaktoro_series = reaktoro_data.get(name)
        if darts_series is None or reaktoro_series is None:
            continue
        ax_aqueous.plot(times_darts, darts_series / darts_ref_amount, color=color, linestyle="-", label=f"open-darts: {name}")
        ax_aqueous.plot(times_reaktoro, reaktoro_series, color=color, linestyle="--", label=f"Reaktoro: {name}")

    for name, darts_rate_key, reaktoro_rate_key, color in reaction_rate_specs:
        darts_series = darts_properties.get(darts_rate_key)
        reaktoro_series = reaktoro_data.get(reaktoro_rate_key)
        if darts_series is None or reaktoro_series is None:
            continue

        vol = 3 * 55.6 / 1349
        conversion_mult = -1000 / 86400 * vol # kmol/day/m3 to mol/s
        s_id = 0
        ax_rates.plot(times_darts[s_id:], (np.squeeze(darts_series) * conversion_mult)[s_id:], color=color, linestyle="-", label=f"open-darts: {name}")
        ax_rates.plot(times_reaktoro, np.squeeze(reaktoro_series), color=color, linestyle="--", label=f"Reaktoro: {name}")

    for name, darts_sr_key, reaktoro_omega_key, color in saturation_ratio_specs:
        darts_series = darts_properties.get(darts_sr_key)
        reaktoro_series = reaktoro_data.get(reaktoro_omega_key)
        if darts_series is None or reaktoro_series is None:
            continue
        ax_saturation.plot(times_darts, np.squeeze(darts_series), color=color, linestyle="-", label=f"open-darts: {name}")
        ax_saturation.plot(times_reaktoro, np.squeeze(reaktoro_series), color=color, linestyle="--", label=f"Reaktoro: {name}")

    ax_minerals.set_ylabel("Amount [mol]", fontsize=font_size)
    if ax_minerals.lines:
        ax_minerals.legend(loc="right", fontsize=legend_font_size)

    ax_aqueous.set_ylabel("Amount [mol]", fontsize=font_size)
    if ax_aqueous.lines:
        ax_aqueous.legend(loc="upper right", fontsize=legend_font_size)

    ax_rates.set_ylabel("Reaction rate [mol/s]", fontsize=font_size)
    # if ax_rates.lines:
    #     ax_rates.legend(loc="upper right", fontsize=legend_font_size)
    ax_rates.set_ylim(-1e-7, 1e-7)

    ax_saturation.set_ylabel("Saturation ratio [-]", fontsize=font_size)
    if ax_saturation.lines:
        ax_saturation.set_yscale("log")
        ax_saturation.relim()
        ax_saturation.autoscale_view()
        y_min, y_max = ax_saturation.get_ylim()
        if not np.isfinite(y_min) or y_min <= 0.0:
            positive_candidates = []
            for line in ax_saturation.lines:
                y_vals = np.asarray(line.get_ydata())
                y_vals = y_vals[np.isfinite(y_vals) & (y_vals > 0.0)]
                if y_vals.size > 0:
                    positive_candidates.append(np.min(y_vals))
            y_min = min(positive_candidates) if positive_candidates else 1e-12
        y_min = min(y_min, 1.0)
        y_max = max(y_max, 1.0)
        ax_saturation.set_ylim(y_min, y_max)

        x_min, x_max = ax_saturation.get_xlim()
        x_text = x_max - 0.7 * (x_max - x_min)
        x_arrow = x_text - 0.02 * (x_max - x_min)

        if y_min < 1.0:
            ax_saturation.axhspan(y_min, 1.0, color="#9ecae1", alpha=0.5, zorder=0)
            ax_saturation.annotate(
                "",
                xy=(x_arrow, 5.6e-1),
                xytext=(x_arrow, 6.4e-1),
                arrowprops=dict(arrowstyle="-|>", color="#084594", lw=1.0),
            )
            ax_saturation.text(
                x_text, 6e-1, "dissolution",
                color="#084594", fontsize=font_size - 2, va="center", ha="left"
            )
        if y_max > 1.0:
            ax_saturation.axhspan(1.0, y_max, color="#fcbba1", alpha=0.5, zorder=0)
            ax_saturation.annotate(
                "",
                xy=(x_arrow, 1.65e-0),
                xytext=(x_arrow, 1.35e-0),
                arrowprops=dict(arrowstyle="-|>", color="#99000d", lw=1.0),
            )
            ax_saturation.text(
                x_text, 1.5e-0, "precipitation",
                color="#99000d", fontsize=font_size - 2, va="center", ha="left"
            )

        # ax_saturation.legend(loc="upper right", fontsize=legend_font_size)
        ax_saturation.set_ylim(1e-2, 1.5e+1)

    ax_rates.set_xlabel("Time [day]", fontsize=font_size)
    ax_saturation.set_xlabel("Time [day]", fontsize=font_size)

    for axis in (ax_minerals, ax_aqueous, ax_rates, ax_saturation):
        axis.tick_params(axis="both", labelsize=font_size)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, hspace=0.20)

    subfigure_captions = [
        (ax_minerals, "(a) Minerals"),
        (ax_aqueous, "(b) Aqueous Species"),
        (ax_rates, "(c) Reaction Rates"),
        (ax_saturation, "(d) Saturation Ratios"),
    ]
    for axis, caption in subfigure_captions:
        caption_offset = 0.08 if axis in (ax_rates, ax_saturation) else 0.035
        bbox = axis.get_position()
        fig.text(
            bbox.x0 + 0.5 * bbox.width,
            bbox.y0 - caption_offset,
            caption,
            ha="center",
            va="top",
            fontsize=font_size,
        )

    comparison_plot_path = os.path.join(output_folder, fig_name)
    fig.savefig(comparison_plot_path)
    plt.close(fig)
    return comparison_plot_path


if __name__ == "__main__":
    n_obl_mult = 1
    model, output_folder = run_darts_simulation(n_obl_mult=n_obl_mult)
    # plot_darts_properties(model=model, output_folder=output_folder)

    # model_ref = run_reaktoro_simulation()
    # plot_reaktoro_properties(model=model_ref)


    darts_h5 = os.path.join(f"output_darts_final_3", "zerod_solution.h5")
    reaktoro_h5 = os.path.join("output_reaktoro", "reaktoro_results.h5")
    output_folder = os.path.join("output_comparison")
    plot_darts_reaktoro_comparison(darts_h5=darts_h5, reaktoro_h5=reaktoro_h5,
    output_folder=output_folder, fig_name=f"darts_vs_reaktoro_comparison_new_obl_{n_obl_mult}.png")
