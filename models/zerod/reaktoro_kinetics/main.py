import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from darts.engines import redirect_darts_output

from model_darts import Model as DartsModel
from model_reaktoro import Model as ReaktoroModel


def _compute_darts_amounts(darts_data, volume=1.0):
    amounts = {}
    poro = np.squeeze(darts_data['properties']['porosity'])
    dens_liq = np.squeeze(darts_data['properties']['dens_m_liq'])
    dens_gas = np.squeeze(darts_data['properties']['dens_m_gas'])
    sat_liq = np.squeeze(darts_data['properties']['sat_liq'])
    sat_gas = np.squeeze(darts_data['properties']['sat_gas'])
    y_co2 = np.squeeze(darts_data['properties']['y_CO2(g)'])
    x_h2o = np.squeeze(darts_data['properties']['x_H2O(aq)'])
    x_ca = np.squeeze(darts_data['properties']['x_Ca+2'])
    x_mg = np.squeeze(darts_data['properties']['x_Mg+2'])

    liq_factor = poro * sat_liq * dens_liq * volume * 1000.0
    amounts["H2O(aq)"] = liq_factor * x_h2o
    amounts["Ca+2"] = liq_factor * x_ca
    amounts["Mg+2"] = liq_factor * x_mg
    amounts["CO2(g)"] = poro * sat_gas * dens_gas * y_co2 * volume * 1000.0

    # minerals
    dens_m_solid_Calcite = np.squeeze(darts_data['properties']['dens_m_solid_CaCO3'])
    dens_m_solid_Dolomite = np.squeeze(darts_data['properties']['dens_m_solid_CaMg(CO3)2'])
    dens_m_solid_Magnesite = np.squeeze(darts_data['properties']['dens_m_solid_MgCO3'])
    sat_Calcite = np.squeeze(darts_data['properties']['sat_CaCO3'])
    sat_Dolomite = np.squeeze(darts_data['properties']['sat_CaMg(CO3)2'])
    sat_Magnesite = np.squeeze(darts_data['properties']['sat_MgCO3'])

    amounts["Calcite"] = dens_m_solid_Calcite * sat_Calcite * volume * 1000.0
    amounts["Dolomite"] = dens_m_solid_Dolomite * sat_Dolomite * volume * 1000.0
    amounts["Magnesite"] = dens_m_solid_Magnesite * sat_Magnesite * volume * 1000.0

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


def run_darts_simulation():
    output_folder = "output_darts"
    os.makedirs(output_folder, exist_ok=True)
    redirect_darts_output(os.path.join(output_folder, "log.txt"))

    model = DartsModel(runtime=1500.0, max_ts=10.0 / 3.0)
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


def plot_darts_reaktoro_comparison(darts_h5, reaktoro_h5, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    darts_data = _load_h5(darts_h5)
    reaktoro_data = _load_h5(reaktoro_h5, group_name="results")
    times_reaktoro = np.squeeze(reaktoro_data['Time'])
    times_darts, darts_amounts = np.squeeze(darts_data['dynamic']['time']), _compute_darts_amounts(darts_data)
    darts_properties = darts_data.get("properties", {})
    darts_ref_amount = darts_amounts['Calcite'][0]

    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 14))

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

    for name, color in aqueous_specs:
        darts_series = darts_amounts.get(name)
        reaktoro_series = reaktoro_data.get(name)
        if darts_series is None or reaktoro_series is None:
            continue
        ax[0].plot(times_darts, darts_series / darts_ref_amount, color=color, linestyle="-", label=f"DARTS {name}")
        ax[0].plot(times_reaktoro, reaktoro_series, color=color, linestyle="--", label=f"Reaktoro {name}")

    for name, color in mineral_specs:
        darts_series = darts_amounts.get(name)
        reaktoro_series = reaktoro_data.get(name)
        if darts_series is None or reaktoro_series is None:
            continue
        ax[1].plot(times_darts, darts_series / darts_ref_amount, color=color, linestyle="-", label=f"DARTS {name}")
        ax[1].plot(times_reaktoro, reaktoro_series, color=color, linestyle="--", label=f"Reaktoro {name}")

    for name, darts_rate_key, reaktoro_rate_key, color in reaction_rate_specs:
        darts_series = darts_properties.get(darts_rate_key)
        reaktoro_series = reaktoro_data.get(reaktoro_rate_key)
        if darts_series is None or reaktoro_series is None:
            continue

        conversion_mult = -1000 / 86400 # kmol/day to mol/s
        s_id = 0
        ax[2].plot(times_darts[s_id:], (np.squeeze(darts_series) * conversion_mult)[s_id:], color=color, linestyle="-", label=f"DARTS {name}")
        ax[2].plot(times_reaktoro, np.squeeze(reaktoro_series), color=color, linestyle="--", label=f"Reaktoro {name}")

    for name, darts_sr_key, reaktoro_omega_key, color in saturation_ratio_specs:
        darts_series = darts_properties.get(darts_sr_key)
        reaktoro_series = reaktoro_data.get(reaktoro_omega_key)
        if darts_series is None or reaktoro_series is None:
            continue
        ax[3].plot(times_darts, np.squeeze(darts_series), color=color, linestyle="-", label=f"DARTS {name}")
        ax[3].plot(times_reaktoro, np.squeeze(reaktoro_series), color=color, linestyle="--", label=f"Reaktoro {name}")

    ax[0].set_ylabel("Amount [mol]")
    ax[0].set_title("Aqueous Species: DARTS vs Reaktoro")
    if ax[0].lines:
        ax[0].legend()

    ax[1].set_ylabel("Amount [mol]")
    ax[1].set_title("Minerals: DARTS vs Reaktoro")
    if ax[1].lines:
        ax[1].legend()

    ax[2].set_ylabel("Reaction rate")
    ax[2].set_title("Mineral Reaction Rates: DARTS vs Reaktoro")
    if ax[2].lines:
        ax[2].legend()
    ax[2].set_ylim(-4e-7, 4e-7)

    ax[3].set_ylabel("Saturation ratio [-]")
    ax[3].set_xlabel("Time [day]")
    ax[3].set_title("Mineral Saturation Ratios: DARTS vs Reaktoro")
    if ax[3].lines:
        ax[3].legend()

    fig.tight_layout()
    comparison_plot_path = os.path.join(output_folder, "darts_vs_reaktoro_comparison.png")
    fig.savefig(comparison_plot_path)
    plt.close(fig)
    return comparison_plot_path


if __name__ == "__main__":
    model, output_folder = run_darts_simulation()
    # plot_darts_properties(model=model, output_folder=output_folder)

    # model_ref = run_reaktoro_simulation()
    # plot_reaktoro_properties(model=model_ref)


    # darts_h5 = os.path.join("output_darts", "zerod_solution.h5")
    # reaktoro_h5 = os.path.join("output_reaktoro", "reaktoro_results.h5")
    # output_folder = os.path.join("output_comparison")
    # plot_darts_reaktoro_comparison(darts_h5=darts_h5, reaktoro_h5=reaktoro_h5, output_folder=output_folder)
