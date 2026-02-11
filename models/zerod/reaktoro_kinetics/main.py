import os

import matplotlib.pyplot as plt
from darts.engines import redirect_darts_output

from model_darts import Model as DartsModel
from model_reaktoro import Model as ReaktoroModel
from reaktplot import Figure


def _find_key(keys, *patterns):
    low_to_key = {k.lower(): k for k in keys}
    for pattern in patterns:
        exact = low_to_key.get(pattern.lower())
        if exact is not None:
            return exact
    for pattern in patterns:
        p = pattern.lower()
        for key in keys:
            if p in key.lower():
                return key
    return None


def run_darts_simulation():
    output_folder = "output"
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
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 8))

    ca_key = _find_key(prop_keys, "x_Ca+2")
    mg_key = _find_key(prop_keys, "x_Mg+2")
    if ca_key is not None:
        ax[0].plot(times, props[ca_key], color="b", label=ca_key)
    if mg_key is not None:
        ax[0].plot(times, props[mg_key], color="r", label=mg_key)

    volume = 1.0
    poro_key = _find_key(prop_keys, "porosity")
    dens_liq_key = _find_key(prop_keys, "dens_m_liq")
    dens_gas_key = _find_key(prop_keys, "dens_m_gas")
    sat_liq_key = _find_key(prop_keys, "sat_liq")
    sat_gas_key = _find_key(prop_keys, "sat_gas")
    y_co2_key = _find_key(prop_keys, "y_CO2(g)", "y_CO2")
    x_h2o_key = _find_key(prop_keys, "x_H2O(aq)", "x_H2O")

    amount_specs = [
        ("Calcite", "tab:blue", "dens_m_solid_Calcite", "sat_Calcite"),
        ("Dolomite", "tab:red", "dens_m_solid_Dolomite", "sat_Dolomite"),
        ("Magnesite", "tab:green", "dens_m_solid_Magnesite", "sat_Magnesite"),
        ("CO2(g)", "tab:orange", dens_gas_key, sat_gas_key),
        ("H2O(aq)", "tab:cyan", dens_liq_key, sat_liq_key),
    ]

    for label, color, dens_name, sat_name in amount_specs:
        dens_key = _find_key(prop_keys, dens_name) if dens_name else None
        sat_key = _find_key(prop_keys, sat_name) if sat_name else None
        if dens_key is None or sat_key is None:
            continue

        if label == "CO2(g)":
            if poro_key is None or y_co2_key is None:
                continue
            amount = (
                props[poro_key]
                * props[sat_key]
                * props[dens_key]
                * props[y_co2_key]
                * volume
                * 1000.0
            )
        elif label == "H2O(aq)":
            if poro_key is None or x_h2o_key is None:
                continue
            amount = (
                props[poro_key]
                * props[sat_key]
                * props[dens_key]
                * props[x_h2o_key]
                * volume
                * 1000.0
            )
        else:
            amount = props[dens_key] * props[sat_key] * volume * 1000.0

        ax[1].plot(times, amount, color=color, label=label)

    ax[0].set_ylabel("xCa2+, xMg2+")
    ax[1].set_ylabel("Amount [mol]")
    ax[1].set_xlabel("Time, days")
    ax[0].legend()
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "properties.png"))
    plt.close(fig)


def run_reaktoro_simulation():
    model = ReaktoroModel()
    model.run()
    return model


def plot_reaktoro_properties(model):
    fig = Figure()
    fig.title("AQUEOUS SPECIES AMOUNTS OVER TIME")
    fig.xaxisTitle("Time [day]")
    fig.yaxisTitle("Amount [mol]")
    fig.drawLine(model.table["Time"], model.table["Ca+2"], "Ca<sup>2+</sup>")
    fig.drawLine(model.table["Time"], model.table["Mg+2"], "Mg<sup>2+</sup>")
    fig.save(file=os.path.join(model.output_folder, "reaktoro_properties.png"))

    fig = Figure()
    fig.title("MINERALS AMOUNTS OVER TIME")
    fig.xaxisTitle("Time [day]")
    fig.yaxisTitle("Amount [mol]")
    fig.drawLine(model.table["Time"], model.table["Calcite"], "Calcite")
    fig.drawLine(model.table["Time"], model.table["Dolomite"], "Dolomite")
    fig.drawLine(model.table["Time"], model.table["Magnesite"], "Magnesite")
    fig.save(file=os.path.join(model.output_folder, "reaktoro_minerals.png"))

    fig = Figure()
    fig.title("PH OVER TIME")
    fig.xaxisTitle("Time [day]")
    fig.yaxisTitle("pH")
    fig.drawLine(model.table["Time"], model.table["pH"], "pH")
    fig.save(file=os.path.join(model.output_folder, "reaktoro_pH.png"))

    fig = Figure()
    fig.title("REACTION RATES OVER TIME")
    fig.xaxisTitle("Time [day]")
    fig.yaxisTitle("Reaction rate [mol/s]")
    fig.drawLine(model.table["Time"], model.table["RateCalcite"], "RateCalcite")
    fig.drawLine(model.table["Time"], model.table["RateDolomite"], "RateDolomite")
    fig.drawLine(model.table["Time"], model.table["RateMagnesite"], "RateMagnesite")
    fig.save(file=os.path.join(model.output_folder, "reaktoro_reaction_rates.png"))


if __name__ == "__main__":
    model, output_folder = run_darts_simulation()
    plot_darts_properties(model=model, output_folder=output_folder)

    # model_ref = run_reaktoro_simulation()
    # plot_reaktoro_properties(model=model_ref)
