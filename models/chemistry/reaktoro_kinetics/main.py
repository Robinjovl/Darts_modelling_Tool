import numpy as np
import os
from model_darts import Model as DartsModel
from model_reaktoro import Model as ReaktoroModel
from darts.engines import redirect_darts_output
import h5py
import matplotlib.pyplot as plt

from reaktplot import *

def run_darts_simulation():
    output_folder = 'output'
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    redirect_darts_output(os.path.join(output_folder, 'log.txt'))

    m = DartsModel()
    m.init(platform='cpu', n_solid=len(m.minerals))
    m.set_output(output_folder=output_folder)
    m.data_ts.dt_first = 1e-5
    dt = 10
    m.data_ts.dt_max = dt / 3
    n_steps = int(1500 / dt)

    # save data for plotting
    props_names = m.physics.property_operators[next(iter(m.physics.property_operators))].props_name
    m.output.output_properties(output_properties=props_names)

    # times = np.zeros(n_steps)
    # props = np.zeros((n_steps, m.physics.n_vars))
    # X = np.asarray(m.physics.engine.X)

    for i in range(n_steps):
        m.run(days=dt)
        m.output.output_properties(output_properties=props_names)

        # times[i] = m.physics.engine.t
        # props[i] = X[:m.physics.n_vars]
        # print(f"p={props[i][0]}\tzCalcite={props[i][1]}\tzDolomite={props[i][2]}\tzMagnesite={props[i][3]} \
        #     \tzCa={props[i][4]}\tzMg={props[i][5]}\tzC={props[i][6]}\tzO={props[i][7]}")

    m.print_timers()
    m.print_stat()

    return m.output.output_folder, m.output.sol_filepath

def plot_darts_properties(output_folder, h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        time = f['dynamic/time'][:]
        # props
        props_names = f['dynamic/properties_name'].asstr()[...]
        props = f['dynamic/properties'][:]
        # vars
        vars_names = f['dynamic/variable_names'].asstr()[...]
        vars = f['dynamic/X'][:]

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 8))

    volume = 1
    dens_m_cal = props[:, 0, np.where(props_names == 'dens_m_solid_CaCO3')[0][0]]
    dens_m_dol = props[:, 0, np.where(props_names == 'dens_m_solid_CaMg(CO3)2')[0][0]]
    dens_m_mag = props[:, 0, np.where(props_names == 'dens_m_solid_MgCO3')[0][0]]
    sat_cal = props[:, 0, np.where(props_names == 'sat_CaCO3')[0][0]]
    sat_dol = props[:, 0, np.where(props_names == 'sat_CaMg(CO3)2')[0][0]]
    sat_mag = props[:, 0, np.where(props_names == 'sat_MgCO3')[0][0]]

    z_cal = vars[:, 0, np.where(vars_names == 'Solid_CaCO3')[0][0]]
    z_dol = vars[:, 0, np.where(vars_names == 'Solid_CaMg(CO3)2')[0][0]]
    z_mag = vars[:, 0, np.where(vars_names == 'Solid_MgCO3')[0][0]]

    porosity = props[:, 0, np.where(props_names == 'porosity')[0][0]]
    dens_m_liq = props[:, 0, np.where(props_names == 'dens_m_liq')[0][0]]
    dens_m_gas = props[:, 0, np.where(props_names == 'dens_m_gas')[0][0]]
    sat_liq = props[:, 0, np.where(props_names == 'sat_liq')[0][0]]
    sat_gas = props[:, 0, np.where(props_names == 'sat_gas')[0][0]]
    y_co2 = props[:, 0, np.where(props_names == 'y_CO2(g)')[0][0]]
    x_h2o = props[:, 0, np.where(props_names == 'x_H2O(aq)')[0][0]]

    n_cal = dens_m_cal * sat_cal * volume * 1000
    n_dol = dens_m_dol * sat_dol * volume * 1000
    n_mag = dens_m_mag * sat_mag * volume * 1000
    n_co2 = porosity * sat_gas * dens_m_gas * y_co2 * volume * 1000
    n_h2o = porosity * sat_liq * dens_m_liq * x_h2o * volume * 1000

    ax[0].plot(time, props[:, 0, np.where(props_names == 'x_Ca+2')[0][0]], color='b', label='xCa2+')
    ax[0].plot(time, props[:, 0, np.where(props_names == 'x_Mg+2')[0][0]], color='r', label='xMg2+')
    ax[1].plot(time, n_cal, color='b', label='Calcite')
    ax[1].plot(time, n_dol, color='r', label='Dolomite')
    ax[1].plot(time, n_mag, color='g', label='Magnesite')
    ax[1].plot(time, n_co2, color='orange', label='CO2(g)')
    ax[1].plot(time, n_h2o, color='c', label='H2O(aq)')
    # ax[1].plot(time, z_cal, color='b', label='zCalcite')
    # ax[1].plot(time, z_dol, color='r', label='zDolomite')
    # ax[1].plot(time, z_mag, color='g', label='zMagnesite')

    ax[0].set_ylabel('xCa2+, xMg2+')
    # ax[1].set_ylabel('zCalcite, zDolomite, zMagnesite')
    ax[1].set_ylabel('Amount [mol]')
    ax[1].set_xlabel('Time, days')
    ax[0].legend()
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'properties.png'))
    plt.show()

def run_reaktoro_simulation():
    m = ReaktoroModel()
    m.run()
    return m

def plot_reaktoro_properties(model):
    # Mg2+ and Ca2+ over time
    fig = Figure()
    fig.title("AQUEOUS SPECIES AMOUNTS OVER TIME")
    fig.xaxisTitle("Time [day]")
    fig.yaxisTitle("Amount [mol]")
    fig.drawLine(model.table["Time"], model.table["Ca+2"], "Ca<sup>2+</sup>")
    fig.drawLine(model.table["Time"], model.table["Mg+2"], "Mg<sup>2+</sup>")
    # fig.show()
    fig.save(file=os.path.join(model.output_folder, 'reaktoro_properties.png'))

    # Calcite, Dolomite, Magnesite over time
    fig = Figure()
    fig.title("MINERALS AMOUNTS OVER TIME")
    fig.xaxisTitle("Time [day]")
    fig.yaxisTitle("Amount [mol]")
    fig.drawLine(model.table["Time"], model.table["Calcite"], "Calcite")
    fig.drawLine(model.table["Time"], model.table["Dolomite"], "Dolomite")
    fig.drawLine(model.table["Time"], model.table["Magnesite"], "Magnesite")
    fig.save(file=os.path.join(model.output_folder, 'reaktoro_minerals.png'))

    # pH over time
    fig = Figure()
    fig.title("PH OVER TIME")
    fig.xaxisTitle("Time [day]")
    fig.yaxisTitle("pH")
    fig.drawLine(model.table["Time"], model.table["pH"], "pH")
    fig.save(file=os.path.join(model.output_folder, 'reaktoro_pH.png'))

    # Reaction rates over time
    fig = Figure()
    fig.title("REACTION RATES OVER TIME")
    fig.xaxisTitle("Time [day]")
    fig.yaxisTitle("Reaction rate [mol/s]")
    fig.drawLine(model.table["Time"], model.table["RateCalcite"], "RateCalcite")
    fig.drawLine(model.table["Time"], model.table["RateDolomite"], "RateDolomite")
    fig.drawLine(model.table["Time"], model.table["RateMagnesite"], "RateMagnesite")
    fig.save(file=os.path.join(model.output_folder, 'reaktoro_reaction_rates.png'))


if __name__ == '__main__':
    # open-darts simulation
    output_folder_darts, h5_file_path_darts = run_darts_simulation()
    plot_darts_properties(output_folder=output_folder_darts, h5_file_path=h5_file_path_darts)

    # reaktoro simulation
    # model = run_reaktoro_simulation()
    # plot_reaktoro_properties(model=model)
