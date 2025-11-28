import numpy as np
import os
from model import Model
from darts.engines import redirect_darts_output
import h5py
import matplotlib.pyplot as plt


def run_simulation():
    output_folder = 'output'
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    redirect_darts_output(os.path.join(output_folder, 'log.txt'))

    m = Model()
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

def plot_properties(output_folder, h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        time = f['dynamic/time'][:]
        # props
        props_names = f['dynamic/properties_name'].asstr()[...]
        props = f['dynamic/properties'][:]
        # vars
        vars_names = f['dynamic/variable_names'].asstr()[...]
        vars = f['dynamic/X'][:]

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 8))

    ax[0].plot(time, props[:, 0, np.where(props_names == 'xCa+2')[0][0]], color='b', label='xCa2+')
    ax[0].plot(time, props[:, 0, np.where(props_names == 'xMg+2')[0][0]], color='r', label='xMg2+')
    ax[1].plot(time, vars[:, 0, np.where(vars_names == 'Solid_CaCO3')[0][0]], color='b', label='zCalcite')
    ax[1].plot(time, vars[:, 0, np.where(vars_names == 'Solid_CaMg(CO3)2')[0][0]], color='r', label='zDolomite')
    ax[1].plot(time, vars[:, 0, np.where(vars_names == 'Solid_MgCO3')[0][0]], color='g', label='zMagnesite')

    ax[0].set_ylabel('zCa2+, zMg2+')
    ax[1].set_ylabel('zCalcite, zDolomite, zMagnesite')
    ax[1].set_xlabel('Time, days')
    ax[0].legend()
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, 'properties.png'))
    plt.show()


if __name__ == '__main__':
    output_folder, h5_file_path = run_simulation()
    plot_properties(output_folder=output_folder, h5_file_path=h5_file_path)
