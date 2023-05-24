import os
import sys
sys.path.append('physics_sup/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_benchmark import Model as ModelBenchmark
from pyevtk.hl import gridToVTK


# Write to file:
def write_out_to_files(filename, dx, dy, Xn, GRAV):
    nx = int(600 / dx)
    ny = int(240 / dy)
    nb = nx * ny
    prof = int(40 / dx)
    nc = 4

    model = ModelBenchmark(grid_1D=True)
    Sg = np.zeros((nb,))
    z_caco3 = np.zeros((nb,))
    X = np.zeros((nb, 3, 2))
    pres = np.zeros((nb,))

    for ii in range(nb):
        state = Xn[ii * nc:(ii + 1)*nc]
        (sat, x, rho, rho_m, mu, kr, pc, ph) = model.property_container.evaluate(state)

        Sg[ii] = sat[0]
        pres[ii] = state[0]
        z_caco3[ii] = 1 - np.sum(state[1:])
        X[ii, 2, 0] = x[1, 2]
        X[ii, 0, 0] = x[1, 0]
        X[ii, 1, 0] = x[1, 1]

    with open(f'{filename}_x40_{GRAV}.csv', 'w+') as f:
        print('y, S_g, P_g, phi, x_H2O, x_CO2, x_Ca+CO3', file=f)
        gas_sat = (Sg.reshape(ny, nx)[:, prof - 1] + Sg.reshape(ny, nx)[:, prof]) / 2
        pres = (Xn[:nb * nc:nc].reshape(ny, nx)[:, prof - 1] +
                Xn[:nb * nc:nc].reshape(ny, nx)[:, prof]) / 2
        poro = (1 - z_caco3.reshape(ny, nx)[:, prof - 1] + 1 - z_caco3.reshape(ny, nx)[:, prof]) / 2
        xH2O = (X[:, 2, 0].reshape(ny, nx)[:, prof - 1] + X[:, 2, 0].reshape(ny, nx)[:, prof]) / 2
        xCO2 = (X[:, 0, 0].reshape(ny, nx)[:, prof - 1] + X[:, 0, 0].reshape(ny, nx)[:, prof]) / 2
        xCACO3 = (X[:, 1, 0].reshape(ny, nx)[:, prof - 1] + X[:, 1, 0].reshape(ny, nx)[:, prof]) / 2
        for ii in range(ny):
            print(
                f'{ii * dy + dy / 2}, {gas_sat[ii]}, {pres[ii] * 1e5}, {poro[ii]}, {xH2O[ii]}, {xCO2[ii]}, {xCACO3[ii]}',
                file=f)

    prof = int(190 / dy)
    with open(f'{filename}_y50_{GRAV}.csv', 'w+') as f:
        print('x, S_g, P_g, phi, x_H2O, x_CO2, x_Ca+CO3', file=f)
        gas_sat = (Sg.reshape(ny, nx)[prof - 1, :] + Sg.reshape(ny, nx)[prof, :]) / 2
        pres = (Xn[:nb * nc:nc].reshape(ny, nx)[prof - 1, :] + Xn[:nb * nc:nc].reshape(ny, nx)[prof, :]) / 2
        poro = (1 - z_caco3.reshape(ny, nx)[prof - 1, :] + 1 - z_caco3.reshape(ny, nx)[prof, :]) / 2
        xH2O = (X[:, 2, 0].reshape(ny, nx)[prof - 1, :] + X[:, 2, 0].reshape(ny, nx)[prof, :]) / 2
        xCO2 = (X[:, 0, 0].reshape(ny, nx)[prof - 1, :] + X[:, 0, 0].reshape(ny, nx)[prof, :]) / 2
        xCACO3 = (X[:, 1, 0].reshape(ny, nx)[prof - 1, :] + X[:, 1, 0].reshape(ny, nx)[prof, :]) / 2
        for ii in range (nx):
            print(f'{ii * dx + dx / 2}, {gas_sat[ii]}, {pres[ii] * 1e5}, {poro[ii]}, {xH2O[ii]}, {xCO2[ii]}, {xCACO3[ii]}', file=f)

    # Write to vtk:
    global_cell_data = dict()
    global_cell_data['S_g'] = np.array(Sg.reshape(ny, 1, nx, order='C').T, copy=True)
    global_cell_data['P_g'] = np.array(Xn[:nb * nc:nc].reshape(ny, 1, nx, order='C').T, copy=True)
    global_cell_data['phi'] = np.array((1 - z_caco3).reshape(ny, 1, nx, order='C').T, copy=True)
    global_cell_data['x_H2O'] = np.array(X[:, 2, 0].reshape(ny, 1, nx, order='C').T, copy=True)
    global_cell_data['x_CO2'] = np.array(X[:, 0, 0].reshape(ny, 1, nx, order='C').T, copy=True)
    global_cell_data['x_Ca+CO3'] = np.array(X[:, 1, 0].reshape(ny, 1, nx, order='C').T, copy=True)

    x = np.linspace(0, 600, nx + 1)
    y = np.linspace(0, 10, 1 + 1)
    z = np.linspace(240, 0, ny + 1)
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')
    gridToVTK(f'{filename}_t1000_{GRAV}', XX, YY, ZZ, cellData=global_cell_data)


# Read restart files and write to vts:
GRAV = 'grav'
nx = 240
ny = 96
FOLDER = f'DARTS_21_{nx}x{ny}_{GRAV}'
dx = 600 / nx
dy = 240 / ny
DIR = os.path.join(FOLDER, "restart.pkl")
obj = pd.read_pickle(DIR)

write_out_to_files(os.path.join(FOLDER, f'DARTS_21_{nx}x{ny}'), dx, dy, obj[1], GRAV)
