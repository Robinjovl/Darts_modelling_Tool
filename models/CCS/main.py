import numpy as np
from darts.engines import value_vector, redirect_darts_output
from model import Model

import matplotlib.pyplot as plt
redirect_darts_output('binary.log')


filename = 'out'

# define the model
m = Model(n_points=1001)

# init the model
m.init()

m.run_python(250)
m.print_timers()
m.print_stat()

ne = m.physics.n_vars
nb = m.reservoir.nb

properties = m.output_properties()
P = properties[:, 0]
Z1 = properties[:, 1]
T = properties[:, 2]
satV = properties[:, m.physics.property_operators.n_vars + 0]
xCO2 = properties[:, m.physics.property_operators.n_vars + 1]

x = np.cumsum(m.x_axes)
y = np.linspace(m.reservoir.nz*2, 0, m.reservoir.nz)
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(3+m.physics.thermal, 1, figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')
pres = axs[0].pcolormesh(X, Y, P.reshape(m.reservoir.nz, m.reservoir.nx))
# pres = axs[0].imshow(P.reshape(m.reservoir.nz, m.reservoir.nx))
axs[0].set_title('Pressure')
plt.colorbar(pres, ax=axs[0])
# sat = axs[1].imshow(satV.reshape(m.reservoir.nz, m.reservoir.nx))
satv = axs[1].pcolormesh(X, Y, satV.reshape(m.reservoir.nz, m.reservoir.nx))
axs[1].set_title('Gas saturation')
plt.colorbar(satv, ax=axs[1])
# conc = axs[2].imshow(xCO2.reshape(m.reservoir.nz, m.reservoir.nx))
conc = axs[2].pcolormesh(X, Y, xCO2.reshape(m.reservoir.nz, m.reservoir.nx))
axs[2].set_title('CO2 concentration')
plt.colorbar(conc, ax=axs[2])

if m.physics.thermal:
    # T = Xn[ne-1:m.reservoir.nb * ne:ne]
    # temp = axs[3].imshow(T.reshape(m.reservoir.nz, m.reservoir.nx))
    temp = axs[3].pcolormesh(X, Y, T.reshape(m.reservoir.nz, m.reservoir.nx))
    axs[3].set_title('Temperature')
    plt.colorbar(temp, ax=axs[3])
    # plt.xscale('log')
plt.show()
