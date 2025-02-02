"""

Coarse-scale, isothermal, SPE11b

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import h5py
from darts.reservoirs.mesh.geometry.map_mesh import MapMesh, _translate_curvature
from model_b import Model, PorPerm, Corey, layer_props
from darts.engines import redirect_darts_output, sim_params
from fluidflower_str_b import FluidFlowerStruct

# For each of the facies within the SPE11b model we define a set of operators in the physics.
property_regions  = [0, 1, 2, 3, 4, 5, 6]
layers_to_regions = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}

# define the Corey parameters for each layer (rock type) according to the technical description of the CSP
corey = {
    0: Corey(nw=1.5, ng=1.5, swc=0.32, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=1.935314, pcmax=300, c2=1.5),
    1: Corey(nw=1.5, ng=1.5, swc=0.14, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.08655, pcmax=300, c2=1.5),
    2: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.0612, pcmax=300, c2=1.5),
    3: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.038706, pcmax=300, c2=1.5),
    4: Corey(nw=1.5, ng=1.5, swc=0.12, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.0306, pcmax=300, c2=1.5),
    5: Corey(nw=1.5, ng=1.5, swc=0.10, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=0.025602, pcmax=300, c2=1.5),
    6: Corey(nw=1.5, ng=1.5, swc=1e-8, sgc=0.10, krwe=1.0, krge=1.0, labda=2., p_entry=1e-2, pcmax=300, c2=1.5)
}

redirect_darts_output('model.log') # redirects run.log to your directory of choice instead of prininting everything off

"""Define realization ID"""
model_specs = [
    {'structured': True,
     'thickness': False,
     'curvature': False,
     'tpfa': True,
     'capillary': True,
     'nx': 170,
     'nz': 60,
     'output_dir': 'SPE11_output'},
]

j = 0
specs = model_specs[j]
m = Model()

"""Define physics"""
zero = 1e-10
m.set_physics(corey=corey, zero=zero, temperature=323.15, n_points=1001, diff=1e-9)

# solver paramters
m.set_sim_params(first_ts=1e-2, mult_ts=2, max_ts=365, tol_linear=1e-3, tol_newton=1e-3,
                 it_linear=50, it_newton=12, newton_type=sim_params.newton_global_chop)
m.params.newton_params[0] = 0.05
m.params.nonlinear_norm_type = m.params.L1

"""Define the reservoir and wells """
well_centers = {
    "I1": [2700.0, 0.0, 300.0],
    "I2": [5100.0, 0.0, 700.0]
}

structured = specs['structured']
m.reservoir = FluidFlowerStruct(timer=m.timer, layer_properties=layer_props, layers_to_regions=layers_to_regions,
                                model_specs=specs, well_centers=well_centers) # structured reservoir

if 0:
    grid = np.meshgrid(np.linspace(0, 8400, m.reservoir.nx), np.linspace(0, 1200, m.reservoir.nz))
    plt.figure(figsize = (10, 2))
    plt.title('Porosity')
    c = plt.pcolor(grid[0], grid[1], m.reservoir.global_data['poro'].reshape(m.reservoir.nz, m.reservoir.nx))
    plt.colorbar(c)
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.show()

""" Define initial and boundary conditions """
# define initial pressure, composition and temperature of the reservoir
pres_in = 212
m.initial_values = {"pressure": pres_in,
                    "H2O": 1. - zero,
                    "temperature": 323.15}
m.gradient = {"pressure": 0.09775, "temperature": 0.0}

# define injection stream of the wells
m.inj_stream = [zero]

inj_rate = 3024 # mass rate per well, kg/day
m.inj_rate = [0, 0] # per well

m.set_str_boundary_volume_multiplier()  # right and left boundary volume multiplier

# now that your reservoir and physics is defined, you can init your DartsModel()
output_dir = specs['output_dir']
m.platform = 'cpu'
m.init(discr_type='tpfa', platform=m.platform, output_folder=output_dir, restart = True)

# equillibration step
m.run(365, verbose = False)
m.physics.engine.t = 0 # return engine time to zero

m.inj_rate = [inj_rate, 0]  # [well 1, well 2]
Nt = 20  # 100 years with output every 5 years
import time

start = time.time()
for i in range(Nt):
    m.run(5 * 365, verbose=True)  # run model for 1 year

    if m.physics.engine.t >= 25 * 365 and m.physics.engine.t < 50 * 365:
        # At 25 years, start injecting in the second well
        m.inj_rate = [inj_rate, inj_rate]

    elif m.physics.engine.t >= 50 * 365:
        # At 50 years, stop injection for both wells
        m.inj_rate = [0, 0]
stop = time.time()
print("Runtime = %3.2f sec" % (stop - start))

nx, nz = m.reservoir.nx, m.reservoir.nz  # grid dimensions

prop_list = list(m.physics.property_containers[0].output_props.keys())

M_H2O = m.physics.property_containers[0].Mw[0] # molar mass water in kg/kmol
M_CO2 = m.physics.property_containers[0].Mw[1] # molar mass CO2 in kg/kmol
PV = np.array(m.reservoir.mesh.volume)[1] * np.array(m.reservoir.mesh.poro) # pore volume

# Generate some example data to animate
time_vector, property_array = m.output_properties(output_properties = prop_list, timestep = None)
data = []
for i in range(len(time_vector)):
    # vapour mass fraction
    wco2 = property_array['yCO2'][i] * M_CO2 / (property_array['yCO2'][i] * M_CO2 + (1 - property_array['yCO2'][i]) * M_H2O)

    # mass of CO2 in the aqueous phase
    mass_co2_aq = PV * (1 - property_array['satV'][i]) * property_array['xCO2'][i] * property_array['rho_mA'][i] * M_CO2

    # mass of CO2 in the vapor phase
    mass_co2_v = PV * property_array['satV'][i] * property_array['rhoV'][i] * wco2

    # total mass of CO2 in kton
    mass_co2 = (mass_co2_aq + mass_co2_v)/1e3

    # append to data list for plotting
    data.append(mass_co2.reshape(m.reservoir.nz, m.reservoir.nx))

# from matplotlib.animation import FuncAnimation
#
# # Initialize the figure and color plot
# fig, ax = plt.subplots(figsize=(16, 4))
# pcolor_plot = ax.pcolor(grid[0], grid[1], data[0], vmin = data[1].min(), vmax = 25.0, cmap = 'cividis')
# plt.colorbar(pcolor_plot, ax=ax)
# ax.set_xlabel("x [m]")
# ax.set_ylabel("z [m]")
#
# # Define the update function for animation
# def update(frame):
#     ax.set_title(f"mCO2 @ year {frame*5}")
#     pcolor_plot.set_array(data[frame].ravel())  # Update the data
#     return pcolor_plot,
#
# # Create the animation
# ani = FuncAnimation(fig, update, frames=len(time_vector), blit=True)
#
# # Display the animation in the Jupyter Notebook
# from IPython.display import HTML
# HTML(ani.to_jshtml())