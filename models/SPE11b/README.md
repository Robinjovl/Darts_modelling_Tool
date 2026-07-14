# SPE11b Carbon Capture and Storage Benchmark
The SPE11 comparative solution project (CSP) aims to provide a reference case for the development of numerical simulation of GCS and offers a baseline for simulation of CO2 storage in aquifers.
The reservoir is a heterogeneous reservoir storage complex reminicent of the Norwegian continental shelf.
Three versions of the benchmark are presented in the CSP. The second, 11b, is a 2D model at reservoir scale and conditions.
The SPE11 CSP explicitly specifies all reservoir and fluid properties in Nordbotten et al. (2024).
This repository contains a model implementation for the SPE11b utilizing the Delft Advanced Research Terra Simulator (DARTS) of Delft University of Technology.

## Running the Simulation

Simulation parameters are configured using the `model_specs` list, where each entry is a dictionary describing a specific realization.
The reporting grid of the SPE11b corresponds to a grid block of 10m by 10m and thus contains approx. 100K grid blocks.

```python
Nt = 50
Dt = 365
nx = 840//10
nz = 120//10

model_specs = [
    'check_rates': True,
    'temperature': 273.15 + 40.,
    '1000years': False,
    'RHS': True,
    'components': ['CO2', 'H2O'],
    'inj_stream': [0.99, 0.01, 283.15],
    'nx': nx,
    'nz': nz,
    'dispersion': True,
    'output_dir': 'OUTPUT',
    'post_process': None,
    'platform': 'cpu'},
]
```

| Option         | Description                                                                                                                                                        |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `check_rates`  | If `True`, checks mass per component in place before and after each timestep.                                                                                      |
| `temperature`  | Temperature in Kelvin specified at the top boundary of the reservoir; if specified the model runs isothermally. if `None` the model will be non-isothermal.        |
| `1000years`    | Integer value that represents the no. of years for initialization.                                                                                                 |
| `RHS`          | If `True`, uses a right-hand side correction to introduce mass to the well block. If `False`, `DartsModel.physics.set_well_controls()` is used with mass rates.    |
| `components`   | List of components in the simulation (e.g., `['CO2', 'H2O']`).                                                                                                     |
| `inj_stream`   | List with component injection molar compositions and injection temperature. Note: the injection temperature is ignored in isothermal conditions.                   |
| `nx`           | Horizontal resolution of the model grid.                                                                                                                           |
| `nz`           | Vertical resolution of the model grid.                                                                                                                             |
| `dispersion`   | If `True`, enables dispersion with dispersivity set to 10; If `False`, dispersion is excluded.                                                                     |
| `output_dir`   | String specifying the the relative output directory for simulation outputs. If `None`, a directory is automatically generated based on the simulation specs.       |
| `post_process` | If a string is provided e.g. `"post"`, instead of running the model results will be post-processes into `"output_dir\post\..."` instead of running the simulation. |
| `platform`     | User can specify `"CPU"` or `"GPU"` to run the model on cpu or graphics card.                                                                                      |

## Output and plots
The simulation results are saved in HDF5 files. These files can be viewed using an HDF5 viewer. The ``reservoir_solution.h5``
file contains the state variables of the reservoir blocks under ``/dynamic/X``, additionally properties are evaluated
with the DARTS interpolators and saved under the group ``/properties``. The inside of ``reservoir_solution.h5`` looks like:

![res_sol](Images/reservoir_solution_screenshot.png "res_sol")

Results can be processed and turned into ``solution_ts0.vtk`` files. For example, the initial conditions are visualized by calling,
```python
    time_vector, property_array = m.output.output_properties(ts_idx=0)
    m.output.output_to_vtk(ith_step = 0, output_data = [time_vector, property_array])
```
, where the function ``output_properties()`` evaluates properties into a ``property_array`` dictionary that can be further
utilized directly for plotting or turned into a ``solution.vtk`` file for viewing.

## DartsModel()
The DartsModel() of the SPE11 combines the different parts of the model definition. Including the reservoir object, physics object,
boundary and initial conditions of the model. All of these elements are required for any ``DartsModel()``.

### DartsModel.set_reservoir()
Seven different facies are defined for SPE11b. Facies 1 represents the storage
complex and serves as a capillary barrier to migrating CO2. Facies 2 through 5 consist of permeable reservoir
sands, while Facies 6 corresponds to fault infill. Finally, Facies 7 forms an impermeable barrier. In our model, the
geometry of the FluidFlower model is converted to the reservoir scale following the SPE11’s description. For all
simulations, a structured mesh is constructed and populated according to the facies descriptions.

![op_num](Images/op_num.png "op_num")

![porosity](Images/porosity.png "porosity")

In model 11b, z=0 is defined at the bottom of the reservoir and x=0 at the left edge.
```python
    self.reservoir = FluidFlowerStruct(timer=m.timer, layer_properties=layer_props, layers_to_regions=layers_to_regions,
                                    model_specs=specs, well_centers=well_centers)
```
### DartsModel.set_wells()
There are two wells. Well 1 is located at (x=2700m, z=300m), and well 2 at (x=5100m, z=1100m).
The wells are defined at their `well centers`, per their coordinates in meters, and passed to the reservoir object.
```python
    well_centers = {
            "I1": [2700.0, 0.0, 300.0],
            "I2": [5100.0, 0.0, 700.0]
            }
```
### DartsModel.set_well_controls()
Injection starts in well 1 at t = 0yr and continues until t = 50yr. Well 2 starts injection at t = 25yr, lasting until t = 50yr. In the post-injection period,
the simulation continues until t = 1000yr. In both wells, CO2 is injected at 10degC. The injection rate is equal to 3024kg/day.
The rate is specified in `self.inj_rate` as a list a where the first and second entry correspond to the mass injection rate of wells 1 and 2.
The injection stream/composition specifies the molar composition of the injected fluid where the entries correspond to component list and the last
entry the injection temperature.
```python
    self.inj_stream = [0.01, 0.99, 283.15] # [zH20, zCO2, Injection temp. in Kelvin]
    self.inj_rate = [inj_rate, inj_rate] # [mass rate well 'I1', mass rate well 'I2'] in kg/day
```
Two options, are included for controling CO2 injection, RHS-correction and ``DartsModel.physics.set_well_controls()``.
If ``self.specs['RHS'] = True``, boundary conditions are applied directly using ``DartsModel.set_rhs_flux()``.
Otherwise the `DartsModel.set_well_controls()` function,
```python
    for i, w in enumerate(self.reservoir.wells):
        self.physics.set_well_controls(wctrl = w.control,
                                       control_type = well_control_iface.MASS_RATE,
                                       is_inj = True,
                                       target = self.inj_rate[i],
                                       phase_name = 'V',
                                       inj_composition = self.inj_stream[:-1],
                                       inj_temp = self.inj_stream[-1]
                                       )
        print(f'Set well {w.name} to {self.inj_rate[i]} kg/day with {self.inj_stream[:-1]} {self.components[:-1]} at 10°C...')
```
is used where each well object gets its own explicit control and is represented by a well block connected to the reservoir block.

### DartsModel.set_initial_conditions()
In the SPE11b the temperatures at the top and bottom boundaries are fixed, all boundaries are impermeable and the boundary
volumes are increased to 5e4Δz. A geothermal gradient of 25 degC/km is applied along with a temperature of 70 degC at the bottom boundary while pressure is hydrostatic.
Initial conditions are set in ``DartsModel.set_initial_conditions()`` where,
```python
    self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
                                                         input_depth=input_depths,
                                                         input_distribution=self.input_distribution)
```
, specifies the initial conditions according to the input_depths and input_distributions. In the case of rhs_correction the
temperature boundary conditions are included in the model by calling `DartsModel.set_top_bot_temp()`,
```python
    def set_top_bot_temp(self):
        nv = self.physics.n_vars
        for bot_cell in self.reservoir.bot_cells:
            # T = 70 - 0.025 * z  - origin at bottom
            T_spec_bot = 273.15 + 70 - self.reservoir.centroids[bot_cell, 2] * 0.025
            target_cell = bot_cell*nv+nv-1
            self.physics.engine.X[target_cell] = T_spec_bot

        for top_cell in self.reservoir.top_cells:
            # T = 70 - 0.025 * z  - origin at bottom
            T_spec_top = 273.15 + 70 - self.reservoir.centroids[top_cell, 2] * 0.025
            target_cell = top_cell*nv+nv-1
            self.physics.engine.X[target_cell] = T_spec_top
        return
```
### DartsModel.set_physics()
#### Thermodynamics
Thermodynamics properties are provided by the DARTS-flash python module.
A negative flash procedure with successive substitution is employed for resolving thermodynamic equilibrium
calculations (Michelsen, 1982; Whitson and Michelsen, 1989). The fugacities of the vapor phase are evaluated
using a cubic equation of state (Peng and Robinson, 1976) and the fugacities of the water phase are calculated
using an activity model based on Henry’s constants (Ziabakhsh-Ganji and Kooi, 2012). The property correlations for SPE11
and open-DARTS are presented in the following table.
![properties](Images/properties.PNG "properties")

Physics are defined in the ``DartsModel.physics`` object,
```python
    nz = len(self.components) - 1
    ax_step = [0.25] + [1e-3] * nz
    ax_origin = [200.0] + [self.zero / 10] * nz
    if thermal:
        ax_step.append(0.1)
        ax_origin.append(273.15)

    self.physics = PhysicsBase(self.components, phases, timer=self.timer,
                                 axes_step=ax_step, axes_origin=ax_origin,
                                 epsilon_z=self.zero / 10,
                                 state_spec=state_spec,
                                 extrapolation_flag=False,
                                 cache=False)
```
, where the unbounded OBL grid is defined by per-axis origin and step. To the physics object we add property correlations using ``DartsModel.physics.property_containers`` per region/facies:
```python
    for i, (region, corey_params) in enumerate(corey.items()):
        property_container = PropertyContainer(components_name=self.components, phases_name=phases, Mw=comp_data.Mw,
                                               eps_z=zero / 10, temperature=temperature)

        property_container.flash_ev = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])
        property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ('Aq', Garcia2001(self.components)), ])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('Aq', Islam2012(self.components)), ])
        property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(nc) * diff_g)),
                                                ('Aq', ConstFunc(np.ones(nc) * diff_w))])
        property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                               ('Aq', EoSEnthalpy(eos=aq)), ])
        property_container.conductivity_ev = dict([('V', ConstFunc(8.4)),
                                                   ('Aq', ConstFunc(170.)),])
        property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey_params, 'V')),
                                               ('Aq', ModBrooksCorey(corey_params, 'Aq'))])
        property_container.capillary_pressure_ev = ModCapillaryPressure(corey_params)
        self.physics.add_property_region(property_container, i)
```

#### Dispersion
To include dispersion use ``DartsModel.init_disperion()``. The velocity vector is reconstructed at cell centers using a least-squares
solution of fluxes across all cell’s interfaces, then averaged between neighboring cells and explicitly incorporated
into the numerical approximation of the dispersion term.
