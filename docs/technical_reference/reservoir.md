# Reservoir

Subsurface reservoirs are the main bodies for the modelling in DARTS. Reservoir is usually represented by its geometry covered by some computaional grid, properties like porosity, permeability or stiffness defined in the grid, boundary conditions that define fluxes over the reservoir boundary and some other things. Along with the model, reservoir defines the parameters required for any kind of modelling in DARTS.

The kind of computational grid spanning reservoir produces two types of reservoirs: structured and unstructured. The treatment of structured reservoir can be generalized that was done by  [StructReservoir](https://gitlab.tudelft.nl/darts/darts-package/-/blob/master/darts/models/reservoirs/struct_reservoir.py) class provided in DARTS. Many models use it directly without overloading and extension built-in methods. The models working with unstructured reservoir have to provide their own implementation that is usually represented by UnstructReservoir class defined in reservoir.py script.

Open-DARTS adds additional cells for the wells (2 cells per well in non multi_segment case), one can filter them out similarly to a line below, since those additional cells are always in the end
```sh
my_array[:self.reservoir.mesh.n_blocks]
```

## Structured reservoir

## CPG reservoir (Corner-Point-Geometry)

### Active cells
This reservoir type supports inactive cells, where some grid part is not used in the simulation. That is defined in a keyword ACTNUM which contains values 0 (inactive) or 1 (active).
Additionally, cells with very low permeability can be excluded from the isothermal simulation. Input arrays, like PORO and PERMX should be defined for the full grid (NX*NY*NZ).
Computational arrays store only active cells data and have self.reservoir.mesh.n_blocks size.
For converting between full and active-cells-only arrays, one can use global_to_local or local_to_global correspondence arrays from the CPG_reservoir class:
```sh
l2g = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
g2l = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)
```
For example, the function make_full_cube converts an active-cells-only array to a full one:
```sh
array_full = np.full(g2l.size, np.nan)
array_full[l2g] = array_active
```
And next line would convert a full array to the active-cells-only array:
```sh
array_active = array_full[g2l >= 0]
```

## Unstructured reservoir
Let us describe the basic parts which unstructured reservoir must and may include.

### Constructor
In many (simple) models the constructor may get some model parameters like porosity and permeability, the type of physics and boundary conditions. In general case it becomes difficult to define reservoir parameters in model and send them to reservoir constructor. In this case all reservoir properties and associated parameters may be defined in reservoir.py.

Below we will discuss only the basic pieces of code, excluding many other lines of code.

Usually constructor starts includes the creating of conn_mesh class defined in C++ back-end:
```sh
self.mesh = conn_mesh()
```
followed by the calling the constructor of unstructured discretizer, e.g.
```sh
self.unstr_discr = UnstructDiscretizer(permx=permx, permy=permy, permz=permz, frac_aper=frac_aper, mesh_file=mesh_file)
```
mesh loading
```sh
self.unstr_discr.load_mesh()
```
and some processing
```sh
self.unstr_discr.calc_cell_information()
```
This is basic scenario for this part of constructor, which can be limited for many models. Some reservoirs require different implementation of last two functions ([unstructured][], [fluidflower][], [mpfa][], [mpsa][]) including some pre- and post-processing of reservoir parameters required for construction of unstructured discretizer, some reservoirs use different discretizer at all ([mpfa-mpsa][]). However, almost all reservoir parameters are defined (or have to be defined) in this block of code in order to initialize and run discretizer. Therefore, in the models working with multi-point approximations this block of code was placed into separate method of unstructured reservoir to be called here in the constructor([mpfa][], [mpsa][] [mpfa-mpsa][]).

Once the necessary properties prescribed and discretizer is initialized, the discretization can be run as follows
```sh
cell_m, cell_p, tran, tran_thermal = self.unstr_discr.calc_connections_all_cells(cache=False)
```
in the case of two-point flux approximation,
```sh
self.cell_m, self.cell_p, self.stencil, self.offset, self.trans = self.unstr_discr.calc_mpfa_connections_all_cells(True)
```
in the case of multi-point flux approximation,
```sh
self.pm.reconstruct_gradients_per_cell(dt)
self.pm.calc_all_fluxes_once(dt)
```
in the case of poromechanics discretizer ``pm``.

The output obtained from discretizer then can be written in files and have to be provided to ``self.mesh`` in an initialization call
```sh
self.mesh.init(...)
```
where the call and number of arguments varies with different types of physics and discretization.

Next step is to expose all the specified arrays to c++ backend. Pybind11 allows to make a numpy wrap around c++ vectors in order to copy the data.

In many models the boundary conditions are specified in the end of constructor because they do not affect the results of discretization. Generally, it is not the case and we need to specify them before running discretization ([mpfa][], [mpsa][], [mpfa-mpsa][]). Also in some models the well locations are found in the constructor. Although it is not needed for discretization, these data is used afterwards in the post-proccessing of discretization results.

The constructor represents the main functionality of ``UnstructReservoir`` class: defining the input data for discretizer, runnning discretization and passing data and results to c++ backend. Some models may introduce extra functions that called in the constructor in order to assign input for discretization, for pre- or post-processing or other purposes ([fluidflower][], [mpfa][], [mpsa][], [mpfa-mpsa][]).

### Well functions

The most of implementations of ``UnstructReservoir`` class contain a few functions that help to initialize wells. The calls
```sh
def add_well(self, name, depth)
def add_perforation(self, well, res_block, well_index)
def init_wells(self)
```
may have different implementations, but they are quite general for most of the models.

### Writing output
One of the calls
```sh
def write_to_vtk(self, ...)
def export_vtk(self, ...)
```
is usually used in order to write output in VTK format. They can have different number of arguments and different implementations specific for a particular model or reservoir. However, it can be generalized.


[unstructured]: https://gitlab.tudelft.nl/darts/darts-models/-/tree/master/Unstructured_fine "Optional Title Here"
[fluidflower]: https://gitlab.tudelft.nl/darts/darts-models/-/tree/master/fluidflower "Optional Title Here"
[mpfa]: https://gitlab.tudelft.nl/darts/darts-models/-/tree/master/multipoint/mpfa "Optional Title Here"
[mpsa]: https://gitlab.tudelft.nl/darts/darts-models/-/tree/master/multipoint/mpsa "Optional Title Here"
[mpfa-mpsa]: https://gitlab.tudelft.nl/darts/darts-models/-/tree/master/multipoint/mpfa_mpsa "Optional Title Here"
