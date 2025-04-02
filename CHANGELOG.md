# 1.3.0 [28-02-2025]
- Unify set_initial_conditions():
  - 1) Uniform or array -> specify constant or array of values for each variable to self.physics.set_initial_conditions_from_array()
  - 2) Depth table -> specify depth table with depths and initial distributions of unknowns over depth to self.physics.set_initial_conditions_from_depth_table() 
  - DartsModel.set_uniform_conditions(): forces user to overload this method in Model() and set initial conditions according to 1) or 2)
  - PhysicsBase.set_initial_conditions_from_array() to set initial conditions uniformly/with array
  - PhysicsBase.set_initial_conditions_from_depth_table() to interpolate/calculate properties based on depth table
  - On the C++ level, there is an array initial_state that lives in the conn_mesh class, which is to be filled with the initial state of all the primary variables in PhysicsBase.set_initial_conditions_from_*()
- PhysicsBase class constructor contains StateSpecification enum to define state variables
  - Unifies P (isothermal), PT (pressure-temperature) and PH (pressure-enthalpy)
  - Compositional constructor contains state_spec variable rather than thermal (bool)
  - Geothermal is PH by default

# 1.2.2 [26-01-2025]
- Remove rock thermal operators; linear rock compressibility is ignored in rock thermal terms
- Temperature and pressure operators are added
- Add enthalpy operators to elastic super engine
- Fix in operator indexing in mechanical models
- GPU-version tests added in CI/CD
- Keep only rates, super, pze_gra and pz_cap_gra interpolators
- Breaking changes (only if InputData was used in a model):
	- The well class is modified and renamed to WellData which now contains a dictionary of Well objects
	- WellControlsConst is replaced by more generic 'WellControl's list from InputData

# 1.2.1 [25-11-2024]
- Remove "IAPWS" suffix from Geothermal PropertyContainer and predefined GeothermalIAPWS physics for backward compatibility.
- Few small changes:
	- CI/CD job updates
 	- empty list of output properties if none have been specified
	- remove SolidFlash class
	- fix to FluidFlower mesh generation
	- Mesh tags and boundary conditions are added to the input_data

# 1.2.0 [15-11-2024]
- Python 3.7-3.8 support dropped.
- Changes to physics/engines:
	- Correct approximation of diffusion fluxes (including energy)
	- Simple mechanical dispersion term
	- Predefined physics Geothermal, GeothermalPH, DeadOil and BlackOil
	- Phase velocities in structured reservoir
- Changes to OBL interpolation:
	- Barycentric linear interpolation with Delaunay triangulation
	- Support of higher-dimensional linear interpolation (requires re-compilation)
	- Tests
- Changes to linear solvers:
	- Hypre added as a submodule
	- FS_CPR preconditioner by default for mechanical tests (in configuration with iter. solvers)
	- Tests
- Changes to CPG model:
	- well perforations definition by X,Y,Z coordinates in addition to I,J,K indices
	- setting rock thermal properties based on the porosity
	- output wells and center points to separate vtk files
	- added extracted energy plot
	- added 2 options for physics: geothermal and dead oil

# 1.1.4 [07-09-2024]
- Generalize super engine physics for solid formulation[(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/112)
- Python 3.12 support
- Recover the well control and constraints in case of switching between BHP and rate control
- Fix local build script for Windows: don't turn off OpenMP if '-b' is specified

# 1.1.3 [13-07-2024]
- Support for *.h5
	- Save well states of well blocks to 'output/well_data.h5': [save well data](https://gitlab.com/open-darts/open-darts/-/blob/v1.1.3/darts/models/darts_model.py#L450)
	- Save states of all reservoir blocks to 'output/solution.h5': [save solution data](https://gitlab.com/open-darts/open-darts/-/blob/v1.1.3/darts/models/darts_model.py#L463)
- Breaking changes:\
	{- Before, input states in the `DartsModel output_properties()` function were read from engine.X}\
	{+ Now, input states in the `DartsModel output_properties()` are read from a user specified time-step in 'solution.h5'}\
	{- Before, the properties_list in the `DartsModel output_properties()` function contained variable names and property names'}\
	{+ Now, the 'properties_list' contains only the property names}\
    {- Before `DartsModel output_to_vtk()` function returned a single *.vtk file}\
    {+ Now, by default `DartsModel output_to_vtk()` returns a *.vtk file for every timestep contained in 'solution.h5'}
- Two examples of models that use the new output_functions: [example](https://gitlab.com/open-darts/darts-models/-/tree/development/publications/24_geothermal_chapter/basic_1d.py) 1, [example](https://gitlab.com/open-darts/darts-models/-/tree/development/publications/24_geothermal_chapter/basic_3d.py) 2
- Molar and phase volumetric well rate calculators.
- Support of restarts.
- Molar and phase volumetric well rate calculators.
- Support of restarts.


# 1.1.2 [12-06-2024]
- Thermo-hydro-mechanical-compositional (THMC) modeling:
 	- Coupled Multi-Point Stress and Multi-Point Flux Approximations
 	- Fully implicit thermo-poroelasticity resolved with collocated FVM and coupled with compositional multiphase transport
	- Tests to compare to Mandel, Terzaghi, two-layer Terzaghi and Bai analytics [(link))](https://gitlab.com/open-darts/open-darts/-/tree/development/models/1ph_1comp_poroelastic_analytics)
	- [Convergence test](https://gitlab.com/open-darts/open-darts/-/tree/development/models/1ph_1comp_poroelastic_convergence)
	- Interface to block-partitioned preconditioner
- Improved performance of discretization (C++)
- [InputData class](https://gitlab.com/open-darts/open-darts/-/blob/development/darts/input/input_data.py) added and used in THM tests
- C++ standard is changed from 14 to 20
- Discretizer binary type changed from shared to static library
- Enable linking to external library (iterative solvers) compiled in debug mode if compiling openDARTS in debug mode.
- Improve documentation on multi-thread version.
- Add `opmcpg` as a main dependency.

# 1.1.1 [15-03-2024]

# 1.1.0 [16-02-2024]
- Migrated to cmake build system [(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/58). We kept the old Visual Studio projects, but they will be removed later.
- Well rates in SuperEngine ("Compositional") are defined in reservoir conditions now, the units are kmol/day
- VTK output unified for all the reservoir classes [(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/79)
- Discrete Fracture mesh generation tool and model added [(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/79)
- Windows build script supports optional arguments  [(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/82)
- Removed python exposures [(See details)](https://gitlab.com/open-darts/open-darts/-/merge_requests/74)
- Breaking changes:
    - The function `darts_model run_python()` is renamed to `run()`.
	- Changes in unstructured mesh processing. It is required to specify tags in mesh file for all control elements now  (matrix, fractures and optionally boundary faces). 
	- Changes in vtk output:\
	    {- Before: input properties were saved to the first timestep vtk file -}\
        {+ Now: input properties saved to the separate "mesh.vtk" file +}\
		{- Struct/CPG: DartsModel.export_vtk(file_name, local_cell_data, global_cell_data, vars_data_dtype, export_grid_data) -}\
		{- Unstruct: UnstructReservoir.output_to_vtk(output_directory, output_filename, property_data, ith_step) -}\
        {+ DartsModel.output_to_vtk(ith_step, output_directory, output_properties) +}
	- No need to call `super().set_physics(physics)`, `super().set_reservoir()` and `super().set_physics()` in user's model: \
	    {- reservoir = UnstructReservoir(...) -}\
		{- super().set_reservoir(reservoir)-}\
        {+ self.reservoir = UnstructReservoir(...) +}
	- Discretization moved from the reservoir constructor to `init_reservoir()` method, which is called in `DartsModel.init()`.
	Thus, reservoir.mesh is not available right after the self.reservoir initialization. If it is needed, one can call `self.reservoir.init_reservoir()`. For example:
	```python	
	self.reservoir = StrucReservoir(...)
	self.reservoir.init_reservoir()
	volume = np.array(reservoir.mesh.volume, copy=False)
	# set boundary volume
	m.init()
	```
	- Adding additional properties to the report changed. Examples:\
	{- from darts.physics.super.operator_evaluator import PropertyOperators -}\
	{- props = [('Brine saturation', 'sat', 1), ('Gas saturation', 'sat', 0)] -} \
	{- physics.add_property_operators(PropertyOperators(props, property_container)) -}\
	{+ property_container.output_props = {'Brine saturation': lambda: property_container.sat[1], 'Gas saturation': lambda: property_container.sat[0]} +}
	- `property_container` and `property_container` are now dictionaries with a key=region index. Examples:			  
	{- sat = self.physics.property_operators.property_container.compute_saturation_full(state) -}\
	{- properties = self.physics.vars + self.physics.property_operators.props_name -}\
	{+ sat = self.physics.property_containers[0].compute_saturation_full(state)  # 0 - is a region index +}\
	{+ properties = self.physics.vars + self.physics.property_operators[0].props_name +}
	- The engine object moved to physics (as before, in version 1.0.4):\
	{- m.engine -}\
    {+ m.physics.engine +}

	- Wells object moved to Reservoir :\
	{- m.wells -}\
    {+ m.reservoir.wells +}

# 1.0.5 [20-11-2023]
- Adjoints with MPFA (C++ discretizer for the unstructured grid)
- MPFA for the heat conduction (C++ discretizer for the unstructured grid)
- Fast and accurate version of CPG discretizer
	- Transmissibility computation at fault cells (NNC)  
	- Set boundary volume takes into account cells inactivity  
	- Fault transmissibility multiplier  
	- Initialization with arrays dictionary  
	- Added over- and underburden layers generation  
	- Faster grid initialization and vtk export  
	- Added export to grdecl files  
- Interface for the direct control of RHS from Python
- Initial project documentation (readthedocs)
- Maximum number of equations increased to 8
- Fully functioning darts-flash
- Simple well index computation for the unstructured reservoir (Python discretizer)
- Generic structure for different engines and reservoirs
	physics_base and reservoir_base classes added
- vtk and shapely added to the requirements
- fixes for GPU version, CUDA12 and updated AMGX compatibility
- Folders reorganized
- Breaking changes:
    - Reservoir classes moved:
        - Before: from darts.models.reservoirs.struct_reservoir import StructReservoir  
        - Now:    from darts.reservoirs.struct_reservoir import StructReservoir  
    - Changes in base darts model:    
        - Before: self.mesh  
        - Now:    self.reservoir.mesh  
	- Changes in add_perforation() function of reservoir classes:  
		- Before: three arguments: i, j, k  
        - Now:    tuple of IJK indices: (i,j,k) as one argument  
		- Before: default values: well_index=-1, well_indexD=-1  
        - Now:    default values: well_index=None, well_indexD=None  
	- set_initial_conditions in DartsModel class using self.initial_values dictionary  
		- self.initial_values = {'pressure': 200, 'w': 0.001}  
	- Physics initialization:  
		- Before: self.physics = Geothermal(...); self.physics.init_physics()  
		- Now:    physics = Geothermal(...); super().set_physics(physics)  
	- Reservoir initialization:  
		- Before: self.reservoir = StructReservoir(...)  
		- Now:    reservoir = StructReservoir(...); super().set_reservoir(reservoir)  
	- Added method set_wells to the DartsModel class (function with the same name in user's model should be renamed)  
		- Before: set_boundary_conditions() # well controls were here  
		- Now:	rename it to set_wells() and added return super().set_wells() in the end of this method  
	- The 'platform' parameter moved from the engine constructor to set_physics:  
		- Before: Geothermal(..., platform='gpu')  
		- Now:	Geothermal(...); super().set_physics(physics, platform='gpu')  

# 1.0.4 [11-09-2023]
Small fixes.

# 1.0.3 [11-09-2023]
- Folders reorganized.
- Breaking changes: physics creation changed:\
	{- Before: -}
	```python
	self.physics = Geothermal(...)
	```
	{+ Now: +}
	```python
	from darts.physics.geothermal.property_container import PropertyContainer
	property_container = PropertyContainer()
	self.physics = Geothermal(...)
	self.physics.add_property_region(property_container)
	self.physics.init_physics()
	```

# 1.0.2 [30-06-2023]
- Wheels creation for Python 3.11 added.

# 1.0.0 [16-06-2023]
- Folders reorganized. 
- Breaking changes: import paths changed:\
    {- Before: -}
	```python
	from darts.models.physics.geothermal import Geothermal
	from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
	```
    {+ Now: +}
	```python
	from darts.physics.geothermal.physics import Geothermal\
	from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
	```
- Stop wheels creation for Python 3.6

# 0.1.4 [13-04-2023]
- Added heat losses from wellbore. It works by default for all thermal models.
- Added capability for connection arbitrary well segments for modeling closed-loop wells. 
- Adedd CoaxWell model which models closed loop well with surrounded reservoir. 
- Added poromechanics tests.

# 0.1.3 [06-03-2023]
Initial release.
