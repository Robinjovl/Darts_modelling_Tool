# 1.1.3 [13-07-2024]
- Support of H5 / xarray output
- Default `*.h5` output of data required for the calculation of well rates.
- User-defined solution output to `*.h5`.
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
