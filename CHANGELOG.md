# 1.3.1 [07-06-2025]
- Add IPhreeqc to thirdparty dependencies, added phreeqc_dissolution model with CO2 injection and dissolution-precipitation kinetics [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/173)
- Apply_rhs_flux supported for GPU platform [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/173)
- Added .pvd file generation for unstructured meshes for Time in ParaView [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/216)
- Reconstruct Darcy velocity for water and steam phases in Geothermal engine [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/209)
- Switched to Python 3.10 by default in CI/CD pipelines
- well_indexD set to 0 by default in cpg_model [link](https://gitlab.com/open-darts/open-darts/-/merge_requests/213)

# 1.3.0 [16-05-2025]
- Unify well controls:
  - A single C++ class well_control_iface class has been generalized to handle bhp and different rate controls (inj/prod) for all engines
  - An instance of well_control_iface is created for every ms_well object upon init_rate_parameters()
  - Well controls and constraints are accessed through setters, with the aid of the PhysicsBase.set_well_controls() method  
  - The set of WellControlOperators calculates BHP, BHT and each phase's volumetric, mass and molar rates
  - An enum of class well_control_iface specifies the control/constraint type: -1) BHP, 0) MOLAR_RATE, 1) MASS_RATE, 2) VOLUMETRIC_RATE, 3) ADVECTIVE_HEAT_RATE; default is BHP
  - API of set_well_controls(): control_type: well_control_iface.WellControlType, is_inj: bool, target: float, 
                                phase_name: str, inj_composition: list, inj_temp: float, is_control: bool
  - A WellInitOperator translates PT-based controls to the given state specification (PT, PH, ...)
- Unify set_initial_conditions():
  - 1) Uniform or array -> specify constant or array of values for each variable to self.physics.set_initial_conditions_from_array()
  - 2) Depth table -> specify depth table with depths and initial distributions of unknowns over depth to self.physics.set_initial_conditions_from_depth_table() 
  - DartsModel.set_uniform_conditions(): forces user to overload this method in Model() and set initial conditions according to 1) or 2)
  - PhysicsBase.set_initial_conditions_from_array() to set initial conditions uniformly/with array
  - PhysicsBase.set_initial_conditions_from_depth_table() to interpolate/calculate properties based on depth table
  - On the C++ level, there is an array initial_state that lives in the conn_mesh class, which is to be filled with the initial state of all the primary variables in PhysicsBase.set_initial_conditions_from_*()
  - The method set_initial_conditions() initializes the mesh for **RESERVOIR BLOCKS ONLY**. Well cell initialization is done inside the engine
- PhysicsBase class constructor contains StateSpecification enum to define state variables
  - Unifies P (isothermal), PT (pressure-temperature) and PH (pressure-enthalpy)
  - Compositional constructor contains state_spec variable rather than thermal (bool)
  - Geothermal is PH by default (as before)
  - All engines have a WellInitOperator to translate PT-based well controls to the given state specification
- DartsModel.output() class now contains all output related functions of open-DARTS. Please refer to this [example](https://gitlab.com/open-darts/open-darts/-/blob/main/tutorials/output_and_restart.py?ref_type=heads) as a reference.
  - The output folder is defined in DartsModel.set_output(output_folder=output) instead of in DartsModel.init()
  - The default name for save file 'solution.h5' is changed to 'reservoir_solution.h5' and now only contains reservoir blocks instead of the entire solution vector. 
  - DartsModel.output.output_properties()'s new default behavior is to return primary variables if the properties list is not defined. If defined, only the listed primary (states) and/or secondary (properties) variables are returned as a dictionary.
  - DartsModel.output.store_well_time_data() is added to store well time data. The reported well time data include BHP, BHT, phases molar rates, phases mass rates, phases volumetric rates, components molar rates, components mass rates, and phases heat 
  rates over time. The rates are reported for each perforation as well. Total rates for each well are reported in two different ways: summation of perforations rates and rates calculated at wellhead. New naming scheme of the new well time data is explained in [documentation](https://gitlab.com/open-darts/open-darts/-/blob/development/docs/technical_reference/well_time_data_guide.md?ref_type=heads).
  - DartsModel.output.plot_well_time_data() is added for saving the figures of well time data.
  - C++ rate calculations will be replaced in the near future with python based rate calculations in the DartsModel.output(). Currently, both are available. However, it is recomended that you start transitioning to the new python rates.
- Save data to *.hdf5 with compression and/or single precision in order to manage file size. 
- Evaluate properties from engine or saved data.
- New option: DartsModel.set_output(all_phase_props=True), creates an extensive list of phase properties in the DartsModel.physics. 
- Build system:
	- The GPU configuration is supported in cmake. [Details](https://gitlab.com/open-darts/open-darts/-/merge_requests/179)
	- C++ standard was changed to C++20. [Details](https://gitlab.com/open-darts/open-darts/-/merge_requests/182)
	  - a custom dockerfile usage with gcc-13 compiler for open-DARTS compilation in gitlab pipelines (Linux only).
	  - libstc++ added to the wheels (for Linux).
	  - DARTS command line interface (CLI) added.
	- The required cmake version changed to 3.26.
- Jacobian, Newton and Timestepping:
	- The BCSR jacobain matrix class exposed to Python. [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/155/diffs?commit_id=766b876deb925ffeba1d6971402c93b80256b5a7)
	- Row-wise jacobian scaling for Geothermal engine. Turned off by default [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/155/diffs?file=611eddccefa3651138d3f99d5c57b29072c40eed#611eddccefa3651138d3f99d5c57b29072c40eed_414_492)
	- Line search for Newton iterations. Turned off by default, can be enabled with model.params.line_search = True. [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/155/diffs?file=4111fc0a9161b7a085b99b1cd0ac668b08efcff7#4111fc0a9161b7a085b99b1cd0ac668b08efcff7_585_613)
    - An optional timestep control by the variable change from the previous newton iteration [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/197) with a new DataTS Python class grouping the simulation parameters 
- Fluxes storage in arrays. Turned off by default, can be enabled by a call `engine.enable_flux_output()`. [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/155/diffs?commit_id=23fd4fdfb9cbcbce7f0f72e0874c83bccae0be73)
- Unstructured mesh 
  - A mesh generation script [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/184)
  - An option to pass .geo file [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/188/diffs?commit_id=009d0f65538d905c31ea23184ed0a33c930bed97), and .msh file will be generated using gmsh command line
- Geomechanics:
	- Convergence tests for geomechanical models [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/184)
	- Displaced fault reactivation model [added] (https://gitlab.com/open-darts/open-darts/-/merge_requests/188)
	- pvd file creation (with Time in addition to Timestep index)[added] (https://gitlab.com/open-darts/open-darts/-/merge_requests/188)
	- SPE10_mech model is enabled in pipelines.
- CPG Reservoir:
  - Fault transmissibility multiplier application fixed.
  - A test case with a fault transmissibility multiplacator [added](https://gitlab.com/open-darts/open-darts/-/blob/development/models/cpg_sloping_fault/main.py?ref_type=heads#L239).
  - The heat capacity and rock conduction storage to vtk file in the cpg model fixed.
  - Description [added](https://gitlab.com/open-darts/open-darts/-/blob/development/models/cpg_sloping_fault/README.md?ref_type=heads)
  - A case with inactive cells [added] (https://gitlab.com/open-darts/open-darts/-/merge_requests/206) to tests 
  - Reading permeability when PERMX is defined and PERMY is not defined, [fixed](https://gitlab.com/open-darts/open-darts/-/merge_requests/206)
- CPG and Struct reservoirs: 
  - Fixed a check for perforations in the inactive block.
  - Save the \*.pvd file in the associated output directory, together with \*.vtk/\*.vtu files [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/200)
- Struct Reservoir:
	- No-flow boundary conditions in reconstruction of velocities on structured grid. [Changes](https://gitlab.com/open-darts/open-darts/-/merge_requests/155/diffs?commit_id=cbdd7d97937c1c44b4ed5c947f46357dc72fa2cf) 
	- Enable the visualization of large volume from mesh.vts for StructReservoir [added](https://gitlab.com/open-darts/open-darts/-/merge_requests/198)
- Mesh processing for DFN model with the Python discretizer [optimized](https://gitlab.com/open-darts/open-darts/-/merge_requests/194).
- Breaking changes:\
	- For open-DARTS wheels installed from pip or gitlab pipelines running command was changed. This is neccessary to ensure the libstdc++ provided with open-DARTS is loaded first. This doesn't impact Windows runs. However, DARTS command line interface (CLI) can also be used on Windows. [Details](https://gitlab.com/open-darts/open-darts/-/merge_requests/182).\
	{- Before, python main.py -}\
	{+ Now, darts main.py +}
	- Physics (P, PT, PH):\
	{- Before: self.physics = Compositional(..., thermal=True/False) -}\
	{+ Now:    state_spec = Compositional.StateSpecification.P/PT/PH +}\
	{+ Compositional(..., state_spec=state_spec) +}
	- Initialization:\
	{- Before: self.initial_values = {self.physics.vars[0]: 50, self.physics.vars[1]: 0.1} -}\
	{+ Now:    def set_initial_conditions(self): +}\
	{+ input_distribution = {self.physics.vars[0]: 50, self.physics.vars[1]: 0.1 +}\
    {+ self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=input_distribution) +}
	- Well controls:\
	{- Before: well.control = self.physics.new_bhp_prod(50) -}\
	{+ Now:    BHP: self.physics.set_well_controls(wctrl=well.control, control_type=well_control_iface.BHP, is_inj=False, target=50) +}\
	{- Before: well.control = self.physics.new_rate_inj(100, inj_stream, phase_index) -} \
	{+ Now:    MOLAR_RATE, MASS_RATE, VOLUMETRIC_RATE: self.physics.set_well_controls(wctrl=well.control, control_type=well_control_iface.MOLAR_RATE, is_inj=False, target=100, phase_name="phase_name", inj_composition=inj_comp, inj_temp=inj_temp) +}
	The `inj_stream` is split into `inj_comp` and `inj_temp` for thermal models. The integer parameter `phase_index` was replaced with the string `phase_name` which is the same as specified in physics.
	To keep the consistency with the previous model, for Geothermal physics `VOLUMETRIC_RATE` should be used, and for Compositional MOLAR_RATE should be used. 
	Each of the different control types can also be uniformly applied to both Geothermal and Compositional models. 

	- Output (hdf5 and well rates):\
	{- Before: m.init(output_folder=...) -}\
			   {- m.run(100)-}\
			   {- time_data_df = pd.DataFrame.from_dict(n.physics.engine.time_data)) -}\
	{+ Now:    m.init() +}\
			   {+ m.set_output(output_folder=...) +}\
			   {+ m.run(100)+}\
			   {+ time_data_dict = m.output.store_well_time_data()+}\
               {+ time_data_df = pd.DataFrame.from_dict(time_data_dict)+}\
			   {+ m.output.plot_well_time_data(types_of_well_rates=["phases_volumetric_rates"])+}\
	
		{- Before: m.save_data_to_h5('solution') -}\
		{+ Now: m.output.save_data_to_h5('reservoir') +}
	- Timesteps: any model should have a `set_sim_params(...)` call after physics initialization and before `model.init()`, where a DataTS instance is created.
	
# 1.2.2 [27-01-2025]
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
