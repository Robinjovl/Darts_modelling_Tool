from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.cpg_reservoir import read_float_array, read_int_array
from darts.models.darts_model import DartsModel
from darts.engines import value_vector, ms_well
from darts.nonlinear_solvers import NewtonSolver
import numpy as np

from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic


class Model(DartsModel):
    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics()

        # solver/time-stepping configuration moved to set_solver() (called from base reset())

        self.timer.node["initialization"].stop()

    def set_solver(self):
        self.ts_control.dt_first = 0.0001
        self.ts_control.dt_min = 1e-15
        self.ts_control.dt_mult = 1.2
        self.ts_control.dt_max = 200
        self.ts_control.runtime = 30 *365
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3)
        self.linear_solver.spec.tolerance = 1e-6

    def set_reservoir(self):
        full_shape = (242, 264, 41)
        grid_start = (0, 0, 0)
        interior_nx, interior_ny, reservoir_nz = (60, 100, 40)
        flow_boundary_layers = 1
        flow_boundary_storage_multiplier = 1.0
        model_x_extent = 12464.0
        model_y_extent = 13189.0
        nx = interior_nx + 2 * flow_boundary_layers
        ny = interior_ny + 2 * flow_boundary_layers
        burden_layers = 5
        burden_layer_dz = 20.0
        reservoir_dz = 100.0 / reservoir_nz
        nz = reservoir_nz + 2 * burden_layers
        n_cells = np.prod(full_shape)
        assert all(index >= 0 for index in grid_start)
        assert all(
            start + size <= limit
            for start, size, limit in zip(
                grid_start, (interior_nx, interior_ny, reservoir_nz), full_shape
            )
        ), f'Grid section {grid_start} + {(interior_nx, interior_ny, reservoir_nz)} exceeds {full_shape}'

        facies_file = r"C:\Users\Acer\Documents\GEIP\facies_70%.GRDECL"
        permeability_file = r"C:\Users\Acer\Documents\GEIP\perm_70%.GRDECL"
        porosity_file = r"C:\Users\Acer\Documents\GEIP\eff_por_70%.GRDECL"

        def read_property(filename, keyword, dtype=float):
            if dtype is int:
                values = read_int_array(filename, keyword)
            else:
                values = read_float_array(filename, keyword)
            assert values.size == n_cells, (
                f'{keyword}: expected {n_cells} values, got {values.size}'
            )
            full_values = values.reshape(full_shape, order='F')
            i_start, j_start, k_start = grid_start
            return full_values[
                i_start:i_start + interior_nx,
                j_start:j_start + interior_ny,
                k_start:k_start + reservoir_nz,
            ]

        self.facies = read_property(facies_file, 'FACIES', dtype=int)
        permeability = read_property(
            permeability_file,
            'MODEL_PERM_70%',
        )
        poro = read_property(
            porosity_file,
            'EFF_POR_MODEL_70%',
        ) /100
        shale_porosity = 1e-5
        shale_permeability = 1e-5
        shale_facies_id = 1
        shale_mask = (
            (self.facies <= 0)
            | ~np.isfinite(poro)
            | (poro <= 0.0)
            | ~np.isfinite(permeability)
            | (permeability <= 0.0)
        )
        self.facies[shale_mask] = shale_facies_id
        poro[shale_mask] = shale_porosity
        permeability[shale_mask] = shale_permeability
        permeability = np.concatenate(
            [
                np.full(
                    (interior_nx, interior_ny, burden_layers), shale_permeability
                ),
                permeability,
                np.full(
                    (interior_nx, interior_ny, burden_layers), shale_permeability
                ),
            ],
            axis=2,
        )
        poro = np.concatenate(
            [
                np.full((interior_nx, interior_ny, burden_layers), shale_porosity),
                poro,
                np.full((interior_nx, interior_ny, burden_layers), shale_porosity),
            ],
            axis=2,
        )
        self.facies = np.concatenate(
            [
                np.full(
                    (interior_nx, interior_ny, burden_layers),
                    shale_facies_id,
                    dtype=self.facies.dtype,
                ),
                self.facies,
                np.full(
                    (interior_nx, interior_ny, burden_layers),
                    shale_facies_id,
                    dtype=self.facies.dtype,
                ),
            ],
            axis=2,
        )
        self.burden_layers = burden_layers
        self.reservoir_nz = reservoir_nz
        dz = np.concatenate(
            [
                np.full((interior_nx, interior_ny, burden_layers), burden_layer_dz),
                np.full((interior_nx, interior_ny, reservoir_nz), reservoir_dz),
                np.full((interior_nx, interior_ny, burden_layers), burden_layer_dz),
            ],
            axis=2,
        )
        lateral_padding = (
            (flow_boundary_layers, flow_boundary_layers),
            (flow_boundary_layers, flow_boundary_layers),
            (0, 0),
        )
        permeability = np.pad(permeability, lateral_padding, mode='edge')
        poro = np.pad(poro, lateral_padding, mode='edge')
        self.facies = np.pad(self.facies, lateral_padding, mode='edge')
        dz = np.pad(dz, lateral_padding, mode='edge')
        nx, ny = permeability.shape[:2]
        dx = model_x_extent / full_shape[0]
        dy = model_y_extent / full_shape[1]
        self.flow_boundary_layers = flow_boundary_layers
        self.flow_boundary_storage_multiplier = flow_boundary_storage_multiplier
        self.permeability = permeability
        self.poro = poro
        depth_layers = np.concatenate(
            [
                burden_layer_dz / 2.0
                + np.arange(burden_layers, dtype=float) * burden_layer_dz,
                100.0 + np.arange(reservoir_nz, dtype=float) * reservoir_dz,
                100.0
                + reservoir_nz * reservoir_dz
                + burden_layer_dz / 2.0
                + np.arange(burden_layers, dtype=float) * burden_layer_dz,
            ]
        )
        depth = np.broadcast_to(
            depth_layers[None, None, :],
            (nx, ny, nz),
        ).copy().flatten(order='F')
        actnum = (
            (self.facies > 0)
            & (poro > 0.0)
            & (permeability > 0.0)
        ).astype(np.int32)

        self.reservoir = StructReservoir(
            self.timer,
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,
            dy=dy,
            dz=dz,
            permx=permeability,
            permy=permeability,
            permz=permeability,
            hcap=2200,
            rcond=181.44,
            poro=poro,
            depth=depth,
            actnum=actnum,
        )
        cell_volumes = dx * dy * dz
        self.reservoir.boundary_volumes = {
            'xy_minus': None,
            'xy_plus': None,
            'yz_minus': flow_boundary_storage_multiplier * cell_volumes[0, :, :],
            'yz_plus': flow_boundary_storage_multiplier * cell_volumes[-1, :, :],
            'xz_minus': flow_boundary_storage_multiplier * cell_volumes[:, 0, :],
            'xz_plus': flow_boundary_storage_multiplier * cell_volumes[:, -1, :],
        }
        self.reservoir.global_data['facies'] = self.facies
        return

    def set_wells(self):
        active_cells = np.argwhere(self.reservoir.actnum > 0)
        assert active_cells.size > 0, 'The selected grid contains no active cells'
        boundary_offset = self.flow_boundary_layers

        def highest_quality_sand_cell(target_i, target_j, target_k, radius=2):
            reservoir_cells = active_cells[
                (active_cells[:, 2] >= self.burden_layers)
                & (
                    active_cells[:, 2]
                    < self.burden_layers + self.reservoir_nz
                )
                & (active_cells[:, 2] == target_k)
            ]
            target = np.array(
                [target_i + boundary_offset, target_j + boundary_offset]
            )
            distances = np.linalg.norm(reservoir_cells[:, :2] - target, axis=1)
            nearby = reservoir_cells[distances <= radius]
            high_porosity = nearby[
                self.poro[nearby[:, 0], nearby[:, 1], nearby[:, 2]] >= 0.05
            ]
            if high_porosity.size == 0:
                raise ValueError(
                    f'No active sand cell with porosity >= 5% within {radius} '
                    f'cells of well target ({target_i}, {target_j})'
                )
            quality = (
                self.poro[high_porosity[:, 0], high_porosity[:, 1], high_porosity[:, 2]]
                * self.permeability[
                    high_porosity[:, 0], high_porosity[:, 1], high_porosity[:, 2]
                ]
            )
            cell = high_porosity[np.argmax(quality)]
            return tuple((cell + 1).tolist())

        well_i = 20
        injector_j = 44
        producer_j = 51
        injector_reservoir_layer = 15  # E.g., deeper injection layer
        producer_reservoir_layer = 40  # E.g., shallower production layer

# 2. Calculate separate target_k values
        injector_k = self.burden_layers + injector_reservoir_layer - 1
        producer_k = self.burden_layers + producer_reservoir_layer - 1
        self.reservoir.add_well("I")
        injector_cell = highest_quality_sand_cell(well_i, injector_j, injector_k)
        self.reservoir.add_perforation("I", res_cell_idx=injector_cell)
        self.reservoir.add_well("P")
        producer_cell = highest_quality_sand_cell(well_i, producer_j, producer_k)
        self.reservoir.add_perforation("P", res_cell_idx=producer_cell)
        self.well_paths = {"I": [injector_cell], "P": [producer_cell]}

    def set_physics(self):
        """Physical properties"""
        zero = 1e-13
        epsilon = 1e-14
        components = ['w']
        phases = ['wat', 'gas']

        self.inj = value_vector([300])

        property_container = ModelProperties(phases_name=phases, components_name=components, eps_z=epsilon)

        # Define property evaluators based on custom properties
        property_container.density_ev = dict([('wat', DensityBasic(compr=1e-5, dens0=1014)),
                                              ('gas', DensityBasic(compr=5e-3, dens0=50))])
        property_container.viscosity_ev = dict([('wat', ConstFunc(0.3)),
                                                ('gas', ConstFunc(0.03))])
        property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("oil", 0.1, 0.1)),
                                               ('gas', PhaseRelPerm("gas", 0.1, 0.1))])
        property_container.enthalpy_ev = dict([('wat', EnthalpyBasic(hcap=4.18)),
                                               ('gas', EnthalpyBasic(hcap=0.035))])
        property_container.conductivity_ev = dict([('wat', ConstFunc(1.)),
                                                   ('gas', ConstFunc(1.))])
        property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)

        # create physics
        thermal = True
        state_spec = PhysicsBase.StateSpecification.PT if thermal else PhysicsBase.StateSpecification.P
        self.physics = PhysicsBase(components, phases, self.timer, state_spec=state_spec,
                                     axes_step=[5.0, 1.0],  # p [bar], T [K]
                                     axes_origin=[0.0, 273.15],
                                     epsilon_z=epsilon, extrapolation_flag=True)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 200.,
                              self.physics.vars[1]: 350.,
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        water_rate_m3_per_hour = 80.0
        water_rate = water_rate_m3_per_hour * 24.0  # DARTS rate units are m^3/day
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.VOLUMETRIC_RATE,
                    is_inj=True,
                    target=water_rate,
                    phase_name='wat',
                    inj_composition=self.inj[:-1],
                    inj_temp=self.inj[-1],
                )
            else:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.VOLUMETRIC_RATE,
                    is_inj=False,
                    target=water_rate,
                    phase_name='wat',
                )


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, eps_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw, eps_z=eps_z, temperature=None)
        self.x = np.ones((self.nph, self.nc))

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        self.pressure = vec_state_as_np[0]
        self.temperature = vec_state_as_np[-1] if self.thermal else self.temperature

        self.ph = np.array([0], dtype=np.intp)

        for j in self.ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(self.pressure, 0)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        self.nu[0] = 1
        self.compute_saturation()
        
        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc[j] = 0

        return
