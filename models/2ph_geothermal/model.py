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
        self.ts_control.dt_mult = 2
        self.ts_control.dt_max = 10
        self.ts_control.runtime = 1000
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-3)
        self.linear_solver.spec.tolerance = 1e-6

    def set_reservoir(self):
        full_shape = (242, 264, 41)
        grid_start = (0, 0, 0)
        nx, ny, reservoir_nz = (60, 100, 40)
        burden_layers = 5
        burden_layer_dz = 20.0
        reservoir_dz = 1.0
        nz = reservoir_nz + 2 * burden_layers
        n_cells = np.prod(full_shape)
        assert all(index >= 0 for index in grid_start)
        assert all(
            start + size <= limit
            for start, size, limit in zip(
                grid_start, (nx, ny, reservoir_nz), full_shape
            )
        ), f'Grid section {grid_start} + {(nx, ny, reservoir_nz)} exceeds {full_shape}'

        facies_file = r'C:\Users\Acer\OneDrive\Documenten\GEIP\Facies_model.GRDECL'
        permeability_file = r'C:\Users\Acer\OneDrive\Documenten\GEIP\Permeability.GRDECL'
        porosity_file = r'C:\Users\Acer\OneDrive\Documenten\GEIP\Porosity_effective.GRDECL'

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
                i_start:i_start + nx,
                j_start:j_start + ny,
                k_start:k_start + reservoir_nz,
            ]

        self.facies = read_property(facies_file, 'FACIES', dtype=int)
        permeability = read_property(
            permeability_file,
            'COPY_OF_COPY_OF_PERMEABILITY',
        )
        poro = read_property(
            porosity_file,
            'COPY_OF_COPY_(2)_OF_EFF_POROSITY',
        ) /100
        burden_poro = 1e-5
        burden_perm = 1e-5
        permeability = np.concatenate(
            [
                np.full((nx, ny, burden_layers), burden_perm),
                permeability,
                np.full((nx, ny, burden_layers), burden_perm),
            ],
            axis=2,
        )
        poro = np.concatenate(
            [
                np.full((nx, ny, burden_layers), burden_poro),
                poro,
                np.full((nx, ny, burden_layers), burden_poro),
            ],
            axis=2,
        )
        self.facies = np.concatenate(
            [
                np.ones((nx, ny, burden_layers), dtype=self.facies.dtype),
                self.facies,
                np.ones((nx, ny, burden_layers), dtype=self.facies.dtype),
            ],
            axis=2,
        )
        self.burden_layers = burden_layers
        self.reservoir_nz = reservoir_nz
        self.permeability = permeability
        self.poro = poro
        dz = np.concatenate(
            [
                np.full((nx, ny, burden_layers), burden_layer_dz),
                np.full((nx, ny, reservoir_nz), reservoir_dz),
                np.full((nx, ny, burden_layers), burden_layer_dz),
            ],
            axis=2,
        )
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
            dx=10.0,
            dy=10.0,
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
        self.reservoir.global_data['facies'] = self.facies
        return

    def set_wells(self):
        active_cells = np.argwhere(self.reservoir.actnum > 0)
        assert active_cells.size > 0, 'The selected grid contains no active cells'

        def nearest_active_cell(target):
            distances = np.sum((active_cells - np.asarray(target)) ** 2, axis=1)
            cell = active_cells[np.argmin(distances)]
            return tuple((cell + 1).tolist())

        def add_vertical_completion(well_name, target_i, target_j):
            perforated = set()
            for layer in range(
                self.burden_layers,
                self.burden_layers + self.reservoir_nz,
            ):
                cell = nearest_active_cell((target_i, target_j, layer))
                if cell not in perforated:
                    self.reservoir.add_perforation(well_name, res_cell_idx=cell)
                    perforated.add(cell)
            return sorted(perforated, key=lambda cell: cell[2])

        def highest_quality_cell(target_i, target_j, radius=8):
            target = np.array([target_i, target_j, 0])
            distances = np.linalg.norm(active_cells[:, :2] - target[:2], axis=1)
            nearby = active_cells[distances <= radius]
            if nearby.size == 0:
                nearby = active_cells
            quality = (
                self.poro[nearby[:, 0], nearby[:, 1], nearby[:, 2]]
                * self.permeability[nearby[:, 0], nearby[:, 1], nearby[:, 2]]
            )
            cell = nearby[np.argmax(quality)]
            return tuple((cell + 1).tolist())

        self.reservoir.add_well("I")
        injector_cells = add_vertical_completion("I", 30, 45)
        self.reservoir.add_well("P")
        producer_anchor = highest_quality_cell(30, 55)
        producer_cells = add_vertical_completion(
            "P", producer_anchor[0] - 1, producer_anchor[1] - 1
        )
        self.well_paths = {"I": injector_cells, "P": producer_cells}

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

    def set_well_controls(self, pressure_fraction=1.0):
        from darts.engines import well_control_iface
        injector_bhp = 200.0 + 50.0 * pressure_fraction
        producer_bhp = 200.0 - 50.0 * pressure_fraction
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.BHP,
                    is_inj=True,
                    target=injector_bhp,
                    inj_composition=self.inj[:-1],
                    inj_temp=self.inj[-1],
                )
            else:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=producer_bhp,
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
