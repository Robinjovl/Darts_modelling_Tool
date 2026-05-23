import numpy as np
from darts.engines import ms_well, sim_params, value_vector, well_control_iface

from darts.models.cicd_model import CICDModel
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic
from darts.physics.properties.flash import ConstantK
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM
from darts.pipes.pipe import Pipe
from darts.pipes.set_initial_conditions import LinearAmbientTemperature
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.struct_reservoir_with_lgr import LGRPatch, StructReservoirWithLGR


class Model(CICDModel):
    def __init__(self):
        super().__init__()

        self.timer.node["initialization"].start()

        self.zero = 1e-8
        self.set_reservoir()
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.set_physics()
        self.set_sim_params(
            first_ts=1e-7,
            mult_ts=2,
            max_ts=1e-4,
            runtime=0.01,
            tol_newton=1e-3,
            tol_linear=1e-4,
            it_newton=12,
            it_linear=60,
            newton_type=sim_params.newton_local_chop,
            coupled_well_res_norm_method=2,
        )

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        parent = StructReservoir(
            self.timer,
            nx=10,
            ny=1,
            nz=1,
            dx=20.0,
            dy=20.0,
            dz=10.0,
            permx=100.0,
            permy=100.0,
            permz=10.0,
            poro=0.3,
            depth=2005.0,
            hcap=2200.0,
            rcond=120.0,
        )
        # TODO: Using large boundary cells below is required because the production well does not produce without a pump. Pump implementation is required.
        parent.boundary_volumes["yz_minus"] = 1e20
        parent.boundary_volumes["yz_plus"] = 1e20

        lgrs = [
            LGRPatch("inj_lgr", (2, 2), (1, 1), (1, 1), (7, 7, 1)),
            LGRPatch("prod_lgr", (9, 9), (1, 1), (1, 1), (7, 7, 1)),
        ]
        self.reservoir = StructReservoirWithLGR(
            self.timer,
            parent,
            lgrs,
            lgr_coarse_fine_tran_mode="flow_based",
        )

    def set_physics(self):
        epsilon = 1e-9
        components = ["CO2", "C1", "H2O"]
        phases = ["G", "L"]
        molecular_weights = [44.01, 16.04, 18.015]

        property_container = PropertyContainer(
            phases_name=phases,
            components_name=components,
            Mw=molecular_weights,
            eps_z=epsilon,
            temperature=None,
        )
        property_container.flash_ev = ConstantK(
            len(components), [4.0, 2.0, 0.1], self.zero
        )
        property_container.density_ev = {
            "G": DensityBasic(compr=1e-3, dens0=200.0),
            "L": DensityBasic(compr=1e-5, dens0=600.0),
        }
        property_container.viscosity_ev = {
            "G": ConstFunc(0.05),
            "L": ConstFunc(0.5),
        }
        property_container.rel_perm_ev = {
            "G": PhaseRelPerm("gas"),
            "L": PhaseRelPerm("oil"),
        }
        # EnthalpyBasic is multiplied by molar density in the energy operators,
        # so heat capacities are specified in kJ/kmol/K. With the old heat capacities,
        # the injection well with both WHP and rate control failed because of unexpected
        # temperature increase of the block below the top segment during injection.
        property_container.enthalpy_ev = {
            "G": EnthalpyBasic(hcap=37.0),
            "L": EnthalpyBasic(hcap=75.3),
        }
        property_container.conductivity_ev = {
            "G": ConstFunc(3.5),
            "L": ConstFunc(75.0),
        }
        property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)

        property_container.IFT_ev = IFT_multicomponent_MCM(components)

        property_container.output_props = {}
        property_container.output_props['temperature'] = lambda: (
            property_container.temperature
        )
        for j, ph in enumerate(phases):
            property_container.output_props['s' + ph] = lambda jj=j: (
                property_container.sat[jj]
            )
            property_container.output_props['rho' + ph] = lambda jj=j: (
                property_container.dens[jj]
            )
            property_container.output_props['miu' + ph] = lambda jj=j: (
                property_container.mu[jj]
            )
            for i, comp in enumerate(components):
                property_container.output_props[f'x{comp}_in_{ph}_mass'] = (
                    lambda jj=j, ii=i: property_container.x_mass[jj, ii]
                )

        self.physics = Compositional(
            components,
            phases,
            self.timer,
            state_spec=Compositional.StateSpecification.PT,
            n_points=200,
            min_p=1.0,
            max_p=300.0,
            min_z=0.0,
            max_z=1.0,
            epsilon_z=epsilon,
            min_t=273.15 + 20.0,
            max_t=273.15 + 300.0,
            extrapolation_flag=True,
        )
        self.physics.add_property_region(property_container)

    def set_wells(self):
        self.wells = {}
        self._add_dfm_well(
            well_name="I1",
            lgr_name="inj_lgr",
            lgr_cell_idx=(4, 4, 1),
        )
        self._add_dfm_well(
            well_name="P1",
            lgr_name="prod_lgr",
            lgr_cell_idx=(4, 4, 1),
        )

    def _add_dfm_well(
        self,
        well_name: str,
        lgr_name: str,
        lgr_cell_idx: tuple[int, int, int],
    ):
        segments_lengths = 50 * np.ones(40)
        segments_lengths = np.append(segments_lengths, 10)  # 10 is the reservoir thickness
        well_diameter = 0.1524
        well_geometry = PipeGeometry(
            well_name,
            segments_lengths,
            well_diameter,
            inclination_angle=0.0,
        )
        pipe_head_pressure = 1.0
        pipe_head_temperature = 25 + 273.15  # Kelvin
        temp_grad = 0.03  # deg C/meter
        pipe_head_segment_index = 0
        initial_conditions = LinearAmbientTemperature(
            pipe_name=well_name,
            pipe_geom=well_geometry,
            physics=self.physics,
            pipe_head_pressure=pipe_head_pressure,
            pipe_head_temperature=pipe_head_temperature,
            temp_grad=temp_grad,
            pipe_head_segment_index=pipe_head_segment_index,
            initial_conditions_dict={
                "phases_names": ["L"],
                "phases_compositions": [[0.01, 0.01, 0.98]],
                "pipe_intervals": [[0.0, well_geometry.pipe_length]],
            },
        )
        self.wells[well_name] = Pipe(
            well_name,
            well_geometry,
            self.physics,
            self.reservoir,
            initial_conditions,
        )

        self.reservoir.add_well(
            well_name,
            ms_well.MS_Type.DFM,
            well_geometry=well_geometry,
        )
        self.reservoir.add_perforation(
            well_name,
            lgr_name=lgr_name,
            lgr_cell_idx=lgr_cell_idx,
            well_seg_idx=well_geometry.num_segments,
            well_diameter=well_geometry.pipe_ID,
            well_indexD=None,
            with_peaceman_for_coupled_well_reservoir=True,
        )

    def set_initial_conditions(self):
        input_distribution = {
            self.physics.vars[0]: 118.74907,
            self.physics.vars[1]: 0.01,
            self.physics.vars[2]: 0.01,
            self.physics.vars[3]: 358.15000,
        }
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh,
            input_distribution=input_distribution,
        )

        for well in self.reservoir.wells:
            well.init_state = value_vector(
                self.wells[well.name].initial_conditions.initial_conditions_vector
            )

    def set_well_controls(self):
        injection_composition = [0.90, 0.099]
        for idx, well in enumerate(self.reservoir.wells):
            if idx == 0:
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.MASS_RATE,
                    is_inj=True,
                    target=5 * 24 * 60 * 60,
                    inj_composition=injection_composition,
                    inj_temp=345.0,
                )
            else:
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=1.0,
                )
