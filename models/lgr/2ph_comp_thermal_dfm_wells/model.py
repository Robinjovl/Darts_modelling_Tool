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
from darts.pipes.pipe import Pipe
from darts.pipes.set_initial_conditions import SingleAmbientTemperature
from darts.reservoirs.lgr_struct_reservoir import LGRPatch, LGRStructReservoir
from darts.reservoirs.struct_reservoir import StructReservoir


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
            runtime=1e-4,
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
            depth=1000.0,
            hcap=2200.0,
            rcond=120.0,
        )
        lgrs = [
            LGRPatch("inj_lgr", (2, 2), (1, 1), (1, 1), (2, 1, 1)),
            LGRPatch("prod_lgr", (9, 9), (1, 1), (1, 1), (2, 1, 1)),
        ]
        self.reservoir = LGRStructReservoir(self.timer, parent, lgrs)

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
        property_container.enthalpy_ev = {
            "G": EnthalpyBasic(hcap=0.04),
            "L": EnthalpyBasic(hcap=4.18),
        }
        property_container.conductivity_ev = {
            "G": ConstFunc(3.5),
            "L": ConstFunc(75.0),
        }
        property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)
        property_container.output_props = {
            "temperature": lambda: property_container.temperature,
            "sG": lambda: property_container.sat[0],
            "sL": lambda: property_container.sat[1],
            "rhoG": lambda: property_container.dens[0],
            "rhoL": lambda: property_container.dens[1],
            "miuG": lambda: property_container.mu[0],
            "miuL": lambda: property_container.mu[1],
            "xCO2_in_G_mass": lambda: property_container.x_mass[0, 0],
            "xC1_in_G_mass": lambda: property_container.x_mass[0, 1],
            "xH2O_in_G_mass": lambda: property_container.x_mass[0, 2],
            "xCO2_in_L_mass": lambda: property_container.x_mass[1, 0],
            "xC1_in_L_mass": lambda: property_container.x_mass[1, 1],
            "xH2O_in_L_mass": lambda: property_container.x_mass[1, 2],
        }

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
            lgr_cell_idx=(1, 1, 1),
        )
        self._add_dfm_well(
            well_name="P1",
            lgr_name="prod_lgr",
            lgr_cell_idx=(2, 1, 1),
        )

    def _add_dfm_well(
        self,
        well_name: str,
        lgr_name: str,
        lgr_cell_idx: tuple[int, int, int],
    ):
        well_diameter = 0.1524
        well_geometry = PipeGeometry(
            well_name,
            [5.0, 5.0, 10.0],
            well_diameter,
            inclination_angle=0.0,
        )
        initial_conditions = SingleAmbientTemperature(
            pipe_name=well_name,
            pipe_geom=well_geometry,
            physics=self.physics,
            ambient_temperature=350.0,
            pipe_head_pressure=100.0,
            pipe_head_segment_index=0,
            initial_conditions_dict={
                "phases_names": ["L"],
                "phases_compositions": [[0.1, 0.2, 0.7]],
                "pipe_intervals": [[0.0, well_geometry.pipe_length]],
            },
        )
        self.wells[well_name] = Pipe(
            well_name,
            well_geometry,
            self.physics,
            self.reservoir,
            initial_conditions,
            enable_drift_velocity=False,
            enable_profile_parameter=False,
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
            self.physics.vars[0]: 100.0,
            self.physics.vars[1]: 0.1,
            self.physics.vars[2]: 0.2,
            self.physics.vars[3]: 350.0,
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
        injection_composition = [0.3, 0.2]
        for idx, well in enumerate(self.reservoir.wells):
            if idx == 0:
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=True,
                    target=101.0,
                    inj_composition=injection_composition,
                    inj_temp=345.0,
                )
            else:
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=99.0,
                )
