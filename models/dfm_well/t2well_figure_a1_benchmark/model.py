import numpy as np

from darts.models.darts_model import DartsModel
from darts.engines import sim_params, ms_well, value_vector
from darts.reservoirs.struct_radial_reservoir import StructRadialReservoir

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity
from darts.physics.properties.flash import Flash

from dartsflash.libflash import EoS
from dartsflash.components import CompData
from dartsflash.mixtures import VL

from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.set_initial_conditions import SingleAmbientTemperature
from darts.pipes.ramp_up_rate import RampUpRate
from darts.pipes.pipe import Pipe
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM


class ImmiscibleCO2WaterFlash(Flash):
    """
    Immiscible two-phase flash for the Figure A1 analytical benchmark.

    The analytical solution does not allow interphase component exchange:
    gas is pure CO2 and liquid is pure H2O. The overall CO2 mole fraction
    therefore directly sets the gas-phase mole amount.
    """

    def __init__(self, eps):
        super().__init__(nph=2, nc=2)
        self.eps = eps

    def evaluate(self, pressure, temperature, zc):
        z_co2 = float(np.clip(zc[0], self.eps, 1.0 - self.eps))
        self.nu = np.array([z_co2, 1.0 - z_co2])
        self.X = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        self.temperature = temperature
        return 0


class Model(DartsModel):
    """
    Figure A1 verification case from Pan et al. (2011), Appendix A.

    The paper solves steady isothermal upward CO2/water flow in a 1000 m
    vertical wellbore. A tiny high-volume top segment is used here only to
    impose the fixed outlet pressure without adding a reservoir-flow boundary.
    """

    def __init__(self):
        super().__init__()
        self.timer.node["initialization"].start()

        self.zero = 1.0e-10
        self.well_name = "I1"
        self.well_id_m = 0.1
        self.well_length_m = 1000.0
        self.physical_segments = 101
        self.physical_segment_length_m = 10.0
        self.top_boundary_length_m = 1.0e-6
        self.top_boundary_volume_m3 = 1.0e20
        self.wall_roughness_m = 2.4e-5
        self.temperature_k = 40.0 + 273.15
        self.top_pressure_bar = 1.0
        self.mass_rate_co2_kg_s = 0.19625
        self.mass_rate_h2o_kg_s = 0.19625

        self.set_reservoir()
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.set_physics()
        self.set_injection_rate()

        self.set_sim_params(
            first_ts=0.0001 / (24 * 60 * 60),
            mult_ts=2.0,
            max_ts=5.0,
            tol_newton=1.0e-3,
            tol_linear=1.0e-4,
            it_newton=10,
            it_linear=10,
            newton_type=sim_params.newton_local_chop,
            coupled_well_res_norm_method=2,
            runtime=0.456869e9 / (24 * 60 * 60),
        )

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        poro = np.ones((2, 1)) * 0.2
        perm = np.ones((2, 1)) * 100.0
        self.reservoir = StructRadialReservoir(
            self.timer,
            nr=2,
            nz=1,
            dr=5.0,
            dz=10.0,
            poro=poro.flatten(order="F"),
            permr=perm.flatten(order="F"),
            permz=perm.flatten(order="F"),
            R0=self.well_id_m / 2.0,
            R1=10.0,
            logspace=True,
            depth=self.well_length_m - 0.5 * self.physical_segment_length_m,
        )
        self.reservoir.boundary_volumes["yz_minus"] = 1.0e20

    def set_physics(self):
        components_names = ["CO2", "H2O"]
        phases_names = ["G", "L"]
        comp_data = CompData(components_names, setprops=True)
        epsilon = self.zero / 10.0

        self.physics = Compositional(
            components_names,
            phases_names,
            self.timer,
            state_spec=Compositional.StateSpecification.P,
            n_points=10000,
            min_p=1.0,
            max_p=500.0,
            min_z=0.0,
            max_z=1.0,
            epsilon_z=epsilon,
            min_t=150.0,
            max_t=500.0,
        )

        property_container = PropertyContainer(
            phases_names,
            components_names,
            Mw=comp_data.Mw,
            eps_z=epsilon,
            temperature=self.temperature_k,
            rock_comp=0.0,
        )

        flash_ev = VL(comp_data)
        flash_ev.set_vl_eos("PR", root_order=[EoS.STABLE])
        pr = flash_ev.eos["VL"]

        property_container.flash_ev = ImmiscibleCO2WaterFlash(epsilon)

        property_container.density_ev = {
            "G": EoSDensity(eos=pr, Mw=comp_data.Mw),
            "L": Garcia2001(components_names),
        }
        property_container.viscosity_ev = {
            "G": Fenghour1998(),
            "L": Islam2012(components_names),
        }
        property_container.rel_perm_ev = {
            "G": PhaseRelPerm("gas", swc=0.0, sgr=0.0, n=1.0),
            "L": PhaseRelPerm("oil", swc=0.0, sgr=0.0, n=1.0),
        }
        property_container.IFT_ev = IFT_multicomponent_MCM(components_names)

        self.physics.add_property_region(property_container)

        property_container.output_props = {
            "temperature": lambda: property_container.temperature,
        }
        for j, ph in enumerate(phases_names):
            property_container.output_props["s" + ph] = (
                lambda jj=j: property_container.sat[jj]
            )
            property_container.output_props["rho" + ph] = (
                lambda jj=j: property_container.dens[jj]
            )
            property_container.output_props["mu" + ph] = (
                lambda jj=j: property_container.mu[jj]
            )
            for i, comp in enumerate(components_names):
                property_container.output_props[f"x{comp}_in_{ph}_mass"] = (
                    lambda jj=j, ii=i: property_container.x_mass[jj, ii]
                )

    def set_injection_rate(self):
        mw = np.asarray(self.physics.property_containers[0].Mw)
        component_mass_rates = np.array(
            [self.mass_rate_co2_kg_s, self.mass_rate_h2o_kg_s]
        )
        component_molar_rates = component_mass_rates / mw * 24.0 * 60.0 * 60.0
        self.injection_component_molar_rates_kmol_day = component_molar_rates
        self.injection_total_molar_rate_kmol_day = float(np.sum(component_molar_rates))
        self.injection_composition = (
            component_molar_rates / self.injection_total_molar_rate_kmol_day
        )

    def set_wells(self):
        segment_lengths = np.concatenate(
            (
                [self.top_boundary_length_m],
                self.physical_segment_length_m * np.ones(self.physical_segments),
            )
        )
        self.bottom_source_segment_idx = len(segment_lengths) - 1

        geometry = PipeGeometry(
            self.well_name,
            segment_lengths,
            self.well_id_m,
            wall_roughness=self.wall_roughness_m,
            verbose=True,
        )

        initial_conditions_dict = {
            "phases_names": ["G"],
            "phases_compositions": [self.injection_composition.tolist()],
            "pipe_intervals": [[0.0, geometry.pipe_length]],
        }
        initial_conditions = SingleAmbientTemperature(
            self.well_name,
            geometry,
            self.physics,
            self.temperature_k,
            self.top_pressure_bar,
            0,
            initial_conditions_dict,
            verbose=True,
        )
        self._apply_figure_a1_initial_guess(initial_conditions)

        ramp_up_rate = RampUpRate(
            self.well_name,
            geometry,
            self.physics,
            self.data_ts.dt_first,
            self.bottom_source_segment_idx,
            "inflow",
            self.injection_total_molar_rate_kmol_day,
            0.0,
            {
                "composition": self.injection_composition,
                "phase_mass_rates": {
                    "G": self.mass_rate_co2_kg_s,
                    "L": self.mass_rate_h2o_kg_s,
                },
            },
            verbose=True,
        )

        self.wells = {
            self.well_name: Pipe(
                self.well_name,
                geometry,
                self.physics,
                self.reservoir,
                initial_conditions,
                source_sinks={"bottom_injection": ramp_up_rate},
                Cmax=1.0,
                enable_profile_parameter=False,
                verbose=True,
            )
        }

        self.reservoir.add_well(
            self.well_name, ms_well.MS_Type.DFM, well_geometry=geometry
        )
        self.reservoir.add_perforation(
            self.well_name,
            res_cell_idx=(1, 1, 1),
            well_seg_idx=geometry.num_segments - 1,
            well_diameter=geometry.pipe_ID,
            well_index=0.0,
            well_indexD=0.0,
        )

    def _co2_mole_fraction_for_gas_saturation(self, pressure_bar, target_sg):
        pc = self.physics.property_containers[0]
        g_idx = self.physics.phases.index("G")
        lo = self.zero
        hi = 1.0 - self.zero
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            pc.compute_saturation_full([pressure_bar, mid])
            if pc.sat[g_idx] < target_sg:
                lo = mid
            else:
                hi = mid
        return hi

    def _apply_figure_a1_initial_guess(self, initial_conditions):
        geometry = initial_conditions.pipe_geom
        depths = np.clip(geometry.TVD_segments, 0.0, self.well_length_m)
        scaled_depth = depths / self.well_length_m

        pressure_pa = 1.0e5 + (2.15e6 - 1.0e5) * scaled_depth**2
        gas_saturation = 0.985 - (0.985 - 0.45) * scaled_depth**1.8
        gas_saturation = np.clip(gas_saturation, 0.45, 0.985)

        state = initial_conditions.initial_conditions_vector
        for seg_idx, (p_pa, sg) in enumerate(zip(pressure_pa, gas_saturation)):
            p_bar = p_pa / 1.0e5
            z_co2 = self._co2_mole_fraction_for_gas_saturation(p_bar, sg)
            start = seg_idx * self.physics.n_vars
            state[start] = p_bar
            state[start + 1] = z_co2

    def set_initial_conditions(self):
        input_distribution = {
            self.physics.vars[0]: 20.0,
            self.physics.vars[1]: self.injection_composition[0],
        }
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh, input_distribution=input_distribution
        )

        well = self.reservoir.wells[0]
        volume = np.array(self.reservoir.mesh.volume, copy=False)
        volume[well.well_head_idx] = self.top_boundary_volume_m3

        for well in self.reservoir.wells:
            well.init_state = value_vector(
                self.wells[well.name].initial_conditions.initial_conditions_vector
            )

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        source = self.wells[self.well_name].source_sinks["bottom_injection"]
        component_rate = source.current_rate * source.inj_fluid_props["composition"]

        rhs_flux = np.zeros(self.reservoir.mesh.n_blocks * self.physics.n_vars)
        source_block = self.reservoir.mesh.n_res_blocks + source.segment_idx
        start = source_block * self.physics.n_vars
        rhs_flux[start : start + self.physics.n_vars] = -component_rate

        return rhs_flux
