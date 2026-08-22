"""Shared base model for the two-phase two-component isothermal OLGA benchmarks.

The vertical and the inclined ``vs_olga`` DFM-well benchmarks are the same
physics/well/reservoir setup evaluated at a different depth, initial pressure,
inclination and first timestep, so the construction lives here once and each
model directory keeps only the constants that differ (and, for the vertical
model, the formulation variants exercised by the test suite).

Import it from a model directory with::

    import os
    import sys

    _DFM_WELL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _DFM_WELL_DIR not in sys.path:
        sys.path.insert(0, _DFM_WELL_DIR)

    from _olga_benchmark_base import OLGABenchmarkModel  # noqa: E402

The lookup is anchored on ``__file__`` rather than on the working directory
because ``model.py`` is imported as a top-level module both by the test suite
(which puts only the model directory on ``sys.path``) and by ``main.py``.
"""

import numpy as np
from darts.engines import ms_well, value_vector, well_control_iface
from dartsflash.components import CompData
from dartsflash.libflash import EoS, NegativeFlash
from dartsflash.mixtures import DARTSFlash, Mixture

from darts.models.cicd_model import CICDModel
from darts.models.conditions import PipeSourceTerm
from darts.nonlinear_solvers import ChopSpec, NewtonSolver
from darts.physics.base.physics import PhysicsBase
from darts.physics.base.property_container import PropertyContainer
from darts.physics.eos_physics import EoSPhysics
from darts.physics.properties.basic import PhaseRelPerm
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.eos_properties import EoSDensity
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.pipes.define_pipe_geometry import PipeGeometry
from darts.pipes.interfacial_tension import IFT_multicomponent_MCM
from darts.pipes.linear_dfm_well_ipr import (
    LinearDFMWellIPRConnection,
    LinearDFMWellIPRHook,
    PI_Type,
)
from darts.pipes.pipe import Pipe
from darts.pipes.set_initial_conditions import SingleAmbientTemperature
from darts.pipes.upstream_ramp_up_rate import UpstreamRampUpRate
from darts.reservoirs.struct_radial_reservoir import StructRadialReservoir


class OLGABenchmarkModel(CICDModel):
    """Injection of pure gaseous CO2 at a constant mass rate into a well containing water.

    Concrete models set the four constants below; everything else is shared.
    """

    #: first timestep, in seconds
    first_ts_seconds: float = None
    #: depth of the centroid of the top reservoir cell, in m
    reservoir_depth: float = None
    #: initial reservoir pressure in bar, from the pressure of the perforated
    #: segment of the wellbore
    p_init_res: float = None
    #: well inclination, in degrees relative to the vertical direction
    inclination_angle: float = None
    #: test-suite variants supported by the concrete model, in addition to ``None``
    supported_formulations: tuple = ()

    def __init__(self, formulation=None):
        """
        :param formulation: optional test-suite variant of the base model, one of
            :attr:`supported_formulations`, or ``None`` for the base model.
        :type formulation: str or None
        """
        # Call base class constructor
        super().__init__()

        assert formulation is None or formulation in self.supported_formulations, (
            f"unknown formulation {formulation!r}"
        )
        self.formulation = formulation

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.reservoir.grav_acceleration_for_spe = 9.80665
        self.zero = 1e-10
        self.set_physics()

        self.nonlinear_solver = NewtonSolver(
            tolerance=1e-3,
            max_iterations=10,
            chop=ChopSpec(mode='local'),
            coupled_well_res_norm_method=2,
        )
        self.set_sim_params(
            first_ts=self.first_ts_seconds / (24 * 60 * 60),
            mult_ts=2,
            max_ts=2 / (24 * 60 * 60),
            tol_linear=1e-4,
            it_linear=10,
            # This runtime will be used when CI test is conducted without the main file
            runtime=5 / 60 / 24,
        )

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        (nr, nz) = (2, 1)
        (dr, dz) = (5, 50)

        poro = np.ones((nr, nz)) * 0.2
        permr = np.ones((nr, nz)) * 200

        permz = permr

        self.well_1_ID = 0.1
        self.reservoir = StructRadialReservoir(
            self.timer,
            nr=nr,
            nz=nz,
            dr=dr,
            dz=dz,
            poro=poro.flatten(order='F'),
            permr=permr.flatten(order='F'),
            permz=permz.flatten(order='F'),
            R0=self.well_1_ID / 2,
            R1=10,
            logspace=True,
            depth=self.reservoir_depth,
        )
        self.reservoir.boundary_volumes['yz_minus'] = 1e20

        return

    def set_initial_conditions(self):
        # the composition of the perforated segment of the wellbore
        z_CO2 = self.zero

        input_distribution = {
            self.physics.vars[0]: self.p_init_res,
            self.physics.vars[1]: z_CO2,
        }
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh, input_distribution=input_distribution
        )

        for well in self.reservoir.wells:
            well.init_state = value_vector(
                self.wells[well.name].initial_conditions.initial_conditions_vector
            )

        return

    def set_physics(self):
        components_names = ['CO2', 'H2O']
        phases_names = ['G', 'L']  # G is the CO2-rich phase and L is the aqueous phase
        comp_data = CompData(components_names, setprops=True)
        epsilon = self.zero / 10

        """ Define state specification and initialize physics object """
        state_spec = PhysicsBase.StateSpecification.P
        self.physics = EoSPhysics(
            components_names,
            phases_names,
            self.timer,
            state_spec=state_spec,
            axes_step=[0.05, 1e-4],
            axes_origin=[1.0, 0.0],
            epsilon_z=epsilon,
        )

        """ Define flash """
        mixture = Mixture(comp_data)

        mixture.set_vl_eos(
            vl_eos_name="PR", hybrid_aq_eos_name="Aq", root_order=[EoS.STABLE]
        )
        mixture.set_aq_eos(aq_eos_name="Aq")

        mixture.init_flash(
            flash_type=DARTSFlash.FlashType.NegativeFlash,
            eos_order=["PR", "Aq"],
            nf_initial_guess=[NegativeFlash.Ki.Henry_VA],
        )
        self.physics.set_mixture(mixture)

        """ Add property region: PropertyContainer object and correlations """
        system_temperature = 40 + 273.15

        property_container = PropertyContainer(
            phases_names,
            components_names,
            Mw=comp_data.Mw,
            eps_z=epsilon,
            temperature=system_temperature,
            rock_comp=0,
        )
        self.physics.add_property_region(property_container)

        property_container.flash_ev = self.physics.get_flash_ev()

        property_container.density_ev = dict(
            [
                ('G', EoSDensity(eos=mixture.eos["PR"])),
                ('L', Garcia2001(components_names)),
            ]
        )

        property_container.viscosity_ev = dict(
            [
                ('G', Fenghour1998()),
                ('L', Islam2012(components_names)),
            ]
        )

        property_container.rel_perm_ev = dict(
            [
                ('G', PhaseRelPerm("gas", swc=0, sgr=0, n=1)),
                ('L', PhaseRelPerm("oil", swc=0, sgr=0, n=1)),
            ]
        )

        property_container.IFT_ev = IFT_multicomponent_MCM(components_names)

        property_container.output_props = {}
        property_container.output_props['temperature'] = lambda: (
            property_container.temperature
        )
        for j, ph in enumerate(phases_names):
            property_container.output_props['s' + ph] = lambda jj=j: (
                property_container.sat[jj]
            )
            property_container.output_props['rho' + ph] = lambda jj=j: (
                property_container.dens[jj]
            )
            property_container.output_props['mu' + ph] = lambda jj=j: (
                property_container.mu[jj]
            )
            for i, comp in enumerate(components_names):
                property_container.output_props[f'x{comp}_in_{ph}_mass'] = (
                    lambda jj=j, ii=i: property_container.x_mass[jj, ii]
                )

        return

    def set_wells(self):
        """================================================= Well 1 ================================================="""
        well_1_name = "I1"
        well_1_ms_type = ms_well.MS_Type.DFM
        # Lengths of the well segments are specified here.
        # The lengths of the well segments in front of the reservoir must be equal to the height of the reservoir cells.
        well_1_segments_lengths = 50 * np.ones(20)  # From top to bottom of the wellbore
        well_1_ID = self.well_1_ID
        well_1_wall_roughness = 2.5e-5
        verbose = True
        well_1_geometry = PipeGeometry(
            well_1_name,
            well_1_segments_lengths,
            well_1_ID,
            self.inclination_angle,
            well_1_wall_roughness,
            verbose,
        )

        # %% Set initial conditions in the pipe using SingleAmbientTemperature
        ambient_temperature = self.physics.property_containers[0].temperature
        pipe_head_pressure = 10  # bar
        pipe_head_segment_index = 0  # index starts from zero

        # 0 is the beginning of the pipe and pipe_intervals are TVD
        initial_conditions_dict = {
            'phases_names': ['L'],
            'phases_compositions': [[1e-10, 1 - 1e-10]],
            'pipe_intervals': [[0, well_1_geometry.pipe_length]],
        }

        well_1_initial_conditions = SingleAmbientTemperature(
            well_1_name,
            well_1_geometry,
            self.physics,
            ambient_temperature,
            pipe_head_pressure,
            pipe_head_segment_index,
            initial_conditions_dict,
            verbose,
        )

        # %% Add source/sink terms
        inj_segment_idx = 0
        target_inj_rate = 58895.98 / 15  # in kmol/day
        ramp_up_period = 0.0

        # Use zero water in the injected fluid in order to avoid sustained accumulation of water in the bottom-hole
        inj_phase_comp = np.array([1 - self.zero, self.zero])
        inj_phase_name = "G"

        ramp_up_rate = UpstreamRampUpRate(
            well_1_name,
            well_1_geometry,
            self.physics,
            self.data_ts.dt_first,
            inj_segment_idx,
            target_inj_rate,
            ramp_up_period,
            composition=inj_phase_comp,
            pressure=pipe_head_pressure,
            temperature=ambient_temperature,
            phase_name=inj_phase_name,
            verbose=verbose,
        )
        # The following dict is used by the PipeSourceTerm condition and the pipe velocity evaluation
        # (the BHP-controlled producer variant has no wellhead injection source)
        source_sinks = (
            None
            if self.formulation == 'ipr_producer'
            else {"RampUpRate1": ramp_up_rate}
        )

        # Drift-flux closure variants change only the closure passed to the pipe
        drift_flux_model = (
            self.formulation
            if self.formulation in ('tang_2019', 'bhagwat_ghajar_2014', 'bai_2023')
            else 'shi_t2well'
        )

        # %% Store well props
        self.wells = {
            'I1': Pipe(
                'I1',
                well_1_geometry,
                self.physics,
                self.reservoir,
                well_1_initial_conditions,
                source_sinks=source_sinks,
                drift_flux_model=drift_flux_model,
                verbose=verbose,
            )
        }

        if source_sinks is not None:
            # The wellhead injection source is applied to the residual by the unified conditions layer
            # (the BHP-controlled producer variant has no such source)
            self.conditions.add(
                PipeSourceTerm(well_name='I1', source_sink_name='RampUpRate1')
            )

        self.reservoir.add_well(
            well_1_name, well_1_ms_type, well_geometry=well_1_geometry
        )

        # Well with a single perforation
        well_1_perforated_segment = well_1_geometry.num_segments

        self.reservoir.add_perforation(
            well_1_name,
            res_cell_idx=(1, 1, 1),
            well_seg_idx=well_1_perforated_segment,
            well_diameter=well_1_geometry.pipe_ID,
            # well_index=65.54393 would model the variable injectivity equivalent to 1e5 kg/day/bar
            well_index=0.0,
            well_indexD=0.0,
        )

        self.conditions.add(
            LinearDFMWellIPRHook(self, [self.get_ipr_connection(well_1_name)])
        )

    def get_ipr_connection(self, well_name):
        """Return the IPR connection of the well perforation for the active formulation."""
        perforation_index = len(self.reservoir.get_well(well_name).perforations) - 1
        if self.formulation == 'ipr_volumetric':
            # Volumetric PI (m3/day/bar at upstream in-situ conditions) of a magnitude
            # equivalent to the base mass PI, with nonzero intercept and pressure offset
            return LinearDFMWellIPRConnection(
                well_name=well_name,
                perforation_index=perforation_index,
                pi=100.0,
                pi_type=PI_Type.VOLUMETRIC,
                ipr_pressure_offset=0.05,
                ipr_intercept=10.0,
            )
        if self.formulation == 'ipr_producer':
            # Molar PI for the BHP-controlled producer variant (reservoir-upstream branch)
            return LinearDFMWellIPRConnection(
                well_name=well_name,
                perforation_index=perforation_index,
                pi=5e3,
                pi_type=PI_Type.MOLAR,
                ipr_pressure_offset=0.0,
            )
        return LinearDFMWellIPRConnection(
            well_name=well_name,
            perforation_index=perforation_index,
            pi=1e5,
            pi_type=PI_Type.MASS,
            ipr_pressure_offset=0.0,
        )

    def set_well_controls(self):
        if self.formulation == 'ipr_producer':
            # Produce by holding the wellhead pressure below its 10 bar initial value, which
            # lowers the whole well column below the reservoir pressure at the perforation
            w = self.reservoir.wells[0]
            self.physics.set_well_controls(
                wctrl=w.control,
                control_type=well_control_iface.BHP,
                is_inj=False,
                target=9.5,
            )
