from types import SimpleNamespace

from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params, well_control_iface, ms_well
from darts.input.input_data import WellData

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic


class Model(CICDModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.zero = 1e-8
        self.set_physics()
        self.set_well_control_schedule()

        self.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=1, runtime=20, tol_newton=1e-2, tol_linear=1e-3,
                            it_newton=10, it_linear=50, newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        nx = 1000
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=1, dy=10, dz=10,
                                         permx=100, permy=100, permz=10, poro=0.3, depth=1000)
        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, 1))
        self.reservoir.add_well("P1")
        self.reservoir.add_perforation("P1", res_cell_idx=(self.reservoir.nx, 1, 1))

    def set_well_control_schedule(self):
        """
        Define scheduled well controls through idata.well_data.

        Ramp-up support is only available through the idata schedule. Direct
        self.physics.set_well_controls(...) calls apply one control immediately
        and do not create intermediate ramp targets.
        """
        inj_composition = [1.0 - 2 * self.zero * 10, self.zero * 10]

        # This example only needs the well schedule part of InputData, so a
        # lightweight namespace is enough. Larger models can use the full
        # darts.input.input_data.InputData container instead.
        self.idata = SimpleNamespace()
        self.idata.well_data = WellData()
        self.idata.well_data.add_well("I1", loc_type="ijk", loc_ijk=(1, 1, 1))
        self.idata.well_data.add_well("P1", loc_type="ijk", loc_ijk=(self.reservoir.nx, 1, 1))

        # Ramped injector rate:
        #   target rate      = 200 kmol/day
        #   ramp duration    = 10 days
        #   ramp intervals   = 100
        #
        # Because both endpoints are included, this creates 101 scheduled
        # controls at 0.1-day spacing:
        #   time: 0.0, 0.1, 0.2, ..., 10.0 days
        #   rate: 0.0, 2.0, 4.0, ..., 200.0 kmol/day
        self.idata.well_data.add_inj_rate_control(
            name="I1",
            rate=200.0,
            rate_type=well_control_iface.MOLAR_RATE,
            phase_name="gas",
            inj_composition=inj_composition,
            time=0.0,
            ramp_up_period=10.0,
            ramp_up_steps=100,
        )

        # The producer is a simple BHP-controlled well. It is put
        # in idata.well_data so all well controls are managed by the same
        # scheduled-control path.
        self.idata.well_data.add_prd_bhp_control(name="P1", bhp=50.0, time=0.0)

    def set_physics(self):
        """Physical properties"""
        epsilon = 1e-9
        # Create property containers
        components = ['CO2', 'C1', 'H2O']
        phases = ['gas', 'oil']
        Mw = [44.01, 16.04, 18.015]

        property_container = PropertyContainer(phases_name=phases, components_name=components,
                                               Mw=Mw, eps_z=epsilon, temperature=1.)

        """ properties correlations """
        property_container.flash_ev = ConstantK(len(components), [4, 2, 1e-1], self.zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('oil', DensityBasic(compr=1e-5, dens0=600))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                                ('oil', ConstFunc(0.5))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('oil', PhaseRelPerm("oil"))])

        """ Activate physics """
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=200, min_p=1, max_p=300, min_z=0., max_z=1., epsilon_z=epsilon,
                                     extrapolation_flag=True)

        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 50,
                              self.physics.vars[1]: 0.1,
                              self.physics.vars[2]: 0.2
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        # DartsModel.init() calls set_well_controls() once before the engine
        # starts. Installing controls through set_well_controls_idata() activates
        # the first scheduled control; later ramp points are applied during run().
        self.set_well_controls_idata(time=0.0)
