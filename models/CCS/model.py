import numpy as np
from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import value_vector, sim_params

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.super.operator_evaluator import PropertyOperators

from darts.physics.properties.basic import PhaseRelPerm, CapillaryPressure, RockEnergyEvaluator, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012

from dartsflash.libflash import NegativeFlash2
from dartsflash.libflash import CubicEoS, Ziabakhsh2012, FlashParams, SplitIG
from dartsflash.components import CompData, EnthalpyIdeal
from dartsflash.eos_properties import EoSDensity, EoSEnthalpy


class Model(DartsModel):
    def __init__(self, n_points=1000, temp_init=350, temp_inj=350):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics(n_points, temp_ini=temp_init)
        self.set_wells(p_inj=60, t_inj=temp_inj, p_prod=30)

        self.set_sim_params(first_ts=1e-3, mult_ts=1.5, max_ts=5,
                            tol_newton=1e-3, tol_linear=1e-5, it_newton=10, it_linear=50,
                            )

        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        nx = 100
        ny = 1
        nz = 40
        nb = nx * ny * nz
        dz = 2
        depth = np.zeros(nb)
        n_layer = nx*ny
        for k in range(nz):
            depth[k*n_layer:(k+1)*n_layer] = 1000 + k * dz

        self.x_axes = np.logspace(-0.3, 2, nx)
        dx = np.tile(self.x_axes, nz)

        self.reservoir = StructReservoir(self.timer, nx, ny, nz, dx=dx, dy=10, dz=dz,
                                         permx=100, permy=100, permz=10, poro=0.2, depth=depth)

        # get wrapper around local array (length equal to active blocks number)
        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)

        hcap.fill(2200)  # 2200 kJ/m3.K for sandstone
        rcond.fill(100)

    def set_wells(self, p_inj=60, t_inj=None, p_prod=30):
        """Create well objects"""
        self.reservoir.add_well('I1')
        for n in range(self.reservoir.nz-1, self.reservoir.nz):
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=n+1,
                                           well_index=100, multi_segment=False, verbose=False)
        self.reservoir.add_well('P1')
        for n in range(self.reservoir.nz):
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.reservoir.nx, j=self.reservoir.ny, k=n+1,
                                           well_index=100, multi_segment=False, verbose=False)

        """Set operational conditions"""
        self.p_inj = p_inj
        self.p_prod = p_prod

        self.inj_stream = [0.99995]
        if self.physics.thermal:
            self.inj_stream.append(t_inj)

    def set_physics(self, n_points, temperature=None, temp_ini=350.):
        self.zero = 1e-7

        """Physical properties"""
        # Fluid components, ions and solid
        components = ["CO2", "H2O"]
        phases = ["Aq", "V"]
        nc = len(components)
        comp_data = CompData(components, setprops=True)

        pr = CubicEoS(comp_data, CubicEoS.PR)
        # aq = Jager2003(comp_data)
        aq = Ziabakhsh2012(comp_data)

        flash_params = FlashParams(nc)

        # EoS-related parameters
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_used = ["AQ", "PR"]

        flash_params.split_initial_guess = SplitIG(comp_data, SplitIG.HENRY, 1)

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        self.ini_stream = [0.00005]
        self.p_init = 60.

        self.temperature = temperature  # if None, then thermal=True
        if self.temperature is None:
            thermal = True
            self.init_temp = temp_ini
        else:
            thermal = False
            self.init_temp = temperature

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               temperature=self.temperature, min_z=self.zero/10)

        property_container.flash_ev = NegativeFlash2(flash_params)
        property_container.density_ev = dict([('V', EoSDensity(pr, comp_data.Mw)),
                                              ('Aq', Garcia2001(components))])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('Aq', Islam2012(components))])
        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas")),
                                               ('Aq', PhaseRelPerm("oil"))])

        h_ideal = EnthalpyIdeal(components)
        property_container.enthalpy_ev = dict([('V', EoSEnthalpy(pr, h_ideal)),
                                               ('Aq', EoSEnthalpy(aq, h_ideal))])
        property_container.conductivity_ev = dict([('V', ConstFunc(0.)),
                                                   ('Aq', ConstFunc(0.)), ])

        self.physics = Compositional(components, phases, self.timer, n_points, min_p=1, max_p=400, min_z=self.zero/10,
                                     max_z=1-self.zero/10, min_t=273.15, max_t=373.15, thermal=thermal, cache=0)
        self.physics.add_property_region(property_container)
        props = [('satA', 'sat', 0), ('satV', 'sat', 1), ('xCO2', 'x', (0, 0)), ('yH2O', 'x', (1, 1))]
        self.physics.init_physics(output_props=PropertyOperators(props, property_container))

    def set_initial_conditions(self):
        if self.physics.thermal:
            self.physics.set_uniform_initial_conditions(self.reservoir.mesh, self.p_init, self.ini_stream, self.init_temp)
        else:
            self.physics.set_uniform_initial_conditions(self.reservoir.mesh, self.p_init, self.ini_stream)

        self.timer.node["initialization"].stop()

    def set_boundary_conditions(self):
        # define all wells as closed
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                w.control = self.physics.new_bhp_inj(self.p_inj, self.inj_stream)
            else:
                w.control = self.physics.new_bhp_prod(self.p_prod)
