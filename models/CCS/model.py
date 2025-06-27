import numpy as np
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData


class Model(DartsModel):
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
                                         permx=100, permy=100, permz=10, hcap=2200, rcond=100, poro=0.2, depth=depth)

        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", cell_index=(1, 1, self.reservoir.nz), well_index=100, well_indexD=100)

        self.reservoir.add_well("P1")
        for k in range(self.reservoir.nz):
            self.reservoir.add_perforation("P1", cell_index=(self.reservoir.nx, self.reservoir.ny, k+1),
                                           well_index=100, well_indexD=100)

    def set_physics(self,  zero, n_points, temperature=None, temp_inj=350.):
        """Physical properties"""
        self.zero = zero

        # Fluid components, ions and solid
        components = ["H2O", "CO2"]
        phases = ["Aq", "V"]
        nc = len(components)
        comp_data = CompData(components, setprops=True)

        pr = CubicEoS(comp_data, CubicEoS.PR)
        # aq = Jager2003(comp_data)
        aq = AQEoS(comp_data, AQEoS.Ziabakhsh2012)

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "PR"]

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        if temperature is None:  # if None, then thermal=True
            thermal = True
            state_spec = Compositional.StateSpecification.PT
        else:
            thermal = False
            state_spec = Compositional.StateSpecification.P

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               temperature=temperature, min_z=zero/10)

        property_container.flash_ev = NegativeFlash(flash_params, ["AQ", "PR"], [InitialGuess.Henry_AV])
        property_container.density_ev = dict([('V', EoSDensity(pr, comp_data.Mw)),
                                              ('Aq', Garcia2001(components))])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('Aq', Islam2012(components))])
        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas")),
                                               ('Aq', PhaseRelPerm("oil"))])

        property_container.enthalpy_ev = dict([('V', EoSEnthalpy(pr)),
                                               ('Aq', EoSEnthalpy(aq))])
        property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
                                                   ('Aq', ConstFunc(180.)), ])

        property_container.output_props = {"satA": lambda: property_container.sat[0],
                                           "satV": lambda: property_container.sat[1],
                                           "xCO2": lambda: property_container.x[0, 1],
                                           "yH2O": lambda: property_container.x[1, 0]
                                           }

        self.physics = Compositional(components, phases, self.timer, n_points, min_p=1, max_p=400, min_z=zero/10,
                                     max_z=1-zero/10, min_t=273.15, max_t=373.15, state_spec=state_spec, cache=False)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        if 1:
            dz = self.reservoir.global_data['dz'][0, 0, :]

            # zH2O = 1
            from darts.physics.super.initialize import Initialize
            # depth corresponding to boundary_idx = 10
            b_depth = self.reservoir.global_data['depth'].min() + (self.reservoir.global_data['depth'].max() - self.reservoir.global_data['depth'].min()) / 4.
            boundary_state = {'H2O': 1 - self.zero, 'pressure': 100., 'temperature': 350.}
            init = Initialize(physics=self.physics)
            X = init.solve(depth_bottom=self.reservoir.global_data['depth'].max(),
                           depth_top=self.reservoir.global_data['depth'].min(),
                           depth_known=b_depth, boundary_state=boundary_state,
                           primary_specs={'H2O': 1 - self.zero}, secondary_specs={})
            self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh, input_depth=init.depths,
                                                                 input_distribution={var: X[i::self.physics.n_vars] for i, var in
                                                                                     enumerate(self.physics.vars)})
        else:
            input_distribution = {self.physics.vars[0]: 100.,
                                  self.physics.vars[1]: 0.99995,
                                  self.physics.vars[2]: 350.,
                                  }
            return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                                  input_distribution=input_distribution)

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=self.p_inj, inj_composition=self.inj_stream[:-1],
                                               inj_temp=self.inj_stream[-1])
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=self.p_prod)
