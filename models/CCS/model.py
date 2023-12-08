import numpy as np
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.super.operator_evaluator import PropertyOperators

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012

from dartsflash.libflash import NegativeFlash2
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData, EnthalpyIdeal
from dartsflash.eos_properties import EoSDensity, EoSEnthalpy


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

        reservoir = StructReservoir(self.timer, nx, ny, nz, dx=dx, dy=10, dz=dz,
                                    permx=100, permy=100, permz=10, hcap=2200, rcond=100, poro=0.2, depth=depth)

        return super().set_reservoir(reservoir)

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation("I1", cell_index=(1, 1, self.reservoir.nz), well_index=100, well_indexD=100)

        self.reservoir.add_well("P1")
        for k in range(self.reservoir.nz):
            self.reservoir.add_perforation("P1", cell_index=(self.reservoir.nx, self.reservoir.ny, k+1),
                                           well_index=100, well_indexD=100)

        return super().set_wells()

    def set_physics(self,  zero, n_points, temperature=None, temp_inj=350.):
        """Physical properties"""
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
        flash_params.eos_used = ["AQ", "PR"]

        flash_params.split_initial_guesses = [InitialGuess.Henry_AV]

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        if temperature is None:  # if None, then thermal=True
            thermal = True
        else:
            thermal = False

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               temperature=temperature, min_z=zero/10)

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

        physics = Compositional(components, phases, self.timer, n_points, min_p=1, max_p=400, min_z=zero/10,
                                max_z=1-zero/10, min_t=273.15, max_t=373.15, thermal=thermal, cache=False)
        physics.add_property_region(property_container)
        props = [('satA', 'sat', 0), ('satV', 'sat', 1), ('xCO2', 'x', (0, 1)), ('yH2O', 'x', (1, 0))]
        physics.add_property_operators(PropertyOperators(props, property_container))

        return super().set_physics(physics)

    def set_well_controls(self):
        # define all wells as closed
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                w.control = self.physics.new_bhp_inj(self.p_inj, self.inj_stream)
            else:
                w.control = self.physics.new_bhp_prod(self.p_prod)
