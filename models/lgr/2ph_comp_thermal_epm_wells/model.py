from darts.engines import sim_params, well_control_iface

from darts.models.cicd_model import CICDModel
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic
from darts.physics.properties.flash import ConstantK
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.struct_reservoir_with_lgr import LGRPatch, StructReservoirWithLGR


class Model(CICDModel):
    def __init__(self):
        super().__init__()

        self.timer.node["initialization"].start()

        self.zero = 1e-8
        self.set_reservoir()
        self.set_physics()
        self.set_sim_params(
            first_ts=1e-4,
            mult_ts=2,
            max_ts=0.5,
            runtime=50,
            tol_newton=1e-3,
            tol_linear=1e-4,
            it_newton=12,
            it_linear=60,
            newton_type=sim_params.newton_local_chop,
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
            LGRPatch("inj_lgr", (2, 2), (1, 1), (1, 1), (7, 7, 1)),
            LGRPatch("prod_lgr", (9, 9), (1, 1), (1, 1), (7, 7, 1)),
        ]
        self.reservoir = StructReservoirWithLGR(
            self.timer,
            parent,
            lgrs,
            lgr_coarse_fine_transmissibility_mode="flow_based",
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
        # so heat capacities are specified in kJ/kmol/K.
        property_container.enthalpy_ev = {
            "G": EnthalpyBasic(hcap=37.0),
            "L": EnthalpyBasic(hcap=75.3),
        }
        property_container.conductivity_ev = {
            "G": ConstFunc(3.5),
            "L": ConstFunc(75.0),
        }
        property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)

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
            max_t=273.15 + 200.0,
            extrapolation_flag=True,
        )
        self.physics.add_property_region(property_container)

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(
            "I1",
            lgr_name="inj_lgr",
            lgr_cell_idx=(4, 4, 1),
            well_diameter=0.1524,
            well_indexD=None,
        )

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation(
            "P1",
            lgr_name="prod_lgr",
            lgr_cell_idx=(4, 4, 1),
            well_diameter=0.1524,
            well_indexD=None,
        )

    def set_initial_conditions(self):
        input_distribution = {
            self.physics.vars[0]: 100.0,
            self.physics.vars[1]: 0.1,
            self.physics.vars[2]: 0.2,
            self.physics.vars[3]: 350.0,
        }
        return self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh,
            input_distribution=input_distribution,
        )

    def set_well_controls(self):
        injection_composition = [1.0 - 2.0 * self.zero, self.zero]
        for idx, well in enumerate(self.reservoir.wells):
            if idx == 0:
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=True,
                    target=140.0,
                    inj_composition=injection_composition,
                    inj_temp=320.0,
                )
            else:
                self.physics.set_well_controls(
                    wctrl=well.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=80.0,
                )
