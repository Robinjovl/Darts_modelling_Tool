import numpy as np
from darts.engines import ms_well, sim_params, well_control_iface

from darts.models.cicd_model import CICDModel
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.enthalpy import EnthalpyBasic
from darts.physics.properties.flash import ConstantK
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.reservoirs.adaptive_lgr import (
    AdaptiveLGRConfig,
    count_lgr_parent_cells,
    initial_lgrs_from_config,
    plan_adaptive_lgr,
    project_reservoir_state,
    reservoir_state_from_engine,
)
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.reservoirs.struct_reservoir_with_lgr import StructReservoirWithLGR


class Model(CICDModel):
    def __init__(self, use_amr: bool = True):
        super().__init__()

        self.timer.node["initialization"].start()

        self.zero = 1e-8
        self.use_amr = use_amr
        self.parent_shape = (10, 1, 1)
        self.perforation_parent_cells = {
            "I1": (2, 1, 1),
            "P1": (9, 1, 1),
        }
        self.amr_config = AdaptiveLGRConfig(
            refine=(7, 7, 1),
            buffer_cells=1,
            gradient_threshold=0.25,
            indicator_variables=("CO2", "C1", "temperature"),
            seed_parent_cells=tuple(self.perforation_parent_cells.values()),
            patch_name_prefix="amr",
        )
        self.lgrs = initial_lgrs_from_config(self.parent_shape, self.amr_config)
        self.amr_history = []

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

    def _make_parent_reservoir(self):
        return StructReservoir(
            self.timer,
            nx=self.parent_shape[0],
            ny=self.parent_shape[1],
            nz=self.parent_shape[2],
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

    def _make_lgr_reservoir(self, lgrs):
        parent = self._make_parent_reservoir()
        return StructReservoirWithLGR(
            self.timer,
            parent,
            lgrs,
            lgr_coarse_fine_tran_mode="normal",
        )

    def set_reservoir(self):
        self.reservoir = StructReservoirWithLGR(
            self.timer,
            self._make_parent_reservoir(),
            self.lgrs,
            lgr_coarse_fine_tran_mode="normal",
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
        inj_cell = self.reservoir.get_cell_index_in_parent_cell(
            self.perforation_parent_cells["I1"]
        )
        self.reservoir.add_perforation(
            "I1",
            res_cell_idx=inj_cell,
            well_diameter=0.1524,
            well_indexD=None,
        )

        self.reservoir.add_well("P1")
        prod_cell = self.reservoir.get_cell_index_in_parent_cell(
            self.perforation_parent_cells["P1"]
        )
        self.reservoir.add_perforation(
            "P1",
            res_cell_idx=prod_cell,
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

    def adapt_lgr(self, verbose: bool = False) -> bool:
        if not self.use_amr:
            return False

        state = reservoir_state_from_engine(self)
        plan = plan_adaptive_lgr(
            self.reservoir,
            state,
            self.physics.vars,
            self.amr_config,
        )
        if not plan.changed:
            return False

        old_reservoir = self.reservoir
        old_time = float(self.physics.engine.t)
        old_vtk_files = dict(getattr(old_reservoir, "vtk_filenames_and_times", {}))

        new_reservoir = self._make_lgr_reservoir(plan.lgrs)
        new_reservoir.init_reservoir(verbose=False)
        new_reservoir.vtk_filenames_and_times = old_vtk_files
        projected_state = project_reservoir_state(
            old_reservoir,
            state,
            new_reservoir,
        )

        self.reservoir = new_reservoir
        self.lgrs = plan.lgrs
        self.set_wells()
        self.has_dfm_well = any(
            well.ms_type == ms_well.MS_Type.DFM for well in self.reservoir.wells
        )
        if not self.has_dfm_well:
            self.wells = None

        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)
        self.set_op_list()
        self.set_boundary_conditions()
        self.set_well_controls()
        self._set_projected_initial_state(projected_state)
        self.reset()
        self._set_engine_reservoir_state(projected_state)
        self.physics.engine.t = old_time
        self._refresh_output_after_amr()

        self.amr_history.append(
            {
                "time": old_time,
                "n_lgrs": len(self.lgrs),
                "n_selected_parent_cells": len(plan.selected_parent_cells),
                "n_refined_parent_cells": count_lgr_parent_cells(self.lgrs),
                "n_res_blocks": self.reservoir.mesh.n_res_blocks,
            }
        )
        if verbose:
            print(
                "AMR updated LGR layout at "
                f"t={old_time:g} days: {len(self.lgrs)} patches, "
                f"{self.reservoir.mesh.n_res_blocks} reservoir blocks."
            )
        return True

    def _set_projected_initial_state(self, projected_state: np.ndarray) -> None:
        input_distribution = {
            var_name: projected_state[:, idx]
            for idx, var_name in enumerate(self.physics.vars)
        }
        self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh,
            input_distribution=input_distribution,
        )

    def _set_engine_reservoir_state(self, projected_state: np.ndarray) -> None:
        flat_state = np.asarray(projected_state, dtype=float).reshape(-1)
        for name in ("X", "Xn"):
            values = getattr(self.physics.engine, name, None)
            if values is not None:
                np.asarray(values)[: flat_state.size] = flat_state

    def _refresh_output_after_amr(self) -> None:
        if not hasattr(self, "output"):
            return
        self.output.reservoir = self.reservoir
        self.output.op_list = self.op_list
        self.output.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        self.output.wells = self.wells
        self.output.has_dfm_well = self.has_dfm_well
