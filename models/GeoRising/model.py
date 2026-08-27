from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params, ms_well, well_control_iface
from darts.nonlinear_solvers import NewtonSolver, ChopSpec

from darts.input.input_data import InputData

from darts.physics.base.physics import PhysicsBase
from darts.physics.iapws_physics import IAPWSPhysics
from darts.physics.base.property_container import PropertyContainer
from dartsflash.mixtures import DARTSFlash, CompData, EoS, IAPWS
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.viscosity import MaoDuan2009


class Model(DartsModel):
    def __init__(self, formulation: str = 'PT'):
        """Single-component-water geothermal model with a selectable thermal formulation.

        :param formulation: thermal state specification of the compositional physics:

            * ``'PT'`` (default) — pressure/temperature state, IAPWS-95 PT-flash.
              Unknowns are ``[pressure, temperature]``.
            * ``'PH'`` — pressure/enthalpy state, IAPWS-95 PH-flash (a
              ``PXFlash`` with enthalpy specification under the hood). Unknowns are
              ``[pressure, enthalpy]``; temperature is a derived output property.

        Both formulations use the same IAPWS-95 equation of state, so they differ only
        in the thermal state variable and are directly comparable.
        :type formulation: str
        """
        # call base class constructor
        super().__init__()

        formulation = formulation.upper()
        assert formulation in ('PT', 'PH'), \
            f"formulation must be 'PT' or 'PH', got {formulation!r}"
        self.formulation = formulation

        self.timer.node["initialization"].start()

        self.set_reservoir()

        self.set_input_data()
        self.set_physics()

        # solver configuration moved to set_solver() (called by base reset())

        self.timer.node["initialization"].stop()

    def set_solver(self):
        self.set_sim_params(first_ts=1e-4, mult_ts=8, max_ts=365, runtime=3650  )
        super().set_solver()  # platform default nonlinear + linear solvers
        self.nonlinear_solver = NewtonSolver(tolerance=1e-2, max_iterations=20,
            chop=ChopSpec(mode='global', factor=1))
        self.linear_solver.spec.tolerance = 1e-6
        self.linear_solver.spec.max_iterations = 40

    def set_reservoir(self):
        (nx, ny, nz) = (60, 60, 3)
        nb = nx * ny * nz
        perm = np.ones(nb) * 2000
        # Heterogeneous permeability shipped alongside the model (relative path so the
        # model runs anywhere, incl. CI; the absolute path from an earlier WIP commit
        # only existed on one workstation).
        perm = load_single_keyword('permXVanEssen.in', 'PERMX')
        perm = perm[:nb]

        poro = np.ones(nb) * 0.2
        dx = 30
        dy = 30
        dz = np.ones(nb) * 30

        # discretize structured reservoir
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=perm, permy=perm, permz=perm * 0.1, poro=poro, depth=2000,
                                         hcap=2200, rcond=500)
        self.reservoir.boundary_volumes['yz_minus'] = 1e8
        self.reservoir.boundary_volumes['yz_plus'] = 1e8
        self.reservoir.boundary_volumes['xz_minus'] = 1e8
        self.reservoir.boundary_volumes['xz_plus'] = 1e8

        return

    def set_wells(self):
        # add well's locations
        iw = [30, 30]
        jw = [14, 46]

        # add well
        self.reservoir.add_well("INJ")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation("INJ", res_cell_idx=(iw[0], jw[0], k + 1),
                                           well_diameter=0.32, ms_epm=True)

        # add well
        self.reservoir.add_well("PRD")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation("PRD", res_cell_idx=(iw[1], jw[1], k + 1),
                                           well_diameter=0.32, ms_epm=True)

    def set_physics(self):
        """Route to the PT- or PH-formulation physics selected in the constructor."""
        if self.formulation == 'PH':
            self.set_iapws_physics(p_step=self.idata.obl.p_step, p_origin=self.idata.obl.p_origin,
                                   second_step=self.idata.obl.e_step, second_origin=self.idata.obl.e_origin,
                                   is_ph=True)
        else:
            self.set_iapws_physics(p_step=self.idata.obl.p_step, p_origin=self.idata.obl.p_origin,
                                   second_step=self.idata.obl.t_step, second_origin=self.idata.obl.t_origin,
                                   is_ph=False)

    def set_iapws_physics(self, p_step, p_origin, second_step, second_origin, is_ph, cache=False):
        """Compositional physics on the IAPWS-95 EoS, for either thermal formulation.

        ``is_ph=False`` (PT): state spec is PT, so ``engine.X`` is ``[pressure,
        temperature]`` and the OBL grid axes are (pressure, temperature).
        ``ice_phase=False`` -> keep the temperature origin at the IAPWS liquid floor
        (273.15 K) so the unbounded grid never samples the sub-freezing (NaN) region.

        ``is_ph=True`` (PH): ``DARTSFlash.FlashType.PHFlash`` builds a
        ``PXFlash(StateSpecification.ENTHALPY)`` internally, so this is the
        pressure-enthalpy flash counterpart of the PT branch on the *same* IAPWS-95
        EoS — the two formulations differ only in the thermal state variable, which
        makes them directly comparable. State spec is PH, so ``engine.X`` is
        ``[pressure, enthalpy]`` and temperature is a derived output; the flash's
        internal PT sub-solve converts the initial temperature to enthalpy.
        """
        from dartsflash.mixtures import DARTSFlash, CompData, IAPWS
        components = ["H2O"]
        phases = ['V', 'L']  # vapor, liquid
        zero = 1e-12
        comp_data = CompData(components=components, setprops=True)

        # state_spec=PH -> OBL axes are [pressure, enthalpy]; state_spec=PT -> [pressure, temperature]
        self.physics = IAPWSPhysics(
            phases, self.timer,
            state_spec=PhysicsBase.StateSpecification.PH if is_ph else PhysicsBase.StateSpecification.PT,
            axes_step=[p_step, second_step],
            axes_origin=[p_origin, second_origin],
            cache=cache,
        )

        mixture = IAPWS(iapws_ideal=True, ice_phase=False)
        if is_ph:
            # PHFlash -> PXFlash(ENTHALPY) under the hood (dartsflash wrapper). Bound the
            # PXFlash temperature root-finding to the IAPWS liquid range: the default
            # t_min=100 K lets the solver sample far below the ice point, where IAPWS-95
            # density bisection diverges ("LIQUID MINIMUM BISECTION not converged").
            mixture.init_flash(flash_type=DARTSFlash.FlashType.PHFlash,
                                t_min=273.15, t_max=575., t_init=350.)
        else:
            mixture.init_flash(flash_type=DARTSFlash.FlashType.PTFlash)
        self.physics.set_mixture(mixture)

        """ Set property container and define properties """
        pc = PropertyContainer(phases_name=phases, components_name=components,
                               Mw=comp_data.Mw, eps_z=zero)
        self.physics.add_property_region(pc)

        pc.flash_ev = self.physics.get_flash_ev()

        pc.density_ev = {
            'V': EoSDensity(eos=mixture.eos["IAPWS"], root_flag=EoS.RootFlag.MAX),
            'L': EoSDensity(eos=mixture.eos["IAPWS"], root_flag=EoS.RootFlag.MIN),
        }
        pc.viscosity_ev = {
            'V': ConstFunc(0.01),                  # cP, steam
            'L': MaoDuan2009(components),          # cP, liquid water (pressure/temperature-dependent)
        }
        pc.enthalpy_ev = {
            'V': EoSEnthalpy(eos=mixture.eos["IAPWS"], root_flag=EoS.MAX),
            'L': EoSEnthalpy(eos=mixture.eos["IAPWS"], root_flag=EoS.MIN),
            # 'V': self.physics.get_enthalpy_ev_from_flash(phase_idx=0),
            # 'L': self.physics.get_enthalpy_ev_from_flash(phase_idx=1),
        }
        pc.rel_perm_ev = {
            'V': PhaseRelPerm("gas", swc=0.0),
            'L': PhaseRelPerm("oil", swc=0.0),
        }
        pc.conductivity_ev = {
            'V': ConstFunc(0.0),
            'L': ConstFunc(172.8),                 # kJ/m/day/K, matches geothermal default
        }
        # output_props exposes derived T (K) via the property interpolator
        pc.output_props = {'temperature': lambda: pc.temperature}

        return pc

    def set_initial_conditions(self):
        # Same physical initial state for both formulations; PhysicsBase converts the
        # temperature to enthalpy internally when state_spec is PH.
        input_distribution = {'pressure': 200.,
                              'temperature': 350.
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)

    def set_well_controls(self):
        # Both formulations label the liquid phase 'L'.
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=8000., phase_name='L',
                                               inj_composition=[], inj_temp=300.)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=False, target=8000., phase_name='L')

    def set_input_data(self):
        #init_type = 'uniform'
        init_type = 'gradient'
        self.idata = InputData(type_hydr='thermal', type_mech='none', init_type=init_type)

        self.idata.rock.compressibility = 0.  # [1/bars]
        self.idata.rock.compressibility_ref_p = 1.  # [bars]
        self.idata.rock.compressibility_ref_T = 273.15  # [K]

        # Fluid evaluator wiring lives in set_iapws_physics(); no idata.fluid needed.
        self.compositional = True

        # OBL grid resolution per formulation. The adaptive interpolator is anchored at
        # (origin) and extends on demand, so origin/step only set the index-0 node and cell size.
        if self.formulation == 'PT':
            # (pressure, temperature) axes; T origin at the IAPWS liquid floor.
            self.idata.obl.p_step = 3.142  # bar
            self.idata.obl.p_origin = 1.0
            self.idata.obl.t_step = 2.377  # K
            self.idata.obl.t_origin = 273.15
        else:
            # (pressure, enthalpy) axes for the PH (PXFlash) formulation. Enthalpy is in
            # kJ/kmol (IAPWS-95): H(273.15 K) ~ 361, H(350 K) ~ 6086; these reproduce the
            # legacy Geothermal P-H grid resolution.
            self.idata.obl.p_step = 2.756  # bar
            self.idata.obl.p_origin = 1.0
            self.idata.obl.e_step = 70.87  # kJ/kmol
            self.idata.obl.e_origin = 1000.0
