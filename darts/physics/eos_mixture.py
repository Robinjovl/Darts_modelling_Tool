import warnings

from dartsflash.components import CompData
from dartsflash.mixtures import Mixture

from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy, EoSFugacity
from darts.physics.super.physics import (
    Compositional,
    HistoryField,
    Iterable,
    timer_node,
)


class EoSMixture(Compositional, Mixture):
    """
    Implementation of EoS-based Physics
    - Multiple inheritance of Compositional and DARTS-flash Mixture classes

    Mixture-specific (see dartsflash.mixtures.Mixture class for description)
    - set_*_eos() methods wrap EoS definition
        - set_vl_eos(): Set V/L phases EoS object (e.g., PR, SRK, CPA, ...)
        - set_aq_eos(): Set aqueous phase EoS object
        - set_ice_eos(): Set ice phase EoS object
        - set_salt_eos(): Set salt phase EoS object (NaCl, CaCl2, KCl)
        - set_h_eos(): Set hydrate phase EoS object (sI, sII, sH)
    - init_*flash() method calls Mixture.init_*flash() method to initialize Flash object (ptflash, pxflash, negativeflash)
    """

    def __init__(
        self,
        phases: list,
        timer: timer_node,
        axes_step: list[float],
        axes_origin: list[float] = None,
        comp_data: CompData = None,
        components: list = None,
        salt_components: list = None,
        epsilon_z: float = 1e-9,
        sim_eps_multiplier: float = 10,
        extrapolation_flag: bool = True,
        state_spec: Compositional.StateSpecification = Compositional.StateSpecification.P,
        cache: bool = False,
        history_fields: Iterable[HistoryField] | None = None,
        mixture_name: str = None,
    ):
        """
        Constructor initializes both Compositional and Mixture parts

        :param phases:

        """
        # If no CompData object has been provided, create instance from list of components
        if comp_data is None:
            assert components is not None, (
                "Neither CompData object nor components list are provided"
            )
            comp_data = CompData(
                components=components, salt_components=salt_components, setprops=True
            )
        elif components is not None:
            warnings.warn(
                "CompData AND components provided, continuing simulation with CompData",
                stacklevel=2,
            )

        # Call Compositional constructor
        Compositional.__init__(
            self,
            components=comp_data.components,
            phases=phases,
            timer=timer,
            axes_step=axes_step,
            axes_origin=axes_origin,
            epsilon_z=epsilon_z,
            sim_eps_multiplier=sim_eps_multiplier,
            extrapolation_flag=extrapolation_flag,
            state_spec=state_spec,
            cache=cache,
            history_fields=history_fields,
        )

        # Call Mixture constructor
        Mixture.__init__(
            self,
            comp_data=comp_data,
            mixture_name=mixture_name,
        )

    def init_physics(
        self,
        discr_type: str = 'tpfa',
        platform: str = 'cpu',
        itor_type: str = 'multilinear',
        itor_mode: str = 'adaptive',
        itor_precision: str = 'd',
        verbose: bool = False,
        is_barycentric: bool = False,
        n_solid: int | None = None,
        parallel_evaluation: bool = False,
        n_workers: int | None = None,
        evaluator_factory_hook=None,
        verbose_evaluators: bool = False,
    ):
        """
        Initialize physics and check consistency of flash definition: do the phase types correspond to EoS objects?
        """
        # Call Compositional.init_physics() logic
        Compositional.init_physics(
            self,
            discr_type=discr_type,
            platform=platform,
            itor_type=itor_type,
            itor_mode=itor_mode,
            itor_precision=itor_precision,
            verbose=verbose,
            is_barycentric=is_barycentric,
            n_solid=n_solid,
            parallel_evaluation=parallel_evaluation,
            n_workers=n_workers,
            evaluator_factory_hook=evaluator_factory_hook,
            verbose_evaluators=verbose_evaluators,
        )

        # Check that phases argument is consistent with flash definition
        assert len(self.nph) == self.flash_params.np_max, (
            "More phases are specified in self.phases than in FlashParams"
        )

        # TODO: check that phase labels correspond to phase types specified in flash setup
        # for phase in self.phases:

    def get_flash_ev(self, region: int = None):
        """
        This class serves as flash_ev object for PropertyContainer objects: return self

        - TODO: In a future version, we may have different flash definitions among regions
            -> return flash instance associated to specific region

        :param region: Key of property region in PropertyContainers
        """
        return self

    def get_density_from_flash(
        self,
        phase_idx: int,
    ):
        """
        Get EoSDensity object that evaluates phase mass density for specified phase from FlashResults

        :param phase_idx: Phase index in flash output
        """
        return EoSDensity(flash_ev=self, phase_idx=phase_idx)

    def get_enthalpy_from_flash(self, phase_idx: int):
        """
        Get EoSEnthalpy object that evaluates phase enthalpy for specified phase from FlashResults

        :param phase_idx: Phase index in flash output
        """
        return EoSEnthalpy(flash_ev=self, phase_idx=phase_idx)

    def get_fugacity_from_flash(self, phase_idx: int):
        """
        Get EoSFugacity object that evaluates component fugacities for specified phase from FlashResults

        :param phase_idx: Phase index in flash output
        """
        return EoSFugacity(flash_ev=self, phase_idx=phase_idx)
