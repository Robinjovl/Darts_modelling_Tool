from dartsflash.mixtures import DARTSFlash, Mixture

from darts.physics.base.physics import (
    HistoryField,
    Iterable,
    PhysicsBase,
    timer_node,
)
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy, EoSFugacity

# Which DARTSFlash.FlashType(s) a given PhysicsBase.StateSpecification may be paired with.
# P/PT are both PT-based (isothermal P uses a fixed T at evaluate() time)
# PH/PS each require the matching PXFlash flash type
_EXPECTED_FLASH_TYPES = {
    PhysicsBase.StateSpecification.P: (
        DARTSFlash.FlashType.PTFlash,
        DARTSFlash.FlashType.NegativeFlash,
    ),
    PhysicsBase.StateSpecification.PT: (
        DARTSFlash.FlashType.PTFlash,
        DARTSFlash.FlashType.NegativeFlash,
    ),
    PhysicsBase.StateSpecification.PH: (DARTSFlash.FlashType.PHFlash,),
    PhysicsBase.StateSpecification.PS: (DARTSFlash.FlashType.PSFlash,),
}


class EoSPhysics(PhysicsBase):
    """
    Implementation of EoS-based Physics
    - Inherited from PhysicsBase only; constructor signature matches PhysicsBase exactly
    - Composes a DARTS-flash Mixture instance, attached separately via :meth:`set_mixture`
    - Checks physics consistency with EoS and flash definition (components, phases, state specification)

    Usage:
        physics = EoSPhysics(components, phases, timer, axes_step, ...)  # same args as PhysicsBase
        mixture = Mixture(comp_data=comp_data)
        mixture.set_vl_eos(...); mixture.set_aq_eos(...); ...
        mixture.init_*flash(eos_order=[...])                            # init_ptflash/init_pxflash/init_negativeflash
        physics.set_mixture(mixture)                                    # checks components immediately
        physics.init_physics(...)                                       # checks phases/correlations
    """

    def __init__(
        self,
        components: list,
        phases: list,
        timer: timer_node,
        axes_step: list[float],
        axes_origin: list[float] = None,
        epsilon_z: float = 1e-9,
        sim_eps_multiplier: float = 10,
        extrapolation_flag: bool = True,
        state_spec: PhysicsBase.StateSpecification = PhysicsBase.StateSpecification.P,
        cache: bool = False,
        history_fields: Iterable[HistoryField] | None = None,
    ):
        """
        Constructor initializes PhysicsBase only.
        Attach a Mixture separately via :meth:`set_mixture` once it has been built
        (components must match ``components`` here).

        :param phases: List of phase labels, expected in flash output order (see
            ``self.flash_evs[region].flash_params.eos_order`` and per-EoS ``root_order``)
        """
        PhysicsBase.__init__(
            self,
            components=components,
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

        self.flash_evs: dict[str | int | None, Mixture] = {}

    def set_mixture(
        self, mixture: Mixture, phases_to_eos: dict[tuple] = None, region: int = None
    ) -> None:
        """
        Attach a DARTS-flash Mixture instance to this physics object as ``self.flash_evs[region]``.

        Checks that:
        - Mixture components + salts match the Physics.components
        - TODO: Phase labels in Physics object are consistent with phase types specified in Mixture
        - Mixture.FlashType is compatible with Physics.StateSpecification

        :param mixture: Configured Mixture instance
        :param phases_to_eos: Dictionary of phase labels to phase types, default is None which throws warning
        :param region: Key of property region in PropertyContainers, defaults to 0
        """
        assert isinstance(mixture, Mixture), (
            "Provide an object of type dartsflash.Mixture"
        )

        # Assert that the physics' component list matches the composed Mixture's ``comp_data``.
        assert list(self.components) == list(mixture.comp_data.species_with_salts), (
            f"Physics components {self.components} do not match mixture.comp_data.species_with_salts "
            f"{mixture.comp_data.species_with_salts}"
        )

        # Assert that the physics' StateSpecification is compatible with the mixture's FlashType
        # P/PT expect a PT-based flash (PTFlash/NegativeFlash with ``init_ptflash()``/``init_negativeflash()``)
        # PH expects a PH-flash (PHFlash with ``init_pxflash(flash_type=DARTSFlash.FlashType.PHFlash)``)
        # PS expects a PS-flash (PSFlash with ``init_pxflash(flash_type=DARTSFlash.FlashType.PSFlash)``)
        # Mismatches here mean the flash is being asked for state variables the physics never provides
        expected = _EXPECTED_FLASH_TYPES.get(self.state_spec)
        assert expected is not None, f"Unknown state_spec {self.state_spec!r}"
        assert mixture.flash_type in expected, (
            f"Physics state_spec is {self.state_spec.name}, but mixture.flash_type is "
            f"{mixture.flash_type.name} - call the matching mixture.init_*flash() method "
            f"({'/'.join(t.name for t in expected)} expected)"
        )

        # Assign flash evaluator to region
        region = region if region is not None else 0
        self.flash_evs[region if region is not None else 0] = mixture

    def init_physics(
        self,
        discr_type: str = 'tpfa',
        platform: str = 'cpu',
        itor_type: str = 'multilinear',
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
        Check that set_mixture() has been called and call PhysicsBase.init_physics() wrapper
        """
        assert len(self.flash_evs) > 0, (
            "No Mixture attached - call set_mixture() before init_physics()"
        )

        # Call PhysicsBase.init_physics() logic
        PhysicsBase.init_physics(
            self,
            discr_type=discr_type,
            platform=platform,
            itor_type=itor_type,
            itor_precision=itor_precision,
            verbose=verbose,
            is_barycentric=is_barycentric,
            n_solid=n_solid,
            parallel_evaluation=parallel_evaluation,
            n_workers=n_workers,
            evaluator_factory_hook=evaluator_factory_hook,
            verbose_evaluators=verbose_evaluators,
        )

    def get_flash_ev(self, region: int = None):
        """
        This class serves as flash_ev provider for PropertyContainer objects: return the
        composed Mixture instance

        :param region: Key of property region in PropertyContainers, defaults to 0
        """
        region = region if region is not None else 0
        return self.flash_evs[region]

    def get_density_ev_from_flash(self, phase_idx: int, region: int = None):
        """
        Get EoSDensity object that evaluates phase mass density for specified phase from FlashResults

        :param phase_idx: Phase index in flash output
        :param region: Key of property region in PropertyContainers, defaults to 0
        """
        region = region if region is not None else 0
        return EoSDensity(flash_ev=self.flash_evs[region], phase_idx=phase_idx)

    def get_enthalpy_ev_from_flash(self, phase_idx: int, region: int = None):
        """
        Get EoSEnthalpy object that evaluates phase enthalpy for specified phase from FlashResults

        :param phase_idx: Phase index in flash output
        :param region: Key of property region in PropertyContainers, defaults to 0
        """
        region = region if region is not None else 0
        return EoSEnthalpy(flash_ev=self.flash_evs[region], phase_idx=phase_idx)

    def get_fugacity_ev_from_flash(self, phase_idx: int, region: int = None):
        """
        Get EoSFugacity object that evaluates component fugacities for specified phase from FlashResults

        :param phase_idx: Phase index in flash output
        :param region: Key of property region in PropertyContainers, defaults to 0
        """
        region = region if region is not None else 0
        return EoSFugacity(flash_ev=self.flash_evs[region], phase_idx=phase_idx)
