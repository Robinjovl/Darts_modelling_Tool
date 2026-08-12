from dartsflash.mixtures import IAPWS

from darts.physics.base.physics import (
    HistoryField,
    Iterable,
    PhysicsBase,
    timer_node,
)
from darts.physics.eos_physics import _EXPECTED_FLASH_TYPES, EoSPhysics


class IAPWSPhysics(EoSPhysics):
    """
    Implementation of EoS-based Physics with IAPWS-specific implementation
    - Inherited from EoSPhysics;
    - Composes a DARTS-flash IAPWS(Mixture) instance, attached separately via :meth:`set_mixture`
    - Checks physics consistency with EoS and flash definition (components, phases, state specification)

    Usage:
        physics = IAPWSPhysics(phases, timer, axes_step, ...)  # same args as EoSPhysics except component-related args
        mixture = IAPWS(iapws_ideal: bool = True, ice_phase: bool = False)
        mixture.init_*flash()                            # init_ptflash/init_pxflash/init_negativeflash
        physics.set_mixture(mixture)                     # checks components immediately
        physics.init_physics(...)                        # checks phases/correlations
    """

    def __init__(
        self,
        phases: list,
        timer: timer_node,
        axes_step: list[float],
        axes_origin: list[float] = None,
        state_spec: PhysicsBase.StateSpecification = PhysicsBase.StateSpecification.P,
        cache: bool = False,
        history_fields: Iterable[HistoryField] | None = None,
    ):
        """
        Constructor initializes EoSPhysics logic and attaches an IAPWS instance to flash_evs[0].
        #TODO: create cleaner logic for multiple regions

        :param phases: List of phase labels, expected in flash output order (see
            ``self.flash_evs[region].flash_params.eos_order`` and per-EoS ``root_order``)
        """
        EoSPhysics.__init__(
            self,
            components=["H2O"],
            phases=phases,
            timer=timer,
            axes_step=axes_step,
            axes_origin=axes_origin,
            state_spec=state_spec,
            cache=cache,
            history_fields=history_fields,
        )

        self.flash_evs: dict[str | int | None, IAPWS] = {}

    def set_mixture(
        self, mixture: IAPWS, phases_to_eos: dict[tuple] = None, region: int = None
    ) -> None:
        """
        Attach a DARTS-flash IAPWS instance to this physics object as ``self.flash_evs[region]``.

        Checks that:
        - IAPWS components + salts match the Physics.components
        - TODO: Phase labels in Physics object are consistent with phase types specified in IAPWS
        - IAPWS.FlashType is compatible with Physics.StateSpecification

        :param mixture: Configured IAPWS instance
        :param phases_to_eos: Dictionary of phase labels to phase types, default is None which throws warning
        :param region: Key of property region in PropertyContainers, defaults to 0
        """
        assert isinstance(mixture, IAPWS), "Provide an object of type dartsflash.IAPWS"

        # Assert that the physics' component list matches the composed IAPWS's ``comp_data``.
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
