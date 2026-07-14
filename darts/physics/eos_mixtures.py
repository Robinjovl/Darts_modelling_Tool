from dartsflash.components import CompData
from dartsflash.libflash import EoS
from dartsflash.mixtures import Mixture

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer


class EoSMixture(Compositional, Mixture):
    """
    Implementation of EoS-based Physics

    Multiple inheritance of Compositional and DARTS-flash Mixture classes
    - Constructor defaults to Compositional constructor
    - set_mixture() method calls Mixture constructor and specifies EoS objects
    - init_flash() method calls Mixture.init_flash() method to initialize Flash object
    """

    def set_mixture(
        self,
        components: list = None,
        comp_data: CompData = None,
        mixture_name: str = None,
        vl_eos: str | EoS = None,
        vl_trial_comps: list = None,
        vl_root_order: list = None,
        vl_rich_phase_order: list = None,
        vl_stability_tol: float = 1e-20,
        vl_stability_switch_tol: float = 1e-3,
        vl_active_comp_idxs: list = None,
        aq_eos: str | EoS = None,
        aq_stability_tol: float = 1e-20,
        aq_stability_switch_tol: float = 1e-3,
        aq_active_comp_idxs: list = None,
        ice_eos: str | EoS = None,
        salt_eos: str | EoS | list = None,
        si_eos: str | EoS = None,
        si_stability_tol: float = 1e-20,
        si_stability_switch_tol: float = 1e2,
        si_active_comp_idxs: list = None,
        sii_eos: str | EoS = None,
        sii_stability_tol: float = 1e-20,
        sii_stability_switch_tol: float = 1e2,
        sii_active_comp_idxs: list = None,
        sh_eos: str | EoS = None,
        sh_stability_tol: float = 1e-20,
        sh_stability_switch_tol: float = 1e2,
        sh_active_comp_idxs: list = None,
    ):
        """
        Specify mixture: components, phase types, EoS and EoS parameters

        :param components: List of components, only used to initialize CompData object if no CompData has been provided
        :param comp_data: CompData object
        :param mixture_name: Name of mixture for filename
        :param vl_eos: V/L EoS object or name for EoS factory
        :param vl_trial_comps: Trial compositions for V/L EoS stability test
        :param vl_root_order: Root order for V/L EoS
        :param vl_rich_phase_order: Rich phase order for V/L EoS
        :param vl_stability_tol: Tolerance for V/L EoS stability test
        :param vl_stability_switch_tol: Switch tolerance for V/L EoS stability test
        :param vl_active_comp_idxs: Indices of active components in V/L phases
        :param aq_eos: Aq EoS object or name for EoS factory
        :param aq_stability_tol: Tolerance for Aq EoS stability test
        :param aq_stability_switch_tol: Switch tolerance for Aq EoS stability test
        :param aq_active_comp_idxs: Indices of active components in Aq phase
        :param ice_eos: Ice EoS object or name for EoS factory
        :param salt_eos: (List of) salt EoS object or name(s) for EoS factory
        :param si_eos: sI EoS object or name for EoS factory
        :param si_stability_tol: Tolerance for sI EoS stability test
        :param si_stability_switch_tol: Switch tolerance for sI EoS stability test
        :param si_active_comp_idxs: Indices of active components in sI phase
        :param sii_eos: sII EoS object or name for EoS factory
        :param sii_stability_tol: Tolerance for sII EoS stability test
        :param sii_stability_switch_tol: Switch tolerance for sII EoS stability test
        :param sii_active_comp_idxs: Indices of active components in sII phase
        :param sh_eos: sH EoS object or name for EoS factory
        :param sh_stability_tol: Tolerance for sH EoS stability test
        :param sh_stability_switch_tol: Switch tolerance for sH EoS stability test
        :param sh_active_comp_idxs: Indices of active components in sH phase
        """
        # If no CompData object has been provided, create instance from list of components
        if comp_data is None:
            assert components is not None, (
                "Neither of CompData object and components list are not provided"
            )
            comp_data = CompData(
                components=components, salt_components=None, setprops=True
            )

        # Initialize Mixture constructor
        Mixture.__init__(
            self,
            comp_data=comp_data,
            mixture_name=mixture_name,
            vl_phase=True,
            hybrid_aq=aq_eos is not None,
            ice_phase=ice_eos is not None,
            salt_phase=salt_eos is not None,
            si_phase=si_eos is not None,
            sii_phase=sii_eos is not None,
            sh_phase=sh_eos is not None,
        )

        # Set V/L EoS object
        if vl_eos is not None:
            Mixture.set_vl_eos(
                self,
                vl_eos=vl_eos is not None,
                root_order=vl_root_order,
                trial_comps=vl_trial_comps,
                rich_phase_order=vl_rich_phase_order,
                stability_tol=vl_stability_tol,
                switch_tol=vl_stability_switch_tol,
                active_components=vl_active_comp_idxs,
            )

        # Set Aq EoS object
        if aq_eos is not None:
            Mixture.set_aq_eos(
                self,
                aq_eos=aq_eos is not None,
                stability_tol=aq_stability_tol,
                switch_tol=aq_stability_switch_tol,
                use_gmix=True,
                active_components=aq_active_comp_idxs,
            )

        # Set Ice EoS object
        if ice_eos is not None:
            Mixture.set_ice_eos(
                self,
                ice_eos=ice_eos,
                use_gmix=True,
            )

        # Set hydrate EoSs
        for name, hydrate_type in zip(
            ["si", "sii", "sh"], ["sI", "sII", "sH"], strict=False
        ):
            if eval(name + "_eos") is not None:
                Mixture.set_h_eos(
                    self,
                    hydrate_type=hydrate_type,
                    vdwp_type="Ballard",
                    stability_tol=eval(name + "_stability_tol"),
                    switch_tol=eval(name + "_stability_switch_tol"),
                    use_gmix=True,
                    gmix_tol=eval(name + "_stability_tol"),
                    gmix_switch_tol=eval(name + "_stability_switch_tol"),
                    active_components=eval(name + "_active_comp_idxs"),
                )

    def set_properties(self, regions: list):
        """
        Create PropertyContainer for each region.
        In addition, set properties directly related to thermodynamics: flash, density and enthalpy
        """
        from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

        for region in regions:
            pc = PropertyContainer(
                phases_name=self.phases,
                components_name=self.components,
                Mw=self.comp_data.Mw,
                eps_z=self.epsilon_z * self.sim_eps_multiplier,
            )
            self.add_property_region(pc, region)

            # Point to self for flash_ev
            pc.flash_ev = self

            # Set EoSDensity and EoSEnthalpy methods for each phase
            for phase in self.phases:
                pc.density_ev[phase] = EoSDensity(
                    eos=self.eos[phase],
                    Mw=self.comp_data.Mw,
                    root_flag=self.root_type[phase],
                    ions=self.salt_components,
                    combined_ions_stoichiometry=None,
                )
                pc.enthalpy_ev[phase] = EoSEnthalpy(eos=self.eos[phase])

        return
