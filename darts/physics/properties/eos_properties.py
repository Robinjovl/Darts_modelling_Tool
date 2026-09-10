import warnings

import numpy as np
from dartsflash.dartsflash import DARTSFlash
from dartsflash.libflash import EoS, VdWP

NA = 6.02214076e23  # Avogadro's number [mol-1]
kB = 1.380649e-23  # Boltzmann constant [J/K]
R = NA * kB  # Gas constant [J/mol.K]


class EoSDensity:
    """
    This class evaluates density (molar volume) from an EoS, either through Flash or directly from EoS.
    """

    def __init__(
        self,
        flash_ev: DARTSFlash = None,
        phase_idx: int = None,
        eos: EoS = None,
        root_flag: EoS.RootFlag = EoS.RootFlag.STABLE,
        ions: list = None,
        combined_ions_stoichiometry: list = None,
    ):
        """
        Constructor of EoSDensity object. Option to either provide flash object + phase idx, or EoS object directly.

        :param flash_ev: Flash object
        :type flash_ev: DARTSFlash
        :param phase_idx: Phase index of specified phase in FlashResults
        :type phase_idx: int
        :param eos: Derived object from :class:`dartsflash.libflash.EoS`
        :type eos: EoS
        :param root_flag: EoS root flag, 0) STABLE, 1) MIN (Liquid), 2) MAX (Vapour); default is STABLE
        :param ions: List of ions, default is None
        :param combined_ions_stoichiometry: List of normalized ion stoichiometry in case they have been lumped in flash output, default is None
        """
        assert (flash_ev is not None and phase_idx is not None) or eos is not None, (
            "Specify either flash object + phase idx or EoS object to "
        )
        if flash_ev is not None and eos is not None:
            warnings.warn(
                "Both flash and EoS objects defined, using Flash object", stacklevel=2
            )

        # Flash object
        self.flash_ev = flash_ev
        self.phase_idx = phase_idx
        self.evaluate_PT_bool = False

        # EoS object and RootFlag
        self.eos = eos
        self.root_flag = root_flag

        # Ions and combined ions
        self.ions = ions
        self.combined_ions_stoichiometry = combined_ions_stoichiometry

    def evaluate(self, pressure, temperature, x):
        """
        Evaluates the EoS for mass density at given pressure, temperature and composition x.

        :param pressure: Pressure in bar
        :type pressure: float
        :param temperature: Temperature in Kelvin
        :type temperature: float
        :param x: Phase composition in mole fractions/mole numbers
        :type x: list

        :returns: Phase density in kg/m3
        :rtype: float
        """
        if self.flash_ev is not None:
            flash_results = self.flash_ev.get_flash_results(
                derivs=False, evaluate_PT=self.evaluate_PT_bool
            )
            # In case of PX-flash, we need to evaluate PT-flash and properties for initialization
            if self.evaluate_PT_bool:
                eos_results = self.flash_ev.f.get_pt_phase_properties(
                    flash_results=flash_results,
                    phase_idx=self.phase_idx,
                    calc_mass_density=True,
                )
            else:
                eos_results = self.flash_ev.f.get_phase_properties(
                    flash_results=flash_results,
                    phase_idx=self.phase_idx,
                    calc_mass_density=True,
                )
            # TODO: Mistake in darts-flash v0.13.0, returns N * Mw / Vm
            # Divide by phase mole numbers to obtain molar density
            nu_phase = np.array(flash_results.nu)[self.phase_idx]
            return eos_results.get_phase_mass_density().value / nu_phase

        else:
            self.eos.set_root_flag(self.root_flag)

            if self.combined_ions_stoichiometry is not None:
                xi = np.append(
                    x[:-1], x[-1] * np.array(self.combined_ions_stoichiometry)
                )
            else:
                xi = x

            Mw = np.sum(np.array(xi) * self.eos.get_comp_data().Mw)
            return Mw * 1e-3 / self.eos.V(pressure, temperature, xi)  # kg/m3


class EoSEnthalpy:
    """
    This class evaluates phase (ideal + residual) enthalpy from an EoS, either through Flash or directly from EoS.
    """

    def __init__(
        self,
        flash_ev: DARTSFlash = None,
        phase_idx: int = None,
        eos: EoS = None,
        root_flag: EoS.RootFlag = EoS.RootFlag.STABLE,
        ions: list = None,
        combined_ions_stoichiometry: list = None,
    ):
        """
        Constructor of EoSEnthalpy object. Option to either provide flash object + phase idx, or EoS object directly.

        :param flash_ev: Flash object
        :type flash_ev: DARTSFlash
        :param phase_idx: Phase index of specified phase in FlashResults
        :type phase_idx: int
        :param eos: Derived object from :class:`dartsflash.libflash.EoS`
        :type eos: EoS
        :param root_flag: EoS root flag, 0) STABLE, 1) MIN (Liquid), 2) MAX (Vapour); default is STABLE
        :param ions: List of ions, default is None
        :param combined_ions_stoichiometry: List of normalized ion stoichiometry in case they have been lumped in flash output, default is None
        """
        assert (flash_ev is not None and phase_idx is not None) or eos is not None, (
            "Specify either flash object + phase idx or EoS object to EoSDensity"
        )
        if flash_ev is not None and eos is not None:
            warnings.warn(
                "Both flash and EoS objects defined, using Flash object", stacklevel=2
            )

        # Flash object
        self.flash_ev = flash_ev
        self.phase_idx = phase_idx
        self.evaluate_PT_bool = False

        # EoS object and RootFlag
        self.eos = eos
        self.root_flag = root_flag

        # Ions and combined ions
        self.ions = ions
        self.combined_ions_stoichiometry = combined_ions_stoichiometry

    def evaluate(self, pressure, temperature, x):
        """
        Evaluates the EoS for phase (ideal + residual) enthalpy at given pressure, temperature and composition x.

        :param pressure: Pressure in bar
        :type pressure: float
        :param temperature: Temperature in Kelvin
        :type temperature: float
        :param x: Phase composition in mole fractions/mole numbers
        :type x: list

        :returns: Phase enthalpy in J/mol
        :rtype: float
        """
        if self.flash_ev is not None:
            flash_results = self.flash_ev.get_flash_results(
                derivs=False, evaluate_PT=self.evaluate_PT_bool
            )
            # In case of PX-flash, we need to evaluate PT-flash and properties for initialization
            if self.evaluate_PT_bool:
                eos_results = self.flash_ev.f.get_pt_phase_properties(
                    flash_results=flash_results,
                    phase_idx=self.phase_idx,
                    calc_enthalpy=True,
                )
            else:
                eos_results = self.flash_ev.f.get_phase_properties(
                    flash_results=flash_results,
                    phase_idx=self.phase_idx,
                    calc_enthalpy=True,
                )
            # TODO: Mistake in darts-flash v0.13.0, returns total enthalpy, not molar enthalpy
            # Divide by phase mole numbers to obtain molar enthalpy
            nu_phase = np.array(flash_results.nu)[self.phase_idx]
            return eos_results.get_phase_enthalpy().value * R / nu_phase

        else:
            self.eos.set_root_flag(self.root_flag)

            if self.combined_ions_stoichiometry is not None:
                xi = np.append(
                    x[:-1], x[-1] * np.array(self.combined_ions_stoichiometry)
                )
            else:
                xi = x

            H = self.eos.H(pressure, temperature, xi)  # H/R
            return H * R  # J/mol == kJ/kmol


class EoSFugacity:
    """
    This class evaluates component fugacities from an EoS, either through Flash or directly from EoS.
    """

    def __init__(
        self,
        flash_ev: DARTSFlash = None,
        phase_idx: int = None,
        eos: EoS = None,
        root_flag: EoS.RootFlag = EoS.RootFlag.STABLE,
        ions: list = None,
        combined_ions_stoichiometry: list = None,
    ):
        """
        Constructor of EoSFugacity object. Option to either provide flash object + phase idx, or EoS object directly.

        :param flash_ev: Flash object
        :type flash_ev: DARTSFlash
        :param phase_idx: Phase index of specified phase in FlashResults
        :type phase_idx: int
        :param eos: Derived object from :class:`dartsflash.libflash.EoS`
        :type eos: EoS
        :param root_flag: EoS root flag, 0) STABLE, 1) MIN (Liquid), 2) MAX (Vapour); default is STABLE
        :param ions: List of ions, default is None
        :param combined_ions_stoichiometry: List of normalized ion stoichiometry in case they have been lumped in flash output, default is None
        """
        assert (flash_ev is not None and phase_idx is not None) or eos is not None, (
            "Specify either flash object + phase idx or EoS object to "
        )
        if flash_ev is not None and eos is not None:
            warnings.warn(
                "Both flash and EoS objects defined, using Flash object", stacklevel=2
            )

        # Flash object
        self.flash_ev = flash_ev
        self.phase_idx = phase_idx
        self.evaluate_PT_bool = False

        # EoS object and RootFlag
        self.eos = eos
        self.root_flag = root_flag

        # Ions and combined ions
        self.ions = ions
        self.combined_ions_stoichiometry = combined_ions_stoichiometry

    def evaluate(self, pressure, temperature, x):
        """
        Evaluates the EoS for component fugacity coefficients lnphii at given pressure, temperature and composition x.

        :param pressure: Pressure in bar
        :type pressure: float
        :param temperature: Temperature in Kelvin
        :type temperature: float
        :param x: Phase composition in mole fractions/mole numbers
        :type x: list

        :returns: Component fugacity coefficients lnphii
        :rtype: list
        """
        if self.flash_ev is not None:
            flash_results = self.flash_ev.get_flash_results(
                derivs=False, evaluate_PT=self.evaluate_PT_bool
            )
            # In case of PX-flash, we need to evaluate PT-flash and properties for initialization
            if self.evaluate_PT_bool:
                eos_results = self.flash_ev.f.get_pt_phase_properties(
                    flash_results=flash_results,
                    phase_idx=self.phase_idx,
                    calc_fugacity=True,
                )
            else:
                eos_results = self.flash_ev.f.get_phase_properties(
                    flash_results=flash_results,
                    phase_idx=self.phase_idx,
                    calc_fugacity=True,
                )
            return eos_results.get_phase_fugacity().value

        else:
            self.eos.set_root_flag(self.root_flag)

            if self.combined_ions_stoichiometry is not None:
                xi = np.append(
                    x[:-1], x[-1] * np.array(self.combined_ions_stoichiometry)
                )
            else:
                xi = x

            return self.eos.lnphi(pressure, temperature, xi)


class VdWPDensity:
    """
    This class can evaluate hydrate density (molar volume) from a Van der Waals-Platteeuw EoS (VdWP) object.
    """

    def __init__(self, eos: VdWP, Mw: list, xH: list = None):
        """
        :param eos: Derived object from :class:`dartsflash.libflash.VdWP`
        :type eos: VdWP
        :param Mw: Molar weights of components [g/mol]
        :type Mw: list
        :param xH: Hydrate composition xH, default is None for which it evaluates hydrate composition from EoS
        :type xH: list
        """
        self.eos = eos
        self.xH = xH
        self.Mw = Mw

    def evaluate(self, pressure, temperature, x: list = None):
        """
        Evaluates the VdWP EoS for molar volume at given pressure, temperature and composition x.
        If xH has been provided in constructor, it takes this composition. Otherwise, it evaluates the EoS for composition.
        Calculates mixture molar weight MW and translates molar volume (m3/mol) to density (kg/m3)

        :param pressure: Pressure in bar
        :type pressure: float
        :param temperature: Temperature in Kelvin
        :type temperature: float
        :param x: Phase composition in mole fractions/mole numbers
        :type x: list

        :returns: Phase density in kg/m3
        :rtype: float
        """
        X = self.xH if self.xH is not None else x
        MW = np.sum(X * np.array(self.Mw)) * 1e-3  # kg/mol

        return MW / self.eos.V(pressure, temperature, X)  # kg/mol / mol/m3


class VdWPEnthalpy:
    """
    This class can evaluate hydrate phase enthalpy. It evaluates ideal gas enthalpy and VdWP-derived residual enthalpy.
    """

    def __init__(self, eos: VdWP, xH: list = None):
        """
        :param eos: Derived object from :class:`dartsflash.libflash.VdWP`
        :type eos: VdWP
        :param xH: Hydrate composition xH, default is None for which it evaluates hydrate composition from EoS
        :type xH: list
        """
        self.eos = eos
        self.xH = xH

    def evaluate(self, pressure, temperature, x: list = None):
        """
        Evaluates the VdWP EoS for residual enthalpy given pressure, temperature and composition x.
        Evaluates the ideal gas enthalpy at temperature and composition x.

        If xH has been provided in constructor, it takes this composition. Otherwise, it evaluates the EoS for composition.

        :param pressure: Pressure in bar
        :type pressure: float
        :param temperature: Temperature in Kelvin
        :type temperature: float
        :param x: Phase composition in mole fractions/mole numbers
        :type x: list

        :returns: Phase enthalpy in J/mol
        :rtype: float
        """
        X = self.xH if self.xH is not None else x

        H = self.eos.H(pressure, temperature, X)  # H/R

        if self.xH is not None:
            nH = self.xH[0] / self.xH[1]
            return H * (nH + 1.0) * R
        else:
            return H * R
