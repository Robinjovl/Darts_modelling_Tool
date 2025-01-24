import numpy as np
from dartsflash.libflash import EoS, VdWP


NA = 6.02214076e23  # Avogadro's number [mol-1]
kB = 1.380649e-23  # Boltzmann constant [J/K]
R = NA * kB  # Gas constant [J/mol.K]


class EoSDensity:
    """
    This class can evaluate density (molar volume) from an EoS object.
    """
    def __init__(self, eos: EoS, Mw: list):
        """
        :param eos: Derived object from :class:`dartsflash.libflash.EoS`
        :type eos: EoS
        :param Mw: Molar weights of components [g/mol]
        :type Mw: list
        """
        self.eos = eos
        self.Mw = Mw

    def evaluate(self, pressure, temperature, x):
        """
        Evaluates the EoS for molar volume at given pressure, temperature and composition x.
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
        MW = np.sum(x * np.array(self.Mw)) * 1e-3  # kg/mol
        return MW / self.eos.V(pressure, temperature, x)  # kg/mol / m3/mol -> kg/m3


class EoSEnthalpy:
    """
    This class can evaluate phase enthalpy. It evaluates ideal gas enthalpy and EoS-derived residual enthalpy.
    """
    def __init__(self, eos: EoS):
        """
        :param eos: Derived object from :class:`dartsflash.libflash.EoS`
        :type eos: EoS
        """
        self.eos = eos

    def evaluate(self, pressure, temperature, x):
        """
        Evaluates the EoS for residual enthalpy at given pressure, temperature and composition x.
        Evaluates the ideal gas enthalpy at temperature and composition x.

        :param pressure: Pressure in bar
        :type pressure: float
        :param temperature: Temperature in Kelvin
        :type temperature: float
        :param x: Phase composition in mole fractions/mole numbers
        :type x: list

        :returns: Phase enthalpy in J/mol
        :rtype: float
        """
        H = self.eos.H_PT(pressure, temperature, x)  # H/R
        return H * R  # J/mol == kJ/kmol


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

        H = self.eos.H_PT(pressure, temperature, X)  # H/R

        if self.xH is not None:
            nH = self.xH[0] / self.xH[1]
            return H * (nH + 1.) * R
        else:
            return H * R
