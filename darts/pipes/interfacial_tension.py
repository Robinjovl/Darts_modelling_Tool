import darts.pipes.library as library
from darts.pipes.units import *


class IFT_multicomponent_MCM:
    """
    Liquid-gas interfacial tension (IFT) or surface tension for multi-component fluids
    Macleod-Sugden surface tension model (MCS) (used in Multiflash of OLGA, PVTi, WinProp, and McCain's PVT book)
    The parameters used in this correlation are as follows:
    parachor is the parachor of the component (parachors of different components are available in the library file)
    rhoG_molar is the molar density of the gaseous phase [mol/cm3]
    rhoL_molar is the molar density of the liquid phase [mol/cm3]
    rhoG is the molar density of the gaseous phase [gram/cm3]
    rhoL is the molar density of the liquid phase in gram/cm3
    MW is the molecular weight of the component [gram/mol]
    IFT that this correlation gives is in dyne/cm.
    """

    def __init__(self, components_names: list):
        """
        :param components_names: Names of the components
        :type components_names: list
        """
        self.components_names = components_names
        num_components = len(components_names)

        # Get components MW and parachor from the library
        self.MW = np.zeros(num_components)
        self.parachor = np.zeros(num_components)

        for i in range(num_components):
            try:
                self.MW[i] = library.components_molecular_weights[components_names[i]]
            except KeyError as err:
                raise KeyError(
                    f"Molecular weight of {components_names[i]} is not in the library!"
                ) from err

            try:
                self.parachor[i] = library.components_parachors[components_names[i]]
            except KeyError as err:
                raise KeyError(
                    f"Parachor of {components_names[i]} is not in the library!"
                ) from err

    def evaluate(self, rhoG, rhoL, xG_mass, xL_mass):
        """
        :param rhoG: Gas density in kg/m3
        :param rhoL: Liquid density in kg/m3
        :param xG_mass: Mass fractions of components in the gaseous phase
        :param xL_mass: Mass fractions of components in the liquid phase

        :returns IFT: Interfacial tension in N/m
        """
        # IFT = (parachor * (rhoL_molar - rhoG_molar)) ** 4   # for molar densities
        rhoG = convertTo(rhoG, gram() / (centi() * meter()) ** 3)
        rhoL = convertTo(rhoL, gram() / (centi() * meter()) ** 3)
        IFT = (
            sum(self.parachor * (rhoL * xL_mass - rhoG * xG_mass) / self.MW)
        ) ** 4  # for mass densities
        # Convert IFT from MCS correlation (dyne/cm) to N/m
        IFT = IFT * dyne() / (centi() * meter())
        return IFT
