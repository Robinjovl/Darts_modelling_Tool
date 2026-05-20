from typing import Literal

from iapws._iapws import _Viscosity
from iapws.iapws97 import (
    Pmin,
    _Backward1_T_Ph,
    _Backward2_T_Ph,
    _Bound_Ph,
    _Bound_TP,
    _Region1,
    _Region2,
    _Region4,
    _ThCond,
    _TSat_P,
)
from scipy.optimize import newton

from darts.interpolators import property_evaluator_iface
from darts.physics.properties.evaluator_base import (
    EvaluatorBase,
    EvaluatorConfigBase,
    register_evaluator,
)


class _EvaluatorIfaceMeta(type(property_evaluator_iface), type(EvaluatorBase)):
    """Combined metaclass so a class can inherit both the C++ pybind11
    ``property_evaluator_iface`` and the Python ``EvaluatorBase`` ABC."""


class _NoArgIapwsConfig(EvaluatorConfigBase):
    """Base for parameter-less IAPWS evaluators."""


class WaterDensityPropertyEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for water_density_property_evaluator (Region 1 only)."""

    kind: Literal["iapws_water_density_region1"] = "iapws_water_density_region1"


class TemperatureRegion1EvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for temperature_region1_evaluator."""

    kind: Literal["iapws_temperature_region1"] = "iapws_temperature_region1"


class IapwsEnthalpyRegion1EvaluatorConfig(EvaluatorConfigBase):
    """Configuration for iapws_enthalpy_region1_evaluator."""

    kind: Literal["iapws_enthalpy_region1"] = "iapws_enthalpy_region1"
    temperature: float


class IapwsViscosityEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_viscosity_evaluator (Region 1)."""

    kind: Literal["iapws_viscosity_region1"] = "iapws_viscosity_region1"


class IapwsTotalEnthalpyEvalutorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_total_enthalpy_evalutor."""

    kind: Literal["iapws_total_enthalpy"] = "iapws_total_enthalpy"


class IapwsTemperatureEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_temperature_evaluator."""

    kind: Literal["iapws_temperature"] = "iapws_temperature"


class IapwsWaterEnthalpyEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_water_enthalpy_evaluator."""

    kind: Literal["iapws_water_enthalpy"] = "iapws_water_enthalpy"


class IapwsSteamEnthalpyEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_steam_enthalpy_evaluator."""

    kind: Literal["iapws_steam_enthalpy"] = "iapws_steam_enthalpy"


class IapwsWaterSaturationEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_water_saturation_evaluator."""

    kind: Literal["iapws_water_saturation"] = "iapws_water_saturation"


class IapwsSteamSaturationEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_steam_saturation_evaluator."""

    kind: Literal["iapws_steam_saturation"] = "iapws_steam_saturation"


class IapwsWaterRelpermEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_water_relperm_evaluator."""

    kind: Literal["iapws_water_relperm"] = "iapws_water_relperm"


class IapwsSteamRelpermEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_steam_relperm_evaluator."""

    kind: Literal["iapws_steam_relperm"] = "iapws_steam_relperm"


class IapwsWaterDensityEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_water_density_evaluator (P-h system)."""

    kind: Literal["iapws_water_density"] = "iapws_water_density"


class IapwsSteamDensityEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_steam_density_evaluator (P-h system)."""

    kind: Literal["iapws_steam_density"] = "iapws_steam_density"


class IapwsWaterViscosityEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_water_viscosity_evaluator (P-h system)."""

    kind: Literal["iapws_water_viscosity"] = "iapws_water_viscosity"


class IapwsSteamViscosityEvaluatorConfig(_NoArgIapwsConfig):
    """Configuration for iapws_steam_viscosity_evaluator (P-h system)."""

    kind: Literal["iapws_steam_viscosity"] = "iapws_steam_viscosity"


class DensityIapwsWaterConfig(_NoArgIapwsConfig):
    """Configuration for Density_iapws_water (P-T system)."""

    kind: Literal["iapws_pt_density_water"] = "iapws_pt_density_water"


class DensityIapwsSteamConfig(_NoArgIapwsConfig):
    """Configuration for Density_iapws_steam (P-T system)."""

    kind: Literal["iapws_pt_density_steam"] = "iapws_pt_density_steam"


class ViscosityIapwsWaterConfig(_NoArgIapwsConfig):
    """Configuration for Viscosity_iapws_water (P-T system)."""

    kind: Literal["iapws_pt_viscosity_water"] = "iapws_pt_viscosity_water"


class ViscosityIapwsSteamConfig(_NoArgIapwsConfig):
    """Configuration for Viscosity_iapws_steam (P-T system)."""

    kind: Literal["iapws_pt_viscosity_steam"] = "iapws_pt_viscosity_steam"


class SaturationIapwsWaterConfig(_NoArgIapwsConfig):
    """Configuration for Saturation_iapws_water (P-T system)."""

    kind: Literal["iapws_pt_saturation_water"] = "iapws_pt_saturation_water"


class SaturationIapwsSteamConfig(_NoArgIapwsConfig):
    """Configuration for Saturation_iapws_steam (P-T system)."""

    kind: Literal["iapws_pt_saturation_steam"] = "iapws_pt_saturation_steam"


class RelpermIapwsWaterConfig(_NoArgIapwsConfig):
    """Configuration for Relperm_iapws_water (P-T system)."""

    kind: Literal["iapws_pt_relperm_water"] = "iapws_pt_relperm_water"


class RelpermIapwsSteamConfig(_NoArgIapwsConfig):
    """Configuration for Relperm_iapws_steam (P-T system)."""

    kind: Literal["iapws_pt_relperm_steam"] = "iapws_pt_relperm_steam"


class EnthalpyIapwsWaterConfig(_NoArgIapwsConfig):
    """Configuration for Enthalpy_iapws_water (P-T system)."""

    kind: Literal["iapws_pt_enthalpy_water"] = "iapws_pt_enthalpy_water"


class EnthalpyIapwsSteamConfig(_NoArgIapwsConfig):
    """Configuration for Enthalpy_iapws_steam (P-T system)."""

    kind: Literal["iapws_pt_enthalpy_steam"] = "iapws_pt_enthalpy_steam"


class ConductivityIapwsWaterConfig(_NoArgIapwsConfig):
    """Configuration for Conductivity_iapws_water (P-T system)."""

    kind: Literal["iapws_pt_conductivity_water"] = "iapws_pt_conductivity_water"


class ConductivityIapwsSteamConfig(_NoArgIapwsConfig):
    """Configuration for Conductivity_iapws_steam (P-T system)."""

    kind: Literal["iapws_pt_conductivity_steam"] = "iapws_pt_conductivity_steam"


class water_density_property_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS Region 1 water density [kmol/m3] from a (P, h) state."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: water molar density [kmol/m3]
        :rtype: float
        """
        temperature = temperature_region1_evaluator()
        temp = temperature.evaluate(state)
        water_density = 1 / _Region1(temp, float(state[0]) * 0.1)['v']
        return water_density / 18.015


register_evaluator(
    "iapws_water_density_region1",
    water_density_property_evaluator,
    WaterDensityPropertyEvaluatorConfig,
)


class temperature_region1_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS Region 1 backward temperature from a (P, h) state."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: temperature [K]
        :rtype: float
        """
        return _Backward1_T_Ph(float(state[0]) * 0.1, state[1] / 18.015)


register_evaluator(
    "iapws_temperature_region1",
    temperature_region1_evaluator,
    TemperatureRegion1EvaluatorConfig,
)


class iapws_enthalpy_region1_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS Region 1 enthalpy [kJ/kmol] at a fixed temperature."""

    def __init__(self, temperature):
        """
        :param temperature: temperature [K]
        :type temperature: float
        """
        super().__init__()
        self.temperature = temperature

    def evaluate(self, state):
        """
        :param state: state vector with pressure as the first element
        :type state: list[float] | np.ndarray
        :return: enthalpy [kJ/kmol]
        :rtype: float
        """
        return (
            _Region1(self.temperature, float(state[0]) * 0.1)['h'] * 18.015
        )  # kJ/kmol


register_evaluator(
    "iapws_enthalpy_region1",
    iapws_enthalpy_region1_evaluator,
    IapwsEnthalpyRegion1EvaluatorConfig,
)


class iapws_viscosity_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS Region 1 water viscosity [cP] from a (P, h) state."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: water viscosity [cP]
        :rtype: float
        """
        temperature = temperature_region1_evaluator().evaluate(state)
        density = water_density_property_evaluator().evaluate(state)
        return _Viscosity(density, temperature) * 1000


register_evaluator(
    "iapws_viscosity_region1",
    iapws_viscosity_evaluator,
    IapwsViscosityEvaluatorConfig,
)


# ====================================== Properties for Region 1 and 4 ============================================
class iapws_total_enthalpy_evalutor(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS total fluid enthalpy [kJ/kmol] across Regions 1, 2, and 4."""

    def __init__(
        self,
    ):
        """No parameters."""
        super().__init__()

    def evaluate(self, state, temperature):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :param temperature: temperature [K]
        :type temperature: float
        :return: total enthalpy [kJ/kmol]
        :rtype: float
        """
        P = state[0] * 0.1
        region = _Bound_TP(temperature, P)
        if region == 1:
            h = _Region1(temperature, P)["h"] * 18.015
        elif region == 4:
            Steam_sat = iapws_steam_saturation_evaluator().evaluate(state)
            rho_steam = iapws_steam_density_evaluator().evaluate(state) / 18.015
            rho_water = iapws_water_density_evaluator().evaluate(state) / 18.015
            x = (
                Steam_sat
                * rho_steam
                / (Steam_sat * rho_steam + (1 - Steam_sat) * rho_water)
            )
            h = _Region4(P, x)["h"] * 18.015
        elif region == 2:
            h = _Region2(temperature, P)["h"] * 18.015
        else:
            raise NotImplementedError(
                'Variables out of bound: p=' + str(P) + ' region=' + str(region)
            )
        return h


register_evaluator(
    "iapws_total_enthalpy",
    iapws_total_enthalpy_evalutor,
    IapwsTotalEnthalpyEvalutorConfig,
)


class iapws_temperature_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS backward temperature [K] from (P, h), branching across Regions 1, 2, and 4."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: temperature [K]
        :rtype: float
        """
        P, h = state[0] * 0.1, state[1] / 18.015
        if P < Pmin:
            P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin:
            h = hmin

        region = _Bound_Ph(P, h)
        if region == 1:
            T = _Backward1_T_Ph(P, h)
        elif region == 4:
            T = _TSat_P(P)
        elif region == 2:
            T = _Backward2_T_Ph(P, h)
        else:
            raise NotImplementedError(
                'Variables out of bound: p='
                + str(state[0])
                + ' bars, h='
                + str(state[1])
                + ' kJ/kmol, region='
                + str(region)
            )
        return T


register_evaluator(
    "iapws_temperature",
    iapws_temperature_evaluator,
    IapwsTemperatureEvaluatorConfig,
)


class iapws_water_enthalpy_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS liquid water enthalpy [kJ/kmol] from (P, h), branching across regions."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: water enthalpy [kJ/kmol]
        :rtype: float
        """
        P, h = state[0] * 0.1, state[1] / 18.015
        if P < Pmin:
            P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin:
            h = hmin
        region = _Bound_Ph(P, h)
        if region == 1:
            water_enth = h
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                water_enth = _Region4(P, 0)["h"]
            else:
                raise NotImplementedError(
                    'Variables out of bound: p='
                    + str(state[0])
                    + ' bars, h='
                    + str(state[1])
                    + ' kJ/kmol, region='
                    + str(region)
                )
        elif region == 2:
            water_enth = 0
        else:
            print(region)
            raise NotImplementedError(
                'Variables out of bound: p='
                + str(state[0])
                + ' bars, h='
                + str(state[1])
                + ' kJ/kmol, region='
                + str(region)
            )
        return water_enth * 18.015


register_evaluator(
    "iapws_water_enthalpy",
    iapws_water_enthalpy_evaluator,
    IapwsWaterEnthalpyEvaluatorConfig,
)


class iapws_steam_enthalpy_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS steam enthalpy [kJ/kmol] from (P, h), branching across regions."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: steam enthalpy [kJ/kmol]
        :rtype: float
        """
        P, h = state[0] * 0.1, state[1] / 18.015
        if P < Pmin:
            P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin:
            h = hmin

        region = _Bound_Ph(P, h)
        if region == 1:
            steam_enth = 0
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                steam_enth = _Region4(P, 1)["h"]
            else:
                raise NotImplementedError(
                    'Variables out of bound: p='
                    + str(state[0])
                    + ' bars, h='
                    + str(state[1])
                    + ' kJ/kmol, region='
                    + str(region)
                )
        elif region == 2:
            To = _Backward2_T_Ph(P, h)
            T = newton(lambda T: _Region2(T, P)["h"] - h, To)
            steam_enth = _Region2(T, P)["h"]
        else:
            raise NotImplementedError(
                'Variables out of bound: p='
                + str(state[0])
                + ' bars, h='
                + str(state[1])
                + ' kJ/kmol, region='
                + str(region)
            )
        return steam_enth * 18.015


register_evaluator(
    "iapws_steam_enthalpy",
    iapws_steam_enthalpy_evaluator,
    IapwsSteamEnthalpyEvaluatorConfig,
)


class iapws_water_saturation_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS water phase saturation from (P, h), branching across regions."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: water saturation
        :rtype: float
        """
        P, h = state[0] * 0.1, state[1] / 18.015
        if P < Pmin:
            P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin:
            h = hmin

        region = _Bound_Ph(P, h)
        if region == 1:
            sw = 1
        elif region == 4:
            hw = _Region4(P, 0)["h"]
            hs = _Region4(P, 1)["h"]
            rhow = 1 / _Region4(P, 0)["v"]
            rhos = 1 / _Region4(P, 1)["v"]
            sw = rhos * (hs - h) / (h * (rhow - rhos) - (hw * rhow - hs * rhos))
        elif region == 2:
            sw = 0
        else:
            raise NotImplementedError(
                'Variables out of bound: p='
                + str(state[0])
                + ' bars, h='
                + str(state[1])
                + ' kJ/kmol, region='
                + str(region)
            )
        return sw


register_evaluator(
    "iapws_water_saturation",
    iapws_water_saturation_evaluator,
    IapwsWaterSaturationEvaluatorConfig,
)


class iapws_steam_saturation_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS steam phase saturation: ``1 - water_saturation``."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: steam saturation
        :rtype: float
        """
        water_saturation = iapws_water_saturation_evaluator()
        ss = 1 - water_saturation.evaluate(state)
        return ss


register_evaluator(
    "iapws_steam_saturation",
    iapws_steam_saturation_evaluator,
    IapwsSteamSaturationEvaluatorConfig,
)


class iapws_water_relperm_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS water relative permeability (linear in saturation)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: water relative permeability
        :rtype: float
        """
        water_saturation = iapws_water_saturation_evaluator()
        water_rp = water_saturation.evaluate(state) ** 1
        return water_rp


register_evaluator(
    "iapws_water_relperm",
    iapws_water_relperm_evaluator,
    IapwsWaterRelpermEvaluatorConfig,
)


class iapws_steam_relperm_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS steam relative permeability (linear in saturation)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: steam relative permeability
        :rtype: float
        """
        steam_saturation = iapws_steam_saturation_evaluator()
        steam_rp = steam_saturation.evaluate(state) ** 1
        return steam_rp


register_evaluator(
    "iapws_steam_relperm",
    iapws_steam_relperm_evaluator,
    IapwsSteamRelpermEvaluatorConfig,
)


class iapws_water_density_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS water density [kg/m3] from (P, h), branching across regions."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: water density [kg/m3]
        :rtype: float
        """
        P, h = state[0] * 0.1, state[1] / 18.015
        if P < Pmin:
            P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin:
            h = hmin

        region = _Bound_Ph(P, h)
        if region == 1:
            temperature = temperature_region1_evaluator()
            T = temperature.evaluate(state)
            water_density = 1 / _Region1(T, P)['v']
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                water_density = 1 / _Region4(P, 0)['v']
            else:
                raise NotImplementedError(
                    'Variables out of bound: p='
                    + str(state[0])
                    + ' bars, h='
                    + str(state[1])
                    + ' kJ/kmol, region='
                    + str(region)
                )
        elif region == 2:
            water_density = 0
        else:
            raise NotImplementedError(
                'Variables out of bound: p='
                + str(state[0])
                + ' bars, h='
                + str(state[1])
                + ' kJ/kmol, region='
                + str(region)
            )
        return water_density


register_evaluator(
    "iapws_water_density",
    iapws_water_density_evaluator,
    IapwsWaterDensityEvaluatorConfig,
)


class iapws_steam_density_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS steam density [kg/m3] from (P, h), branching across regions."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: steam density [kg/m3]
        :rtype: float
        """
        P, h = state[0] * 0.1, state[1] / 18.015
        if P < Pmin:
            P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin:
            h = hmin

        region = _Bound_Ph(P, h)
        if region == 1:
            steam_density = 0
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                steam_density = 1 / _Region4(P, 1)['v']
            else:
                raise NotImplementedError(
                    'Variables out of bound: p='
                    + str(state[0])
                    + ' bars, h='
                    + str(state[1])
                    + ' kJ/kmol, region='
                    + str(region)
                )
        elif region == 2:
            To = _Backward2_T_Ph(P, h)
            T = newton(lambda T: _Region2(T, P)["h"] - h, To)
            steam_density = 1 / _Region2(T, P)["v"]
        else:
            raise NotImplementedError(
                'Variables out of bound: p='
                + str(state[0])
                + ' bars, h='
                + str(state[1])
                + ' kJ/kmol, region='
                + str(region)
            )
        return steam_density


register_evaluator(
    "iapws_steam_density",
    iapws_steam_density_evaluator,
    IapwsSteamDensityEvaluatorConfig,
)


class iapws_water_viscosity_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS water viscosity [cP] from (P, h)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: water viscosity [cP]
        :rtype: float
        """
        temperature = iapws_temperature_evaluator().evaluate(state)
        density = iapws_water_density_evaluator().evaluate(state)
        return _Viscosity(density, temperature) * 1000


register_evaluator(
    "iapws_water_viscosity",
    iapws_water_viscosity_evaluator,
    IapwsWaterViscosityEvaluatorConfig,
)


class iapws_steam_viscosity_evaluator(
    property_evaluator_iface, EvaluatorBase, metaclass=_EvaluatorIfaceMeta
):
    """IAPWS steam viscosity [cP] from (P, h)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, state):
        """
        :param state: state vector ``[p (bar), h (kJ/kmol)]``
        :type state: list[float] | np.ndarray
        :return: steam viscosity [cP]
        :rtype: float
        """
        temperature = iapws_temperature_evaluator().evaluate(state)
        density = iapws_steam_density_evaluator().evaluate(state)
        return _Viscosity(density, temperature) * 1000


register_evaluator(
    "iapws_steam_viscosity",
    iapws_steam_viscosity_evaluator,
    IapwsSteamViscosityEvaluatorConfig,
)


# ---------------IAPWS based on Pressure-Temperature system, mainly for P-T super engine (Xiaoming Tian)---------------
class Density_iapws_water(EvaluatorBase):
    """IAPWS water density [kg/m3] from (P, T)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: water density [kg/m3]
        :rtype: float
        """
        P = pressure * 0.1  # MPa
        T = temperature  # K

        if P < Pmin:
            P = Pmin

        region = _Bound_TP(
            T, P
        )  # warning: with P-T system, this function can't return Region 4 (two phase region)

        if region == 1:
            water_density = 1 / _Region1(T, P)['v']
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                water_density = 1 / _Region4(P, 0)['v']
            else:
                raise NotImplementedError(
                    "water: Incoming out of bound of IAPWS Region 4 (two phase region)"
                )
        elif region == 2:
            water_density = 0
        else:
            raise NotImplementedError("water: Incoming out of bound of IAPWS Regions")
        return water_density


register_evaluator(
    "iapws_pt_density_water", Density_iapws_water, DensityIapwsWaterConfig
)


class Density_iapws_steam(EvaluatorBase):
    """IAPWS steam density [kg/m3] from (P, T)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: steam density [kg/m3]
        :rtype: float
        """
        P = pressure * 0.1  # MPa
        T = temperature  # K

        if P < Pmin:
            P = Pmin

        region = _Bound_TP(
            T, P
        )  # warning: with P-T system, this function can't return Region 4 (two phase region)

        if region == 1:
            steam_density = 0
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                steam_density = 1 / _Region4(P, 1)['v']
            else:
                raise NotImplementedError(
                    "steam: Incoming out of bound of IAPWS Region 4 (two phase region)"
                )
        elif region == 2:
            steam_density = 1 / _Region2(T, P)["v"]
        else:
            raise NotImplementedError("steam: Incoming out of bound of IAPWS Regions")
        return steam_density


register_evaluator(
    "iapws_pt_density_steam", Density_iapws_steam, DensityIapwsSteamConfig
)


class Viscosity_iapws_water(EvaluatorBase):
    """IAPWS water viscosity [cP] from (P, T)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: water viscosity [cP]
        :rtype: float
        """
        pressure * 0.1  # MPa
        T = temperature  # K

        den = Density_iapws_water().evaluate(pressure, temperature)
        return _Viscosity(den, T) * 1000


register_evaluator(
    "iapws_pt_viscosity_water", Viscosity_iapws_water, ViscosityIapwsWaterConfig
)


class Viscosity_iapws_steam(EvaluatorBase):
    """IAPWS steam viscosity [cP] from (P, T)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: steam viscosity [cP]
        :rtype: float
        """
        T = temperature  # K

        den = Density_iapws_steam().evaluate(pressure, temperature)
        return _Viscosity(den, T) * 1000


register_evaluator(
    "iapws_pt_viscosity_steam", Viscosity_iapws_steam, ViscosityIapwsSteamConfig
)


class Saturation_iapws_water(EvaluatorBase):
    """IAPWS water phase saturation from (P, T) (no two-phase information)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: water saturation
        :rtype: float
        """
        P = pressure * 0.1  # MPa
        T = temperature  # K

        if P < Pmin:
            P = Pmin

        region = _Bound_TP(
            T, P
        )  # warning: with P-T system, this function can't return Region 4 (two phase region)
        if region == 1:
            sw = 1
        elif region == 4:
            sw = 0
            # todo: it is hard to get vapor quality (or saturation) in P-T system
        elif region == 2:
            sw = 0
        else:
            raise NotImplementedError("Incoming out of bound of IAPWS Regions")
        return sw


register_evaluator(
    "iapws_pt_saturation_water", Saturation_iapws_water, SaturationIapwsWaterConfig
)


class Saturation_iapws_steam(EvaluatorBase):
    """IAPWS steam phase saturation: ``1 - water_saturation``."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: steam saturation
        :rtype: float
        """
        water_saturation = Saturation_iapws_water()
        ss = 1 - water_saturation.evaluate(pressure, temperature)
        return ss


register_evaluator(
    "iapws_pt_saturation_steam", Saturation_iapws_steam, SaturationIapwsSteamConfig
)


class Relperm_iapws_water(EvaluatorBase):
    """IAPWS water relative permeability (P-T system, linear in saturation)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: water relative permeability
        :rtype: float
        """

        water_saturation = Saturation_iapws_water()
        water_rp = water_saturation.evaluate(pressure, temperature) ** 1
        return water_rp


register_evaluator(
    "iapws_pt_relperm_water", Relperm_iapws_water, RelpermIapwsWaterConfig
)


class Relperm_iapws_steam(EvaluatorBase):
    """IAPWS steam relative permeability (P-T system, linear in saturation)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: steam relative permeability
        :rtype: float
        """
        steam_saturation = Saturation_iapws_steam()
        steam_rp = steam_saturation.evaluate(pressure, temperature) ** 1
        return steam_rp


register_evaluator(
    "iapws_pt_relperm_steam", Relperm_iapws_steam, RelpermIapwsSteamConfig
)


class Enthalpy_iapws_water(EvaluatorBase):
    """IAPWS water enthalpy [kJ/kmol] from (P, T), branching across regions."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: water enthalpy [kJ/kmol]
        :rtype: float
        """
        P = pressure * 0.1  # MPa
        T = temperature  # K

        if P < Pmin:
            P = Pmin

        region = _Bound_TP(
            T, P
        )  # warning: with P-T system, this function can't return Region 4 (two phase region)

        if region == 1:
            water_enth = _Region1(T, P)["h"]
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                water_enth = _Region4(P, 0)["h"]
            else:
                raise NotImplementedError(
                    "water: Incoming out of bound of IAPWS Region 4 (two phase region)"
                )
        elif region == 2:
            water_enth = 0
        else:
            print(region)
            raise NotImplementedError(
                "Variables out of bound: p="
                + str(pressure)
                + " bars, T="
                + str(temperature)
                + " K, region="
                + str(region)
            )
        return water_enth * 18.015


register_evaluator(
    "iapws_pt_enthalpy_water", Enthalpy_iapws_water, EnthalpyIapwsWaterConfig
)


class Enthalpy_iapws_steam(EvaluatorBase):
    """IAPWS steam enthalpy [kJ/kmol] from (P, T), branching across regions."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: steam enthalpy [kJ/kmol]
        :rtype: float
        """
        P = pressure * 0.1  # MPa
        T = temperature  # K

        if P < Pmin:
            P = Pmin

        region = _Bound_TP(
            T, P
        )  # warning: with P-T system, this function can't return Region 4 (two phase region)

        if region == 1:
            steam_enth = 0
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                steam_enth = _Region4(P, 1)["h"]
            else:
                raise NotImplementedError(
                    "Variables out of bound: p="
                    + str(pressure)
                    + " bars, T="
                    + str(temperature)
                    + " K, region="
                    + str(region)
                )
        elif region == 2:
            steam_enth = _Region2(T, P)["h"]
        else:
            raise NotImplementedError(
                "Variables out of bound: p="
                + str(pressure)
                + " bars, T="
                + str(temperature)
                + " K, region="
                + str(region)
            )
        return steam_enth * 18.015


register_evaluator(
    "iapws_pt_enthalpy_steam", Enthalpy_iapws_steam, EnthalpyIapwsSteamConfig
)


class Conductivity_iapws_water(EvaluatorBase):
    """IAPWS water thermal conductivity [kJ/m/day/K] from (P, T)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: water thermal conductivity [kJ/m/day/K]
        :rtype: float
        """
        P = pressure * 0.1  # MPa
        T = temperature  # K

        rho_w = Density_iapws_water().evaluate(P, T)
        k = _ThCond(rho_w, T)  # [W/(mK)]
        return k / 1000 * 3600 * 24  # kJ/m/day/K


register_evaluator(
    "iapws_pt_conductivity_water",
    Conductivity_iapws_water,
    ConductivityIapwsWaterConfig,
)


class Conductivity_iapws_steam(EvaluatorBase):
    """IAPWS steam thermal conductivity [kJ/m/day/K] from (P, T)."""

    def __init__(self):
        """No parameters."""
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        """
        :param pressure: pressure [bar]
        :type pressure: float
        :param temperature: temperature [K]
        :type temperature: float
        :return: steam thermal conductivity [kJ/m/day/K]
        :rtype: float
        """
        P = pressure * 0.1  # MPa
        T = temperature  # K

        rho_s = Density_iapws_steam().evaluate(P, T)
        k = _ThCond(rho_s, T)  # [W/(mK)]
        return k / 1000 * 86400  # kJ/m/day/K


register_evaluator(
    "iapws_pt_conductivity_steam",
    Conductivity_iapws_steam,
    ConductivityIapwsSteamConfig,
)
