from darts.engines import property_evaluator_iface
from iapws.iapws97 import _Region1, _Region2, _Region4, _Backward1_T_Ph, _Backward2_T_Ph, _Bound_Ph, _Bound_TP, _TSat_P, Pmin, _ThCond
from iapws._iapws import _D2O_Viscosity, _Viscosity
from scipy.optimize import newton


class water_density_property_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        #sat_steam_enthalpy = saturated_steam_enthalpy_evaluator()
        #sat_steam_enth = sat_steam_enthalpy.evaluate(state)
        #sat_water_enthalpy = saturated_water_enthalpy_evaluator()
        #sat_water_enth = sat_water_enthalpy.evaluate(state)
        #temperature = temperature_evaluator(sat_water_enthalpy, sat_steam_enthalpy)
        #temp = temperature.evaluate(state)
        temperature = temperature_region1_evaluator()
        temp = temperature.evaluate(state)
        water_density = 1 / _Region1(temp, float(state[0])*0.1)['v']
        return water_density / 18.015

class temperature_region1_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        return (_Backward1_T_Ph(float(state[0])*0.1, state[1] / 18.015))

class iapws_enthalpy_region1_evaluator(property_evaluator_iface):
    def __init__(self, temp):
        #super().__init__()
        self.temperature = temp
    def evaluate(self, state):
        return _Region1(self.temperature, float(state[0])*0.1)['h'] * 18.015        #kJ/kmol

class iapws_viscosity_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        temperature = temperature_region1_evaluator().evaluate(state)
        density = water_density_property_evaluator().evaluate(state)
        return (_Viscosity(density, temperature)*1000)



#====================================== Properties for Region 1 and 4 ============================================ 
class iapws_total_enthalpy_evalutor(property_evaluator_iface):
    def __init__(self, ):
        super().__init__()
    def evaluate(self, state, temperature):
        P = state[0]*0.1
        region = _Bound_TP(temperature, P)
        if (region == 1):
            h = _Region1(temperature, P)["h"] * 18.015
        elif (region == 4):
            Steam_sat = iapws_steam_saturation_evaluator().evaluate(state)
            rho_steam = iapws_steam_density_evaluator().evaluate(state) / 18.015
            rho_water = iapws_water_density_evaluator().evaluate(state) / 18.015
            x = Steam_sat * rho_steam / (Steam_sat * rho_steam + (1-Steam_sat) * rho_water)
            h = _Region4(P, x)["h"] * 18.015
        elif (region == 2):
            h = _Region2(temperature, P)["h"] * 18.015
        else:
            raise NotImplementedError('Variables out of bound: p=' + str(P) + ' region=' + str(region))
        return h


class iapws_temperature_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin

        region = _Bound_Ph(P, h)
        if (region == 1):
            T = _Backward1_T_Ph(P, h)
        elif (region == 4):
            T = _TSat_P(P)
        elif (region == 2):
            T = _Backward2_T_Ph(P, h)
        else:
            raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        return T

class iapws_water_enthalpy_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin
        region = _Bound_Ph(P, h)
        if (region == 1):
            water_enth =  h
        elif (region == 4):
            T = _TSat_P(P)
            if T <= 623.15:
               water_enth = _Region4(P, 0)["h"]
            else:
               raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        elif (region == 2):
            water_enth = 0
        else:
            print(region)
            raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        return water_enth * 18.015


class iapws_steam_enthalpy_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin

        region = _Bound_Ph(P, h)
        if (region == 1):
            steam_enth =  0
        elif (region == 4):
            T = _TSat_P(P)
            if T <= 623.15:
               steam_enth = _Region4(P, 1)["h"]
            else:
               raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        elif (region == 2):
            To = _Backward2_T_Ph(P, h)
            T = newton(lambda T: _Region2(T, P)["h"]-h, To)
            steam_enth = _Region2(T, P)["h"]
        else:
            raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        return steam_enth * 18.015


class iapws_water_saturation_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin

        region = _Bound_Ph(P, h)
        if (region == 1):
            sw = 1
        elif (region == 4):
            hw = _Region4(P, 0)["h"]
            hs = _Region4(P, 1)["h"]
            rhow = 1 / _Region4(P, 0)["v"]
            rhos = 1 / _Region4(P, 1)["v"]
            sw = rhos * (hs - h) / (h * (rhow - rhos) - (hw * rhow - hs * rhos))
        elif (region == 2):
            sw = 0
        else:
             raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        return sw

class iapws_steam_saturation_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        water_saturation = iapws_water_saturation_evaluator()
        ss = 1 - water_saturation.evaluate(state)
        return ss

class iapws_water_relperm_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        water_saturation = iapws_water_saturation_evaluator()
        water_rp = water_saturation.evaluate(state)**1
        return water_rp

class iapws_steam_relperm_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        steam_saturation = iapws_steam_saturation_evaluator()
        steam_rp = steam_saturation.evaluate(state)**1
        return steam_rp


class iapws_water_density_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin

        region = _Bound_Ph(P, h)
        if (region == 1):
            temperature = temperature_region1_evaluator()
            T = temperature.evaluate(state)
            water_density = 1 / _Region1(T, P)['v']
        elif (region == 4):
            T = _TSat_P(P)
            if (T <= 623.15):
               water_density = 1 / _Region4(P, 0)['v']
            else:
               raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        elif (region == 2):
            water_density = 0
        else:
               raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        return water_density


class iapws_steam_density_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        P, h = state[0]*0.1, state[1]/18.015
        if P < Pmin :
           P = Pmin
        hmin = _Region1(273.15, P)["h"]
        if h < hmin :
           h = hmin

        region = _Bound_Ph(P, h)
        if (region == 1):
            steam_density = 0
        elif (region == 4):
            T = _TSat_P(P)
            if T <= 623.15:
               steam_density = 1 / _Region4(P, 1)['v']
            else:
               raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        elif (region == 2):
            To = _Backward2_T_Ph(P, h)
            T = newton(lambda T: _Region2(T, P)["h"]-h, To)
            steam_density = 1 / _Region2(T, P)["v"]
        else:
               raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        return steam_density


class iapws_water_viscosity_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        temperature = iapws_temperature_evaluator().evaluate(state)
        density = iapws_water_density_evaluator().evaluate(state)
        return (_Viscosity(density, temperature)*1000)


class iapws_steam_viscosity_evaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()
    def evaluate(self, state):
        temperature = iapws_temperature_evaluator().evaluate(state)
        density = iapws_steam_density_evaluator().evaluate(state)
        return (_Viscosity(density, temperature)*1000)




# ---------------IAPWS based on Pressure-Temperature system, mainly for P-T super engine (Xiaoming Tian)---------------
class Density_iapws_water():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for water density
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: water density, [kg/m3]
        '''
        P = pressure * 0.1  # MPa
        T = temperature     # K

        if P < Pmin:
           P = Pmin

        region = _Bound_TP(T, P)  # warning: with P-T system, this function can't return Region 4 (two phase region)

        if region == 1:
            water_density = 1 / _Region1(T, P)['v']
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                water_density = 1 / _Region4(P, 0)['v']
            else:
                raise NotImplementedError("water: Incoming out of bound of IAPWS Region 4 (two phase region)")
        elif region == 2:
            water_density = 0
        else:
            raise NotImplementedError("water: Incoming out of bound of IAPWS Regions")
        return water_density


class Density_iapws_steam():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for steam density
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: steam density, [kg/m3]
        '''
        P = pressure * 0.1  # MPa
        T = temperature     # K

        if P < Pmin:
           P = Pmin

        region = _Bound_TP(T, P)  # warning: with P-T system, this function can't return Region 4 (two phase region)

        if region == 1:
            steam_density = 0
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                steam_density = 1 / _Region4(P, 1)['v']
            else:
                raise NotImplementedError("steam: Incoming out of bound of IAPWS Region 4 (two phase region)")
        elif region == 2:
            # To = _Backward2_T_Ph(P, h)
            # T = newton(lambda T: _Region2(T, P)["h"]-h, To)
            steam_density = 1 / _Region2(T, P)["v"]
        else:
            raise NotImplementedError("steam: Incoming out of bound of IAPWS Regions")
        return steam_density


class Viscosity_iapws_water():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for water viscosity
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: water viscosity, [cP]
        '''
        P = pressure * 0.1  # MPa
        T = temperature     # K

        den = Density_iapws_water().evaluate(pressure, temperature)
        return _Viscosity(den, T)*1000


class Viscosity_iapws_steam():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for steam viscosity
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: steam viscosity, [cP]
        '''
        P = pressure * 0.1  # MPa
        T = temperature     # K

        den = Density_iapws_steam().evaluate(pressure, temperature)
        return _Viscosity(den, T)*1000


class Saturation_iapws_water():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for water saturation
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: water saturation
        '''
        P = pressure * 0.1  # MPa
        T = temperature     # K

        if P < Pmin:
           P = Pmin

        region = _Bound_TP(T, P)  # warning: with P-T system, this function can't return Region 4 (two phase region)
        if region == 1:
            sw = 1
        elif region == 4:
            sw = 0
            # todo: it is hard to get vapor quality (or saturation) in P-T system
            # hw = _Region4(P, 0)["h"]
            # hs = _Region4(P, 1)["h"]
            # rhow = 1 / _Region4(P, 0)["v"]
            # rhos = 1 / _Region4(P, 1)["v"]
            # sw = rhos * (hs - h) / (h * (rhow - rhos) - (hw * rhow - hs * rhos))
        elif region == 2:
            sw = 0
        else:
            raise NotImplementedError("Incoming out of bound of IAPWS Regions")
        return sw

class Saturation_iapws_steam():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for steam saturation
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: steam saturation
        '''
        P = pressure * 0.1  # MPa
        T = temperature  # K

        water_saturation = Saturation_iapws_water()
        ss = 1 - water_saturation.evaluate(pressure, temperature)
        return ss


class Relperm_iapws_water():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for water relative permeability
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: water relative permeability
        '''
        P = pressure * 0.1  # MPa
        T = temperature  # K

        water_saturation = Saturation_iapws_water()
        water_rp = water_saturation.evaluate(pressure, temperature)**1
        return water_rp

class Relperm_iapws_steam():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for steam relative permeability
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: steam relative permeability
        '''
        P = pressure * 0.1  # MPa
        T = temperature  # K

        steam_saturation = Saturation_iapws_steam()
        steam_rp = steam_saturation.evaluate(pressure, temperature)**1
        return steam_rp


class Enthalpy_iapws_water():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for water enthalpy
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: water enthalpy
        '''
        P = pressure * 0.1  # MPa
        T = temperature     # K

        if P < Pmin:
           P = Pmin

        region = _Bound_TP(T, P)  # warning: with P-T system, this function can't return Region 4 (two phase region)

        if region == 1:
            water_enth = _Region1(T, P)["h"]
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                water_enth = _Region4(P, 0)["h"]
            else:
                raise NotImplementedError("water: Incoming out of bound of IAPWS Region 4 (two phase region)")
        elif region == 2:
            water_enth = 0
        else:
            print(region)
            raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        return water_enth * 18.015

class Enthalpy_iapws_steam():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for steam enthalpy
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: steam enthalpy
        '''
        P = pressure * 0.1  # MPa
        T = temperature     # K

        if P < Pmin:
           P = Pmin

        region = _Bound_TP(T, P)  # warning: with P-T system, this function can't return Region 4 (two phase region)

        if region == 1:
            steam_enth =  0
        elif region == 4:
            T = _TSat_P(P)
            if T <= 623.15:
                steam_enth = _Region4(P, 1)["h"]
            else:
                raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        elif region == 2:
            # To = _Backward2_T_Ph(P, h)
            # T = newton(lambda T: _Region2(T, P)["h"]-h, To)
            steam_enth = _Region2(T, P)["h"]
        else:
            raise NotImplementedError('Variables out of bound: p=' + str(state[0]) + ' bars, h=' + str(state[1]) + ' kJ/kmol, region=' + str(region))
        return steam_enth * 18.015


class Conductivity_iapws_water():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for water conductivity
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: water conductivity
        '''
        P = pressure * 0.1  # MPa
        T = temperature     # K

        rho_w = Density_iapws_water().evaluate(P, T)
        k = _ThCond(rho_w, T)  # [W/(mK)]
        return k / 1000 * 3600 * 24  # kJ/m/day/K


class Conductivity_iapws_steam():
    def __init__(self):
        super().__init__()

    def evaluate(self, pressure: float, temperature: float) -> float:
        '''
        evaluation function for steam conductivity
        :param pressure: state pressure, [bars]
        :param temperature: state temperature, [K]
        :return: steam conductivity
        '''
        P = pressure * 0.1  # MPa
        T = temperature     # K

        rho_s = Density_iapws_steam().evaluate(P, T)
        k = _ThCond(rho_s, T)  # [W/(mK)]
        return k / 1000 * 86400  # kJ/m/day/K

