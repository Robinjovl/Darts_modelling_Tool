import numpy as np

""" Prefixes """
def pico():
    return 1e-12
def nano():
    return 1e-9
def micro():
    return 1e-6
def milli():
    return 1e-3
def centi():
    return 1e-2
def deci():
    return 1e-1

def kilo():
    return 1e+3
def mega():
    return 1e+6
def giga():
    return 1e+9


""" Length units """
def meter():
    return 1
def inch():
    return 2.54 * centi() * meter()
def ft():
    return 0.3048 * meter()


""" Time units """
def second():
    return 1
def minute():
    return 60 * second()
def hour():
    return 60 * minute()
def day():
    return 24 * hour()
def year():
    return 365.2425 * day()


""" Mass units """
def kg():
    return 1
def gram():
    return 1e-3 * kg()
def lbm():
    return 453.59237 * gram()
def tonne():
    return 1000 * kg()


""" Unit of amount of substance """
def mol():
    return 1


""" Volume units """
# US liquid gallon
def gallon():
    return 231 * inch() ** 3
def liter():
    return 1e-3 * meter() ** 3

def barrel():
    return 42 * gallon()


""" Force units """
def Newton():
    return 1
def dyne():
    return 1e-5 * Newton()
def lbf():
    g = 9.80665 * meter() / (second() ** 2)
    return g * lbm()


""" Pressure units """
def Pascal():
    return 1
def psi():
    # Gives pressure in Pascal
    return lbf() / inch() ** 2   # 6894.76 Pascal
def bar():
    # Gives pressure in Pascal
    return 1e+5 * Pascal()
def atm():
    # Gives pressure in Pascal
    return 101325 * Pascal()


""" Energy units """
def Joule():
    return 1 * Newton() * meter()
def BTU():
    # Based on https://en.wikipedia.org/wiki/British_thermal_unit, it has a value around 1055 Joules
    return 1054.3503 * Joule()


""" Viscosity units """
def Poise():
    return 0.1 * Pascal() * second()


""" Temperature units """
def Kelvin():
    return 1
def Rankine():
    # Temperature of one Rankine (in units of Kelvin)
    return 5 / 9
def Fahrenheit():
    # Temperature of one Fahrenheit (in units of Kelvin)
    return 5 / 9


""" Power units """
def Watt():
    return Joule() / second()


""" Angle units """
def radian():
    return 1
def degree():
    # Gives angle in radian
    return np.pi / 180


def Darcy():
    # Gives permeability in m2
    miu = centi() * Poise()
    delta_p = atm() / (centi() * meter())
    A = (centi() * meter()) ** 2
    rate = (centi() * meter()) ** 3 / second()
    return rate * miu / (A * delta_p)


def convertTo(value, unit_name):
    # value must be in a SI unit
    # unit_name is the name of the function of the unit with ()
    return value / unit_name
