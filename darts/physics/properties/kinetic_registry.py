"""
Kinetic reactions registry for supported minerals and their kinetic mechanisms.

Entries:
- k: Arrhenius pre-exponential factor [mol/m2/s]
- Ea: Activation energy [J/mol]
- T_ref_K: Reference temperature [K]
- n: reaction order w.r.t. mechanism activity
- p, q: chemical affinity parameters in (1 - SR^p)^q term
"""

KINETIC_REGISTRY = {
    'CaCO3': {
        'acidic': {
            'k': 10 ** (-0.3),
            'Ea': 14400,
            'n': 1,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'neutral': {
            'k': 10 ** (-5.81),
            'Ea': 23500,
            'n': 0,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'carbonate': {
            'k': 10 ** (-3.48),
            'Ea': 35400,
            'n': 1,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
    },
    'CaMg(CO3)2': {
        'acidic': {
            'k': 10 ** (-3.19),
            'Ea': 36100,
            'n': 0.5,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'neutral': {
            'k': 10 ** (-7.53),
            'Ea': 52200,
            'n': 0,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'carbonate': {
            'k': 10 ** (-5.11),
            'Ea': 34800,
            'n': 0.5,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
    },
    'MgCO3': {
        'acidic': {
            'k': 10 ** (-6.38),
            'Ea': 14400,
            'n': 1,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'neutral': {
            'k': 10 ** (-9.34),
            'Ea': 23500,
            'n': 0,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
        'carbonate': {
            'k': 10 ** (-5.22),
            'Ea': 62800,
            'n': 1,
            'p': 1,
            'q': 1,
            'T_ref_K': 273.15 + 25,
        },
    },
}

__all__ = ["KINETIC_REGISTRY"]
