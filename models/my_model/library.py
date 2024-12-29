#%% Thermal properties of some materials for lateral heat transfer calculations
# Source: OLGA
"""
If the exact values of the properties of the following materials are not available, this file can be used.
This file contains the typical physical properties of some materials to define the heat transfer properties in thermal
computations. These values are typical values suggested by Schlumberger, and they may differ considerably from case to
case and product to product.

c: Heat capacity of the material in J/kg-C
K: Thermal conductivity of the material in W/m-C
rho: Material density in kg/m3
"""
mats_thermal_props = {
    "Bitumen Enamel":       {"c": 1300, "K": 0.7,  "rho": 1325},
    "Carbon Steel":         {"c": 470,  "K": 45,   "rho": 7850},
    "Concrete Coating HD":  {"c": 880,  "K": 2.7,  "rho": 3000},
    "Concrete Coating LD":  {"c": 880,  "K": 1.8,  "rho": 2250},
    "FBE Coating":          {"c": 1410, "K": 0.3,  "rho": 1300},   # FBE stands for Fusion Bond Epoxy
    "HYDRATE":              {"c": 4000, "K": 0.5,  "rho": 1000},
    "Poly Ethylene-foam":   {"c": 2300, "K": 0.04, "rho": 32  },
    "Poly Propylene":       {"c": 2200, "K": 0.4,  "rho": 960 },
    "Poly Propylene-foam":  {"c": 2000, "K": 0.17, "rho": 750 },
    "Rock":                 {"c": 2500, "K": 3,    "rho": 2100},
    "Stainless Steel":      {"c": 450,  "K": 20,   "rho": 7850},
    "Subsea sandy soil":    {"c": 1250, "K": 2.6,  "rho": 1750},
    "Subsea silt and clay": {"c": 2000, "K": 1.3,  "rho": 1350},
    "Syntactic Foam":       {"c": 2000, "K": 0.1,  "rho": 450 }
}

#%% Parachors of some components for computing interfacial tension (IFT)
# Source: The book "The Properties of Petroleum Fluids, 2nd edition (McCain)" or PVTi
components_parachors = {"H2"      :  34,     # McCain and PVTi
                        "N2"      :  41,     # McCain and PVTi
                        "H2O"     :  53.1,   # PVTi
                        "CO"      :  60,     # PVTi
                        "CO2"     :  78,     # McCain and PVTi
                        "H2S"     :  80,     # PVTi

                        # Hydrocarbon components
                        "C1"      :  77.0,   # McCain and PVTi
                        "C2"      : 108.0,   # McCain and PVTi
                        "C3"      : 150.3,   # McCain and PVTi
                        "i-C4"    : 181.5,   # McCain and PVTi
                        "n-C4"    : 189.9,   # McCain
                        "i-C5"    : 225.0,   # McCain and PVTi
                        "n-C5"    : 231.5,   # McCain and PVTi
                        "n-C6"    : 271.0,   # McCain and PVTi --> In PVTi it is named C6 with no prefix
                        "n-C7"    : 312.5,   # McCain and PVTi --> In PVTi it is named C7 with no prefix
                        "n-C8"    : 351.5,   # McCain and PVTi --> In PVTi it is named C8 with no prefix
                        "C9"      : 380,     # PVTi
                        "C10"     : 404.9,   # PVTi
                        "C11"     : 429.3,   # PVTi
                        "C12"     : 453.8,   # PVTi
                        "C13"     : 478.3,   # PVTi
                        "C14"     : 502.8,   # PVTi
                        "C15"     : 550.6,   # PVTi
                        "C16"     : 585.2,   # PVTi
                        "C17"     : 619.8,   # PVTi
                        "C18"     : 654.4,   # PVTi
                        "C19"     : 688.9,   # PVTi
                        "C20"     : 723.6,   # PVTi
                        "Benzene" : 237.47,  # PVTi
                        "Toluene" : 283.67   # PVTi
                        }

#%% Molecular weights of some components
components_molecular_weights = {"H2"  : 2.0158,   # Multiflash of OLGA
                                # "N2"  : 28.01352,   # PVTsim
                                "N2"  : 28.01348,   # Multiflash of OLGA
                                "H2O" : 18.0152,   # Multiflash of OLGA
                                "CO"  : 28.011,   # Multiflash of OLGA
                                "CO2" : 44.0098,   # Multiflash of OLGA and PVTsim
                                # "CO2" : 44.0095,   # NIST
                                # "H2S" : 34.08,   # PVTsim
                                "H2S" : 34.08088,   # Multiflash of OLGA

                                # Hydrocarbon components
                                "C1"  : 16.04288,   # PVTsim
                                "C2"  : 30.06982,   # PVTsim
                                "C3"  : 44.09676,   # PVTsim
                                "C4"  : 58.1237,   # PVTsim
                                "C5"  : 72.15064,   # PVTsim
                                "C6"  : 86.178,   # PVTsim
                                "C7"  : 96,   # PVTsim
                                "C8"  : 107,   # PVTsim
                                "C9"  : 121,   # PVTsim
                                "C10" : 134,   # PVTsim
                                "C11" : 147,   # PVTsim
                                "C12" : 161,   # PVTsim
                                "C13" : 175,   # PVTsim
                                "C14" : 190,   # PVTsim
                                "C15" : 206,   # PVTsim
                                "C16" : 222,   # PVTsim
                                "C17" : 237,   # PVTsim
                                "C18" : 251,   # PVTsim
                                "C19" : 263,   # PVTsim
                                "C20" : 275,   # PVTsim
                                "Benzene" : 78.11399841,   # Multiflash of OLGA
                                "Toluene" : 92.14089966   # Multiflash of OLGA
                                }
