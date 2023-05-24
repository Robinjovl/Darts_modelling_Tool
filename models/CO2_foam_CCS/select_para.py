import numpy as np

def select_parameters(components):
    if components == ["CO2", "H2O"]:
        """ 2 component system """
        Pc = np.array([73.9, 220.65])  # IN BAR                CO2, H2O
        Tc = np.array([304.25, 647.096])  # IN KELVIN             CO2, H2O
        omega = np.array([0.239, 0.334])  # Accentric factor [-]  CO2, H2O
        MWi = np.array([44.01, 18.015])  # Molar weight in g/mol CO2, H2O

    elif components == ["C1", "H2O"]:
        """ 2 component system """
        Pc = np.array([45.99, 220.65])  # IN BAR                CH4, H2O
        Tc = np.array([190.56, 647.096])  # IN KELVIN             CH4, H2O
        omega = np.array([0.011, 0.334])  # Accentric factor [-]  CH4, H2O
        MWi = np.array([16.04, 18.015])  # Molar weight in g/mol CH4, H2O

    elif components == ["N2", "H2O"]:
        """ 2 component system """
        Pc = np.array([33.9, 220.65])  # IN BAR                N2, H2O
        Tc = np.array([126.2, 647.096])  # IN KELVIN             N2, H2O
        omega = np.array([0.039, 0.334])  # Accentric factor [-]  N2, H2O
        MWi = np.array([28.013, 18.015])  # Molar weight in g/mol N2, H2O

    elif components == ["H2S", "H2O"]:
        """ 2 component system """
        Pc = np.array([89.4, 220.65])  # IN BAR                H2S, H2O
        Tc = np.array([373.2, 647.096])  # IN KELVIN             H2S, H2O
        omega = np.array([0.081, 0.334])  # Accentric factor [-]  H2S, H2O
        MWi = np.array([34.081, 18.015])  # Molar weight in g/mol H2S, H2O

    elif components == ["CO2", "N2", "H2O"]:
        """ 3 component system """
        Pc = np.array([73.9, 33.9, 220.65])  # IN BAR                CO2, N2, H2O
        Tc = np.array([304.25, 126.2, 647.096])  # IN KELVIN             CO2, N2, H2O
        omega = np.array([0.239, 0.039, 0.334])  # Accentric factor [-]  CO2, N2, H2O
        MWi = np.array([44.01, 28.013, 18.015])  # Molar weight in g/mol CO2, N2, H2O

    elif components == ["CO2", "C1", "H2O"]:
        """ 3 component system """
        Pc = np.array([73.9, 45.99, 220.65])  # IN BAR                CO2, CH4, H2O
        Tc = np.array([304.25, 190.56, 647.096])  # IN KELVIN             CO2, CH4, H2O
        omega = np.array([0.239, 0.011, 0.334])  # Accentric factor [-]  CO2, CH4, H2O
        MWi = np.array([44.01, 16.04, 18.015])  # Molar weight in g/mol CO2, CH4, H2O

    elif components == ["CO2", "C1", "H2S", "H2O"]:
        """ 4 component system """
        Pc = np.array([73.9, 45.99, 89.4, 220.65])  # IN BAR                CO2, CH4, H2S, H2O
        Tc = np.array([304.25, 190.56, 373.2, 647.096])  # IN KELVIN             CO2, CH4, H2S, H2O
        omega = np.array([0.239, 0.011, 0.081, 0.334])  # Accentric factor [-]  CO2, CH4, H2S, H2O
        MWi = np.array([44.01, 16.04, 34.081, 18.015])  # Molar weight in g/mol CO2, CH4, H2S, H2O


    else:
        """ [CO2, N2, C1, H2O] """
        """ 4 component system """
        Pc = np.array([73.9, 33.9, 45.99, 220.65])      # IN BAR                CO2, N2, CH4, H2O
        Tc = np.array([304.25, 126.2, 190.56, 647.096]) # IN KELVIN             CO2, N2, CH4, H2O
        omega = np.array([0.239, 0.039, 0.011, 0.334])  # Accentric factor [-]  CO2, N2, CH4, H2O
        MWi = np.array([44.01, 28.013, 16.04, 18.015])  # Molar weight in g/mol CO2, N2, CH4, H2O


    return omega, MWi, Pc, Tc


def select_parameter_activity(components):
    if components == ["N2", "H2O"]:
        # second order interaction parameters:  #N2-Na
        order2 = np.array([[-2.0939363, 3.1445269e-3, 3.9139160e2, -2.9973977e-7, 0, -1.5918098e-5, 0, 0, 0, 0]])
        # third order interaction parameters:   #N2-Na-Cl
        order3 = np.array([[-6.3981858e-3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    elif components == ["C1", "H2O"]:
        # second order interaction parameters:  #CH4-Na
        order2 = np.array([[-5.7066455e-1, 7.2997588e-4, 1.5176903e2, 3.1927112e-5, 0, -1.6426510e-5, 0, 0, 0, 0]])
        # third order interaction parameters:   #CH4-Na-Cl
        order3 = np.array([[-2.9990084e-3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    elif components == ["H2S", "H2O"]:
        # second order interaction parameters:  #H2S-Na
        order2 = np.array([[1.03658689, -1.1784797e-3, -1.7754826e2, -4.53132285e-4, 0, 0, 0, 0, 0, 0.47751650e2]])
        # third order interaction parameters:   #H2S-Na-Cl
        order3 = np.array([[-0.010274152, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    elif components == ["CO2", "C1", "H2O"]:
        # second order interaction parameters:  #CO2-Na,#N2-Na,#CH4-Na
        order2 = np.array([[-0.0652869, 1.6790636e-4, 40.838951, 0, 0, -3.9266518e-2, 0, 2.1157167e-2, 6.5486487e-6, 0], \
                           [-5.7066455e-1, 7.2997588e-4, 1.52e2, 3.1927112e-5, 0, -1.6426510e-5, 0, 0, 0, 0]])
        # third order interaction parameters:   #CO2-Na-Cl, #N2-Na-Cl, #CH4-Na-Cl
        order3 = np.array([[-1.144624e-2, 2.8274958e-5, 0, 0, 0, 1.3980876e-2, 0, -1.4349005e-2, 0, 0], \
                           [-2.9990084e-3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    elif components == ["CO2", "C1", "H2S", "H2O"]:
        # second order interaction parameters:  #CO2-Na,#N2-Na,#CH4-Na
        order2 = np.array([[-0.0652869, 1.6790636e-4, 40.838951, 0, 0, -3.9266518e-2, 0, 2.1157167e-2, 6.5486487e-6, 0], \
                           [-5.7066455e-1, 7.2997588e-4, 1.52e2, 3.1927112e-5, 0, -1.6426510e-5, 0, 0, 0, 0], \
                           [1.03658689, -1.1784797e-3, -1.7754826e2, -4.53132285e-4, 0, 0, 0, 0, 0, 0.47751650e2]])
        # third order interaction parameters:   #CO2-Na-Cl, #N2-Na-Cl, #CH4-Na-Cl
        order3 = np.array([[-1.144624e-2, 2.8274958e-5, 0, 0, 0, 1.3980876e-2, 0, -1.4349005e-2, 0, 0], \
                           [-2.9990084e-3, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                           [-0.010274152, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    else:
        """ Valid for [Co2], [CO2, N2] and for [CO2, N2, CH4] """
        # second order interaction parameters:  #CO2-Na,#N2-Na,#CH4-Na
        order2 = np.array([[-0.0652869, 1.6790636e-4, 40.838951, 0, 0, -3.9266518e-2, 0, 2.1157167e-2, 6.5486487e-6, 0], \
                           [-2.0939363, 3.1445269e-3, 3.91e2, -2.9973977e-7, 0, -1.5918098e-5, 0, 0, 0, 0], \
                           [-5.7066455e-1, 7.2997588e-4, 1.52e2, 3.1927112e-5, 0, -1.6426510e-5, 0, 0, 0, 0]])
        # third order interaction parameters:   #CO2-Na-Cl, #N2-Na-Cl, #CH4-Na-Cl
        order3 = np.array([[-1.144624e-2, 2.8274958e-5, 0, 0, 0, 1.3980876e-2, 0, -1.4349005e-2, 0, 0], \
                           [-6.3981858e-3, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                           [-2.9990084e-3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    return (order2, order3)


def select_parameter_Henry(components):
    # Henry's parameters  # Ziabakhsh-ganji & Kooi
    if components == ["CO2", "H2O"]:
        mu = np.array([-0.114535])
        tau = np.array([-5.279063])
        beta = np.array([6.187967])

    elif components == ["N2", "H2O"]:
        mu = np.array([-0.008194])
        tau = np.array([-5.175337])
        beta = np.array([6.906469])

    elif components == ["C1", "H2O"]:
        mu = np.array([-0.092248])
        tau = np.array([-5.779280])
        beta = np.array([7.262730])

    elif components == ["H2S", "H2O"]:
        # Henry's parameters # [H2S] # Ziabakhsh-ganji & Kooi
        mu = np.array([0.77357854])
        tau = np.array([0.27049433])
        beta = np.array([0.27543436])

    elif components == ["CO2", "N2", "H2O"]:
        """ 3 component system """
        mu = np.array([-0.114535, -0.008194])  # [CO2, N2]
        tau = np.array([-5.279063, -5.175337])  # [CO2, N2]
        beta = np.array([6.187967, 6.906469])  # [CO2, N2]

    elif components == ["CO2", "C1", "H2O"]:
        """ 3 component system """
        mu = np.array([-0.114535, -0.092248])  # [CO2, CH4]
        tau = np.array([-5.279063, -5.779280])  # [CO2, CH4]
        beta = np.array([6.187967, 7.262730])  # [CO2, CH4]

    elif components == ["CO2", "C1", "H2S", "H2O"]:
        mu = np.array([-0.114535, -0.092248, 0.77357854])  # [CO2, CH4]
        tau = np.array([-5.279063, -5.779280, 0.27049433])  # [CO2, CH4]
        beta = np.array([6.187967, 7.262730, 0.27543436])  # [CO2, CH4]

    else:
        """ [CO2, N2, C1, H2O] """
        mu = np.array([-0.114535, -0.008194, -0.092248])  # [CO2, N2, CH4]
        tau = np.array([-5.279063, -5.175337, -5.779280])  # [CO2, N2, CH4]
        beta = np.array([6.187967, 6.906469, 7.262730])  # [CO2, N2, CH4]

    return (mu, tau, beta)

def select_comp_mass(comp):
    Mw =0
    if comp == ["CO2"]:
        Mw = 44.01
    elif comp == ["N2"]:
        Mw = 28.013
    elif comp == ["C1"]:
        Mw = 16.04
    elif comp == ["H2S"]:
        Mw = 34.081
    elif comp == ["H2O"]:
        Mw = 18.015

    return Mw

def props(component, property):
    properties = [["C1", "CO2", "H2O", "N2", "H2S", "C2H6", "SO2", "NaCl"],  # component
                  [190.58, 304.10, 647.14, 126.20, 373.53, 305.32, "", ""],  # T_c [K]
                  [46.04, 73.75, 220.50, 34.00, 89.63, 48.72, "", ""],  # p_c [bar]
                  [0.012, 0.239, 0.328, 0.0377, 0.0942, 0.0995, "", ""],  # acentric factor [-]
                  [16.043, 44.01, 18.015, 28.013, 34.076, 30.07, "", 58.44],  # molecular mass [g/mol]
                  [0.286, 0.274, 0.3074, 0.289, "", 0.279, "", ""]]  # critical compressibility factors

    prop = ["Tc", "Pc", "ac", "Mw", "Zc"]
    index1 = prop.index(property) + 1
    index2 = properties[0][:].index(component)
    c = properties[index1][index2]

    return c