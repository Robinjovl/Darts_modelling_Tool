import numpy as np

from model import ModelProperties
from phreeqc_dissolution.conversions import convert_composition, calculate_injection_stream, \
    get_mole_fractions, convert_rate, bar2atm
import warnings

from dartsflash.libflash import NegativeFlash, FlashParams, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS
from dartsflash.components import CompData

try:
    from phreeqpy.iphreeqc.phreeqc_com import IPhreeqc
except ImportError:
    from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc

import matplotlib
# matplotlib.use('pgf')
# matplotlib.rc('pgf', texsystem='pdflatex', preamble=r'\usepackage{color}')
from matplotlib import pyplot as plt
from matplotlib import rcParams
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.rc('legend',fontsize=16)

def correct_composition(composition, comp_min):
    import numpy as np
    mask = np.zeros(len(composition))
    for i in range(len(composition)):
        if composition[i] == 0:
            mask[i] = 1
    factor = np.count_nonzero(mask)
    composition = np.multiply(composition, 1 - factor * comp_min)
    composition += mask * comp_min
    return composition

def load_database(database, db_path):
    try:
        database.load_database(db_path)
    except Exception as e:
        warnings.warn(f"Failed to load '{db_path}': {e}.", Warning)

def interpret_results(database):
    results_array = np.array(database.get_selected_output_array()[2])

    volume_gas = results_array[2] / 1000  # liters to m3
    co2_gas_mole = results_array[3]
    h2o_gas_mole = results_array[4]
    total_mole_gas = 3 * (co2_gas_mole + h2o_gas_mole)

    # interpret aqueous phase
    hydrogen_mole_aq = results_array[5]
    oxygen_mole_aq = results_array[6]
    carbon_mole_aq = results_array[7]
    calcium_mole_aq = results_array[8]

    volume_aq = results_array[9] / 1000  # liters to m3
    total_mole_aq = (hydrogen_mole_aq + oxygen_mole_aq + carbon_mole_aq + calcium_mole_aq)  # mol
    rho_aq = total_mole_aq / volume_aq / 1000  # kmol/m3

    # molar fraction of elements in aqueous phase
    x = np.array([0,
                  calcium_mole_aq / total_mole_aq,
                  carbon_mole_aq / total_mole_aq,
                  oxygen_mole_aq / total_mole_aq,
                  hydrogen_mole_aq / total_mole_aq])

    # in gaseous phase
    if total_mole_gas > 1.e-8:
        rho_g = total_mole_gas / volume_gas / 1000  # kmol/m3
        y = np.array([0,
                      0,
                      co2_gas_mole / total_mole_gas,
                      (2 * co2_gas_mole + h2o_gas_mole) / total_mole_gas,
                      2 * h2o_gas_mole / total_mole_gas])
    else:
        rho_g = 0.0
        y = np.zeros(len(x))

    # molar densities
    rho_phases = {'aq': rho_aq, 'gas': rho_g}
    # molar fraction of gaseous phase in fluid
    nu_v = total_mole_gas / (total_mole_aq + total_mole_gas)

    # interpret kinetic parameters
    kin_state = {'SI': results_array[10],
                 'SR': results_array[11],
                 'Act(H+)': results_array[12],
                 'Act(CO2)': results_array[13],
                 'Act(H2O)': results_array[14]}
    species_molalities = results_array[15:]
    return nu_v, x, y, rho_phases, kin_state, volume_aq, species_molalities

def run_phreeqc(pressure, temperature, z_h2o_init):
    phreeqc_species = ["OH-", "H+", "H2O", "C(-4)", "CH4", "C(4)", "HCO3-", "CO2", "CO3-2", "CaHCO3+", "CaCO3",
                       "(CO2)2", "Ca+2", "CaOH+", "H(0)", "H2", "O(0)", "O2"]
    species_2_element_moles = np.array([2, 1, 3, 1, 5, 1, 5, 3, 4, 6, 5, 6, 1, 3, 1, 2, 1, 2])
    species_headings = " ".join([f'MOL("{sp}")' for sp in phreeqc_species])
    species_punch = " ".join([f'MOL("{sp}")' for sp in phreeqc_species])

    # Define primary fluid constituents
    fc_mask = np.array([False, True, True, True, True], dtype=bool)
    elements = np.array(['Solid', 'Ca', 'C', 'O', 'H'])
    fc_idx = {comp: i for i, comp in enumerate(elements[fc_mask])}
    molar_weight_h2o = 0.018016

    phreeqc = IPhreeqc()
    load_database(phreeqc, "phreeqc.dat")
    pitzer = IPhreeqc()
    load_database(pitzer, "pitzer.dat")
    # phreeqc.phreeqc.OutputFileOn = True
    # phreeqc.phreeqc.SelectedOutputFileOn = True

    phreeqc_template = f"""
    USER_PUNCH            
    -headings    H(mol)      O(mol)      C(mol)      Ca(mol)      Vol_aq   SI            SR            ACT("H+") ACT("CO2") ACT("H2O") {species_headings}
    10 PUNCH    TOTMOLE("H") TOTMOLE("O") TOTMOLE("C") TOTMOLE("Ca") SOLN_VOL SI("Calcite") SR("Calcite") ACT("H+") ACT("CO2") ACT("H2O") {species_punch}

    SELECTED_OUTPUT
    -selected_out    true
    -user_punch      true
    -reset           false
    -high_precision  true
    -gases           CO2(g) H2O(g)

    SOLUTION 1
    temp      {{temperature:.2f}}
    pressure  {{pressure:.4f}}
    pH        7 charge
    -water    {{water_mass:.10f}} # kg

    REACTION 1
    H         {{hydrogen:.10f}}
    O         {{oxygen:.10f}}
    C         {{carbon:.10f}}
    Ca        {{calcium:.10f}}
    1

    KNOBS
    -convergence_tolerance  1e-10

    GAS_PHASE 1
    pressure  {{pressure:.4f}}       
    temp      {{temperature:.2f}}  
    CO2(g)     0

    PRINT
        -species            true
        -totals             true
        -equilibrium_phases true
        -gas_phase          true

    END
    """
    E = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                  [1, 0, 1, 2, 3, 3, 3, 0, 1, 3, 0],
                  [2, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0]])
    min_z = 1e-11

    inj_stream_components = np.array([z_h2o_init, 0, 0, 1.0 - z_h2o_init, 0, 0, 0, 0, 0, 0, 0])
    inj_stream = convert_composition(inj_stream_components, E)
    inj_stream = correct_composition(inj_stream, min_z)

    # calculate amount of moles of each component in 1000 moles of mixture
    total_moles = 1000
    fluid_moles = total_moles * inj_stream[1:]

    # adjust oxygen and hydrogen moles for water formation
    init_h_moles, init_o_moles = fluid_moles[fc_idx['H']], fluid_moles[fc_idx['O']]
    if init_h_moles / 2 <= init_o_moles:
        water_mass = init_h_moles / 2 * molar_weight_h2o
        fluid_moles[fc_idx['H']] = 0
        fluid_moles[fc_idx['O']] = init_o_moles - init_h_moles / 2
    else:
        water_mass = init_o_moles * molar_weight_h2o
        fluid_moles[fc_idx['H']] = init_h_moles - 2 * init_o_moles
        fluid_moles[fc_idx['O']] = 0

    # Check if solvent (water) is enough
    ion_strength = np.sum(fluid_moles) / (water_mass + 1.e-8)
    if ion_strength > 12:
        print(f'ion_strength = {ion_strength}')
    # assert ion_strength < 7, "Not enough water to form a realistic brine"

    # Generate and execute PHREEQC input
    input_string = phreeqc_template.format(
        temperature=temperature - 273.15,
        pressure=bar2atm(pressure),
        water_mass=water_mass,
        hydrogen=fluid_moles[fc_idx['H']],
        oxygen=fluid_moles[fc_idx['O']],
        carbon=fluid_moles[fc_idx['C']],
        calcium=fluid_moles[fc_idx['Ca']]
    )

    try:
        phreeqc.run_string(input_string)
        nu_v, x, y, rho_phases, kin_state, fluid_volume, species_molalities = interpret_results(phreeqc)
    except Exception as e:
        warnings.warn(f"Failed to run PHREEQC: {e}", Warning)
        print(
            f"h20_mass={water_mass}, p={pressure}, Ca={fluid_moles[fc_idx['Ca']]}, C={fluid_moles[fc_idx['C']]}, O={fluid_moles[fc_idx['O']]}, H={fluid_moles[fc_idx['H']]}")
        pitzer.run_string(input_string)
        nu_v, x, y, rho_phases, kin_state, fluid_volume, species_molalities = interpret_results(pitzer)

    species_molar_fractions = species_molalities * water_mass * species_2_element_moles / total_moles
    species_molar_fractions_dict = {phreeqc_species[i]:species_molar_fractions[i] for i in range(species_molar_fractions.size)}

    return nu_v, species_molar_fractions_dict

def get_h2o_co2_composition(zO, zH, zC, min_z):
    z = zO + zH + zC
    zH /= z
    zO /= z
    zC /= z

    if zH / 2 <= zO:
        zH2O = zH / 2
        zO = zO - zH / 2
        zH = 0
    else:
        zH2O = zO
        zH = zH - 2 * zO
        zO = 0

    ret = np.array([zH2O, 0])
    ret /= np.sum(ret)
    ret[ret > 1 - min_z] = 1 - min_z
    ret[ret < min_z] = min_z
    return ret

def get_element_composition(z_h2o, z_co2):
    # C, O, H
    z_c = z_co2 / 3
    z_o = z_h2o / 3 + 2 * z_co2 / 3
    z_h = 2 * z_h2o / 3
    return np.array([z_c, z_o, z_h])

def run_darts_flash(pressure, temperature, z_h2o_init):
    min_z = 1e-11

    # input state
    components = ["H2O", "CO2"]
    zc = [z_h2o_init, 1. - z_h2o_init] # pure water [1 - min_z, min_z]

    # darts-flash
    comp_data = CompData(components, setprops=True)
    flash_params = FlashParams(comp_data)
    flash_params.add_eos("PR", CubicEoS(comp_data, CubicEoS.PR))
    flash_params.add_eos("AQ", AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                                                 AQEoS.CompType.solute: AQEoS.Ziabakhsh2012,
                                                 AQEoS.CompType.ion: AQEoS.Jager2003
                                                 }))
    flash_params.eos_order = ["PR", "AQ"]
    darts_flash = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])
    error_output = darts_flash.evaluate(pressure, temperature, zc)
    flash_results = darts_flash.get_flash_results()
    nu = np.array(flash_results.nu)
    x = np.array(flash_results.X).reshape(2, 2)

    return nu, x

def run_comparison():

    # against pressure
    pressures = np.arange(10, 100, 5)
    z_h2o_inits = np.array([0.8, 0.9, 0.99])
    temp = 273.15 + 50

    colors = ['b', 'r', 'g', 'm', 'y', 'orange', 'k']
    fig, ax = plt.subplots(ncols=1, figsize=(8, 6))
    ms = 5

    nu_v_phreeqc = np.zeros((z_h2o_inits.size, pressures.size))
    nu_v_darts = np.zeros((z_h2o_inits.size, pressures.size))
    for i in range(z_h2o_inits.size):
        for j in range(pressures.size):
            nu_v_p, _ = run_phreeqc(pressure=pressures[j], temperature=temp, z_h2o_init=z_h2o_inits[i])
            nu_v_d, x_darts = run_darts_flash(pressure=pressures[j], temperature=temp, z_h2o_init=z_h2o_inits[i])
            nu_v_phreeqc[i, j] = nu_v_p
            nu_v_darts[i, j] = nu_v_d[0]

        ax.plot(pressures, nu_v_phreeqc[i], color=colors[i], linestyle='-', marker='o', markersize=ms,
                markerfacecolor='none', label=f'PHREEQC: zH2O = {round(z_h2o_inits[i], 4)}')
        ax.plot(pressures, nu_v_darts[i], color=colors[i], linestyle='--', marker='x', markersize=ms,
                markerfacecolor='none', label=f'darts-flash: zH2O = {round(z_h2o_inits[i], 4)}')

    ax.set_xlabel('Pressure, bar', fontsize=16)
    ax.set_ylabel('Gas Phase Molar Fraction', fontsize=16)
    ax.legend(loc='upper right', prop={'size': 12}, framealpha=0.9)
    fig.tight_layout()
    fig.savefig('cmp_darts_flash_phreeqc_pressure.png', dpi=300)
    plt.close(fig)

    # against initial H2O composition
    pressures = np.array([10, 30, 50])
    z_h2o_inits = np.arange(0.7, 0.99, 0.02)
    temp = 273.15 + 50

    colors = ['b', 'r', 'g', 'm', 'y', 'orange', 'k']
    fig, ax = plt.subplots(ncols=1, figsize=(8, 6))
    ms = 5

    nu_v_phreeqc = np.zeros((pressures.size, z_h2o_inits.size))
    nu_v_darts = np.zeros((pressures.size, z_h2o_inits.size))
    for i in range(pressures.size):
        for j in range(z_h2o_inits.size):
            nu_v_p, _ = run_phreeqc(pressure=pressures[i], temperature=temp, z_h2o_init=z_h2o_inits[j])
            nu_v_d, x_darts = run_darts_flash(pressure=pressures[i], temperature=temp, z_h2o_init=z_h2o_inits[j])
            nu_v_phreeqc[i, j] = nu_v_p
            nu_v_darts[i, j] = nu_v_d[0]

        ax.plot(z_h2o_inits, nu_v_phreeqc[i], color=colors[i], linestyle='-', marker='o', markersize=ms,
                markerfacecolor='none', label=f'PHREEQC: p = {round(pressures[i], 4)} bar')
        ax.plot(z_h2o_inits, nu_v_darts[i], color=colors[i], linestyle='--', marker='x', markersize=ms,
                markerfacecolor='none', label=f'darts-flash: p = {round(pressures[i], 4)} bar')

    ax.set_xlabel('Initial zH2O', fontsize=16)
    ax.set_ylabel('Gas Phase Molar Fraction', fontsize=16)
    ax.legend(loc='upper right', prop={'size': 12}, framealpha=0.9)
    fig.tight_layout()
    fig.savefig('cmp_darts_flash_phreeqc_zH2O.png', dpi=300)
    plt.close(fig)


run_comparison()


