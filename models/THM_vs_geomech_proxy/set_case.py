from cases.case_1 import input_data_case_1, input_data_case_2, input_data_case_3, input_data_case_4
from cases.case_5 import input_data_case_5
from cases.no_damage_zone import input_data_no_damage_zone
from cases.no_damage_zone_heter_mech_prop import input_data_no_damage_zone_heter_mech_prop

from cases.base import input_data_struct_like
from darts.engines import well_control_iface
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_input_data(
    case: str,
    model_folder=None,
    physics_type="single_phase_thermal",
    wells_type="doublet",
):

    case_ = os.path.basename(case) # without meshes/ part
    match case_:
        case "case_1": # uset heterogeneous poro and perm by interpolation
            input_data = input_data_case_1(physics_type)
        case "case_2": # case_1 geometry/mesh but rock props assigned per tag
            input_data = input_data_case_2(physics_type)
        case "case_3": # case_2 but heterogeneour rock thermal props
            input_data = input_data_case_3(physics_type)
        case "case_4": # case_3 but heterogeneous geomechanical props (E, nu, biot, th_expn) per tag
            input_data = input_data_case_4(physics_type)
        case "case_5":
            input_data = input_data_case_5(physics_type)
        case "no_damage_zone":
            input_data = input_data_no_damage_zone(physics_type)
        case "no_damage_zone_heter_mech_prop":
            input_data = input_data_no_damage_zone_heter_mech_prop(physics_type)
        case _:  # default (struct-like meshes)
            model_folder = os.path.basename(model_folder)
            os.makedirs(os.path.join(BASE_DIR, "meshes", model_folder), exist_ok=True)
            input_data = input_data_struct_like(
                model_folder=model_folder,
                physics_type=physics_type,
                wells_type=wells_type,
            )
            
            if "zero_rate" in case:
                # zero-rate check: force rate control at 0. Isothermal cases default to BHP
                # control in _set_wells (target = p_cell +/- delta_p), which ignores well_rate;
                # switch to rate control so the well is genuinely shut in. Thermal already uses
                # rate control, so this is consistent for both.
                input_data.other.wctrl_type = well_control_iface.MASS_RATE
                input_data.other.delta_p = None
                input_data.other.well_rate = 0.

    return input_data
