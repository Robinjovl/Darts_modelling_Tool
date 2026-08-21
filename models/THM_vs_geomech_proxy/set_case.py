from cases.case_1 import input_data_case_1, input_data_case_2, input_data_case_3, input_data_case_4, input_data_zero_rate
from cases.case_5 import input_data_case_5
from cases.no_damage_zone import input_data_no_damage_zone
from cases.no_damage_zone_heter_mech_prop import input_data_no_damage_zone_heter_mech_prop

from cases.base import input_data_struct_like, _set_wells
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_input_data(
    case: str,
    model_folder=None,
    physics_type="single_phase_thermal",
    wells_type="doublet",
    prod_well_coords=None,
    inj_well_coords=None,
):

    thermal = "thermal" in physics_type

    case_ = os.path.basename(case) # without meshes/ part
    match case_:
        case "case_1": # uset heterogeneous poro and perm by interpolation
            input_data = input_data_case_1()
        case "case_2": # case_1 geometry/mesh but rock props assigned per tag
            input_data = input_data_case_2()
        case "case_3": # case_2 but heterogeneour rock thermal props
            input_data = input_data_case_3()
        case "case_4": # case_3 but heterogeneous geomechanical props (E, nu, biot, th_expn) per tag
            input_data = input_data_case_4()
        case "zero_rate":
            input_data = input_data_zero_rate()
        case "case_5":
            input_data = input_data_case_5(
                prod_well_coords=prod_well_coords,
                inj_well_coords=inj_well_coords,
            )
        case "no_damage_zone":
            input_data = input_data_no_damage_zone()
        case "no_damage_zone_heter_mech_prop":
            input_data = input_data_no_damage_zone_heter_mech_prop()
        case _:  # default (struct-like meshes)
            model_folder = os.path.basename(model_folder)
            os.makedirs(os.path.join(BASE_DIR, "meshes", model_folder), exist_ok=True)
            input_data = input_data_struct_like(
                model_folder=model_folder,
                physics_type=physics_type,
                wells_type=wells_type,
            )

    # Named cases are built from input_data_base(), whose default is thermal=True.
    # Apply the caller's requested physics/well mode here as the single source of
    # truth; otherwise an isothermal Model creates thermal physics (p, T) while its
    # initialization supplies pressure only.
    input_data.type_hydr = "thermal" if thermal else "isothermal"
    input_data.type_mech = "thermoporoelasticity" if thermal else "poroelasticity"
    input_data.other.thermal = thermal
    input_data.other.wells_type = wells_type
    _set_wells(input_data)

    # Explicit coordinates take precedence over the case defaults recomputed above.
    if prod_well_coords is not None:
        input_data.other.prod_well_coords = list(prod_well_coords)
    if inj_well_coords is not None:
        input_data.other.inj_well_coords = list(inj_well_coords)

    return input_data
