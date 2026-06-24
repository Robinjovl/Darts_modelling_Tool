from examples.case_1 import input_data_case_1
from examples.generate_model_case import input_data_other_model
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_input_data(
    case: str,
    model_folder=None,
    physics_type="single_phase_thermal",
    wells_type="doublet",
    return_well_init_depth=False,
):
    case_name = case.lower()

    if case_name == "case_1":
        input_data = input_data_case_1()
        well_init_depth = (input_data.other.rsv_top + input_data.other.rsv_bottom) * 0.5
    else:
        model_folder = os.path.basename(model_folder)
        os.makedirs(os.path.join(BASE_DIR, "meshes", model_folder), exist_ok=True)
        input_data, well_init_depth = input_data_other_model(
            model_folder=model_folder,
            physics_type=physics_type,
            wells_type=wells_type,
        )

    if return_well_init_depth:
        return input_data, well_init_depth
    return input_data
