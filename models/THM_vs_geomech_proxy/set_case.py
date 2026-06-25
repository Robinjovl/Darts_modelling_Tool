from examples.case_1 import input_data_case_1
from examples.generate_model_case import input_data_struct_like
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_input_data(
    case: str,
    model_folder=None,
    physics_type="single_phase_thermal",
    wells_type="doublet",
):

    if  "case_1" in case.lower():
        input_data = input_data_case_1()
    else:
        model_folder = os.path.basename(model_folder)
        os.makedirs(os.path.join(BASE_DIR, "meshes", model_folder), exist_ok=True)
        input_data = input_data_struct_like(
            model_folder=model_folder,
            physics_type=physics_type,
            wells_type=wells_type,
        )

    return input_data
