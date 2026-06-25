from examples.base import InputDataConfig, build_input_data, parse_structured_dims


def input_data_struct_like(model_folder, physics_type, wells_type):
    # an unstructured mesh is generated on the fly, with cell shapes being rectangular hexahedrons
    thermal = "thermal" in physics_type
    if thermal:
        type_hydr = "thermal"
        type_mech = "thermoporoelasticity"
    else:
        type_hydr = "isothermal"
        type_mech = "poroelasticity"

    initial_composition = None
    if physics_type in ("dead_oil", "dead_oil_thermal"):
        initial_composition = [0.67]

    config = InputDataConfig(
        type_hydr=type_hydr,
        type_mech=type_mech,
        permeability=10.0,
        rsv_top=2000.0,
        rsv_bottom=2400.0,
        rsv_xy=1000.0,
        wells_type=wells_type,
        doublet_shift=500.0,
        cell_shift=0.0,
        thermal=thermal,
        matrix_tags=(99991,),
        initial_composition=initial_composition,
        dims=parse_structured_dims(model_folder),
        add_structured_mesh=True,
    )
    return build_input_data(config)
