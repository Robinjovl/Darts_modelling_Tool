def check_initial_conditions(initial_conditions, components_names, isothermal):
    """
    This function checks if all the required initial conditions of the wellbore/pipe are specified by the user.
    The required initial conditions are:
    Pressure profile of the fluid in the wellbore/pipe
    Overall mole fractions profiles of the components specified in components_names except the last component
    If the system is not isothermal, temperature profile of the fluid in the wellbore/pipe

    :param initial_conditions: Initial conditions of the wellbore/pipe
    :type initial_conditions: dict
    :param components_names: Names of the components
    :type components_names: list
    :param isothermal: If the system is isothermal or not
    :type isothermal: bool
    """
    assert 'initial_pressure' in initial_conditions, 'Initial pressure is not specified in initial conditions!'

    for component_name in components_names[:-1]:
        assert 'initial_' + component_name + '_mole_fraction' in initial_conditions, (
               'initial_' + component_name + '_mole_fraction is not specified in initial conditions!')

    if not isothermal:
        assert 'initial_temperature' in initial_conditions, \
            'Initial temperature is not specified in initial conditions!'
    elif isothermal:
        assert 'initial_temperature' not in initial_conditions, \
            'Initial temperature must not be specified if the model is isothermal!'
