import numpy as np

class RockProps():
    '''
    Rock hydrodynamic properties
    '''
    def __init__(self, type_hydr='', type_mech=''):
        '''
        :param type_hydr: if '' - idothermal flow; if 'thermal' - thermal flow
        :param type_mech: if '' - mechanics off; options: 'poroelasticity', 'thermoporoelasticity'
        '''
        self.porosity = None
        self.perm = None  # Permeability tensor, 9 values [mD]
        self.permx = self.permy = self.permz = None  # Permeability [mD]
        self.compressibility = None
        self.density = None
        
        if type_hydr == 'thermal':  # thermal properties
            self.heat_capacity = None  # [kJ/m3/K]
            self.conductivity = None   # thermal conductivity [kJ/m/day/K]
        
        if type_mech != '': # geomechanical properties
            self.E = None   # Young modulus [bars]
            self.nu = None  # Poisson ratio
            self.stiffness = None  # Stiffness tensor
            self.biot = None  # Biot
        else: # only hydrodynamic
            self.compressibility = 1.   # [1/bar]

        if type_mech == 'thermoporoelasticity': # THM
            self.th_expn = None  # thermal expansion coefficient # [1/K] #TODO Linear?

    def get_permxyz(self):
        if self.perm is None:
            return self.permx, self.permy, self.permz
        else:
            if np.isscalar(self.perm):
                return self.perm, self.perm, self.perm
            else: # tensor
                return self.perm
                

class FluidProps():
    '''
    Fluid properties
    '''
    def __init__(self):
        self.compressibility = None  #TODO units
        self.density = None  # Density at reference conditions, #TODO units
        self.viscosity = None  #TODO units
        self.Mw = None  # molar weight, [g/mol]
        
class InitialSolution():
    '''
    Class for initial values
    '''
    def __init__(self, type='uniform'):
        if type == 'uniform':
            self.initial_pressure = None  # [bars]
            self.initial_temperature = None  # [K]
        elif type == 'gradient':
            self.reference_depth_for_temperature = None  # [m]
            self.temperature_gradient = None  # [K/m]
            self.reference_depth_for_pressure = None  # [m]
            self.pressure_gradient = None  # [bar/m]
        self.initial_displacements = None  #  [U_x, U_y, U_z] [m]
        self.initial_composition = None

class OBLParams():
    '''
    OBL range, number of points
    '''
    def __init__(self):
        self.zero = None
        self.n_points = None
        self.min_p = None
        self.max_p = None
        self.min_t = None
        self.max_t = None
        self.min_z = None
        self.max_z = None

class OtherProps():
    '''
    Other user defined properties
    '''
    def __init__(self):
        pass

class InputData():
    '''
    Class for initial values
    '''
    def __init__(self, type_hydr, type_mech):
        self.type_hydr = type_hydr
        self.type_mech = type_mech
        self.rock = RockProps(type_hydr, type_mech)
        self.fluid = FluidProps()
        self.obl = OBLParams()
        self.initial = InitialSolution()
        self.other = OtherProps()
        
    def check(self):
        assert self.type_hydr in ['isothermal', 'thermal'], 'input_data: Unknown type_hydr'
        assert self.type_mech in ['poroelasticity', 'thermoporoelasticity'], 'input_data: Unknown type_mech'
        for k in self.__dict__.keys():  #  loop over the attributes (self.rock, self.fluid, ..)
            sub_obj = self.__getattribute__(k)
            if not hasattr(sub_obj, '__dict__'):
                continue
            if k == 'initial':  # do not check initial currently #TODO
                continue
            for k2 in sub_obj.__dict__.keys(): #  loop over the attributes in sub object
                value = sub_obj.__getattribute__(k2)
                if value is None:
                    # either perm or permx+permy+permx should be specified
                    if k2 == 'permx' or k2 == 'permy' or k2 == 'permz':
                        if sub_obj.__dict__['perm'] is not None:
                            continue
                    if k2 == 'perm':
                        if sub_obj.__dict__['permx'] is not None and \
                                sub_obj.__dict__['permy'] is not None and \
                                sub_obj.__dict__['permz'] is not None:
                            continue
                    # if stiffness specified, then allow E and nu non-specified
                    if k2 == 'E' or k2 == 'nu':
                        if sub_obj.__dict__['stiffness'] is not None:
                            continue
                    print('Error in InputData check: property', k, k2, 'is not initialized!')
                    assert False
                    
    def make_prop_arrays(self):
        '''
        one can specify different property values with a numpy array, and another property as a scalar value
        in that case, this function can replace scalar properties with a uniform array, so the props
        can be later used in operations. If some of props are not initialized (i.e. =None) they will be skipped.
        :return:
        '''
        # count number of regions (one value per region)
        max_n_regions = 1
        for k in self.__dict__.keys():  #  loop over the attributes (self.rock, ..)
            sub_obj = self.__getattribute__(k)
            if not hasattr(sub_obj, '__dict__'):
                continue
            for k2 in sub_obj.__dict__.keys():  #  loop over the attributes in sub object
                value = sub_obj.__getattribute__(k2)
                if value is None:
                    continue
                if not np.isscalar(value):  # if np.array
                    max_n_regions = value.size
        # make arrays from scalar fields
        for k in self.__dict__.keys():  # loop over the attributes (self.rock, self.fluid, ..)
            if k == 'fluid' or k == 'obl':
                continue
            sub_obj = self.__getattribute__(k)
            if not hasattr(sub_obj, '__dict__'):
                continue
            for k2 in sub_obj.__dict__.keys():  # loop over the attributes in sub object
                if k2 == 'compressibility' or (k == 'rock' and k2 == 'density'):
                    continue
                value = sub_obj.__getattribute__(k2)
                if value is None:
                    continue
                if np.isscalar(value):
                    self.__dict__[k].__dict__[k2] = np.zeros(max_n_regions, dtype=type(value)) + value