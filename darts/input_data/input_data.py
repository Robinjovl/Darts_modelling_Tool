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
        self.permx = self.permy = self.permz = None  # Permeability [mD]
        self.compressibility = None
        
        if type_hydr == 'thermal':  # thermal properties
            self.heat_capacity = None  # [kJ/m3/K]
            self.conductivity = None   # thermal conductivity [kJ/m/day/K]
        
        if type_mech != '': # geomechanical properties
            self.E = None   # Young modulus [bars]
            self.nu = None  # Poisson ratio
            self.biot = None  # Biot
            self.kd_cur = None  # Bulk modulus #TODO units
        else: # only hydrodynamic
            self.compressibility = 1.   #TODO check

        if type_mech == 'thermoporoelasticity': # THM
            self.th_expn = None  # thermal expansion coefficient # [1/K] #TODO Linear?

class FluidProps():
    '''
    Fluid properties
    '''
    def __init__(self):
        self.compressibility = None  #TODO units
        self.density_ref = None  # Density at reference conditions, #TODO units
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
            self.temperature_gradient = None  # [C/m]
            self.reference_depth_for_pressure = None  # [m]
            self.pressure_gradient = None  # [bar/m]

class InputData():
    '''
    Class for initial values
    '''
    def __init__(self, type_hydr, type_mech):
        self.rock = RockProps(type_hydr, type_mech)
        self.fluid = FluidProps()
        #self.initial = InitialSolution() #TODO
        
    def check(self):
        for k in self.__dict__.keys():  #  loop over the attributes (self.rock, self.fluid, ..)
            sub_obj = self.__getattribute__(k)
            for k2 in sub_obj.__dict__.keys(): #  loop over the attributes in sub object
                value = sub_obj.__getattribute__(k2)
                if value is None:
                    print('Error in InputData check: property', k, k2, 'is not initialized!')
                    assert False
                    
    def make_prop_arrays(self):
        # count number of regions
        max_n_regions = 1
        for k in self.__dict__.keys():  #  loop over the attributes (self.rock, self.fluid, ..)
            sub_obj = self.__getattribute__(k)
            for k2 in sub_obj.__dict__.keys(): #  loop over the attributes in sub object
                value = sub_obj.__getattribute__(k2)
                if not np.isscalar(value):
                    max_n_regions = value.size()
        # make arrays from scalar fields
        for k in self.__dict__.keys():  # loop over the attributes (self.rock, self.fluid, ..)
            sub_obj = self.__getattribute__(k)
            for k2 in sub_obj.__dict__.keys():  # loop over the attributes in sub object
                value = sub_obj.__getattribute__(k2)
                if np.isscalar(value):
                    self.__dict__[k].__dict__[k2] = np.zeros(max_n_regions, dtype=type(value)) + value