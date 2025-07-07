from typing import Dict, List, Union

import numpy as np


class RockProps:
    """
    Rock hydrodynamic properties
    """

    def __init__(self, type_hydr='', type_mech=''):
        """
        :param type_hydr: if 'isothermal' - isothermal flow; if 'thermal' - thermal flow
        :param type_mech: if 'none' - mechanics off; other options: 'poroelasticity', 'thermoporoelasticity'
        """
        self.porosity = None
        self.perm = None  # Permeability tensor, 9 values [mD]
        self.permx = self.permy = self.permz = None  # Permeability [mD]
        self.compressibility = None
        self.density = None

        if type_hydr == 'thermal':  # thermal properties
            self.heat_capacity = None  # [kJ/m3/K]
            self.conductivity = None  # thermal conductivity [kJ/m/day/K]

        if type_mech != 'none':  # geomechanical properties
            self.E = None  # Young modulus [bars]
            self.nu = None  # Poisson ratio
            self.stiffness = None  # Stiffness tensor
            self.biot = None  # Biot
        else:  # only hydrodynamic
            self.compressibility = None  # [1/bar]

        if type_mech == 'thermoporoelasticity':  # THM
            self.th_expn = None  # thermal expansion coefficient # [1/K] #TODO Linear?

    def get_permxyz(self):
        if self.perm is None:
            return self.permx, self.permy, self.permz
        else:
            if np.isscalar(self.perm):
                return self.perm, self.perm, self.perm
            else:  # tensor
                return self.perm


class FluidProps:
    """
    Fluid properties
    """

    def __init__(self):
        self.compressibility = None  # TODO units
        self.density = None  # Density at reference conditions, #TODO units
        self.viscosity = None  # TODO units
        self.Mw = None  # molar weight, [g/mol]


class InitialSolution:
    """
    Class for initial values
    """

    def __init__(self, type='uniform'):
        self.type = type
        if type == 'uniform':
            self.initial_pressure = None  # [bars]
            self.initial_temperature = None  # [K]
        elif type == 'gradient':
            self.reference_depth_for_temperature = None  # [m]
            self.temperature_gradient = None  # [K/km]
            self.temperature_at_ref_depth = None  # [K]
            self.reference_depth_for_pressure = None  # [m]
            self.pressure_gradient = None  # [bar/km]
            self.pressure_at_ref_depth = None  # [bars]
        self.initial_displacements = None  #  [U_x, U_y, U_z] [m]
        self.initial_composition = None


class MeshData:
    """
    tags
    """

    def __init__(self):
        self.bnd_tags = None
        self.matrix_tags = None
        self.mesh_filename = None


class WellControl:
    """
    constant well controls during the simulation
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.type = None  # 'prod' or 'inj'
        self.mode = None  # 'rate' or 'bhp'
        # bhp control
        self.bhp = None  # bars
        # rate control
        self.rate = (
            None  # m3/day for Geothermal physics ans kmol/day for Compositional physics
        )
        self.bhp_constraint = None  # lower limit for bhp, bars
        # if thermal
        self.inj_bht = None  # K
        # if Compositional
        self.phase_name = None  # phase name for well control, [str]

    def prod_rate_control(self, rate, rate_type, bhp_constraint=None, phase_name=None):
        self.reset()
        self.type = 'prod'
        self.mode = 'rate'
        self.rate = rate
        self.rate_type = rate_type
        self.bhp_constraint = bhp_constraint
        # if Compositional
        self.phase_name = phase_name  # produced phase name, [str]

    def prod_bhp_control(self, bhp):
        self.reset()
        self.type = 'prod'
        self.mode = 'bhp'
        self.bhp = bhp

    def inj_rate_control(
        self,
        rate,
        rate_type,
        bhp_constraint=None,
        temperature=None,
        phase_name=None,
        inj_composition=None,
    ):
        self.reset()
        self.type = 'inj'
        self.mode = 'rate'
        self.rate = rate
        self.rate_type = rate_type
        self.bhp_constraint = bhp_constraint
        # if thermal
        self.inj_bht = temperature  # K
        # if Compositional
        self.phase_name = phase_name  # injected phase name, [str]
        self.inj_composition = inj_composition  #  0 < injected composition < 1

    def inj_bhp_control(
        self, bhp, temperature=None, phase_name=None, inj_composition=None
    ):
        self.reset()
        self.type = 'inj'
        self.mode = 'bhp'
        self.bhp = bhp
        # if thermal
        self.inj_bht = temperature  # K
        # if Compositional
        self.phase_name = phase_name  # injected phase name, [str]
        self.inj_composition = inj_composition  #  0 < injected composition < 1


class WellLocIJK:
    """
    well location for structured grid, 1-based integer grid cell indices I,J,K
    """

    def __init__(self):
        self.I = None
        self.J = None
        self.K = None


class WellLocXYZ:
    """
    well location for any kind of grid, real coordinates X, Y, Z
    """

    def __init__(self):
        self.X = None
        self.Y = None
        self.Z = None


class Well:
    """
    well definition
    """

    def __init__(self, loc_type: str):
        self.controls = []  # List[WellControl]
        self.perforations = []
        if loc_type == 'ijk':
            self.location = WellLocIJK()
        elif loc_type == 'xyz':
            self.location = WellLocXYZ()
        else:
            print('Unknown loc_type', loc_type)
            exit(1)


class WellPerforation:
    def __init__(
        self,
        loc_ijk: Union[int, tuple],
        status: str,
        well_radius: float,
        well_index: float,
        well_indexD: float,
        multi_segment: bool,
    ):
        self.loc_ijk = loc_ijk
        self.status = status
        self.well_radius = well_radius
        self.well_index = well_index
        self.well_indexD = well_indexD
        self.multi_segment = multi_segment


class WellData:
    """
    well definition
    """

    def __init__(self):
        self.wells = dict()

    def add_well(
        self,
        name: str,
        loc_type: str,
        loc_ijk: Union[int, tuple] = None,
        loc_xyz: Union[float, tuple] = None,
    ):
        assert name not in self.wells, 'The well ' + name + ' has been already added!'
        w = Well(loc_type=loc_type)
        if loc_ijk is not None and loc_type == 'ijk':
            w.location.I, w.location.J, w.location.K = loc_ijk
        elif loc_xyz is not None and loc_type == 'xyz':
            w.location.X, w.location.Y, w.location.Z = loc_xyz
        else:
            print('Unknown loc_type', loc_type)
            exit(1)
        self.wells[name] = w

    def add_perforation(
        self,
        name: str,
        time: float,
        loc_ijk: Union[int, tuple],
        status: str,
        well_radius: float,
        well_index: float,
        well_indexD: float,
        multi_segment: bool,
    ):
        """
        :param name: well name
        :param time: simulation timestep, [days]
        """
        if name not in self.wells:
            self.add_well(name=name, loc_type='ijk', loc_ijk=loc_ijk)
        eps = 1e-5
        if status == 'close':
            # well connections in DARTS cannot be changed during the simulation, so they can be only closed
            # and re-opened throughout timesteps. well_index and well_indexD can be changed as well.
            # multi_segment option can't be changed and should be the same for all perforations
            well_index_ = well_indexD_ = eps
        else:
            well_index_ = well_index
            well_indexD_ = well_indexD
        perf = WellPerforation(
            loc_ijk=loc_ijk,
            status=status,
            well_radius=well_radius,
            well_index=well_index_,
            well_indexD=well_indexD_,
            multi_segment=multi_segment,
        )
        self.wells[name].perforations.append((time, perf))

    def read_and_add_perforations(self, sch_fname, verbose: bool = False):
        """
        read COMPDAT from SCH file in Eclipse format, add wells and perforations
        note: uses only I,J,K1,K2 and optionally WellIndex parameters from the COMPDAT keyword
        :param: sch_fname - path to file
        """
        if sch_fname is None:
            return
        print('reading wells (COMPDAT) from', sch_fname)
        well_diam = 0.152  # m.  #TODO read from the keyword parameters
        well_radius = well_diam / 2.0

        keep_reading = True
        prev_well_name = ''
        with open(sch_fname) as f:
            while keep_reading:
                buff = f.readline()
                if 'COMPDAT' in buff:
                    while True:  # be careful here
                        buff = f.readline()
                        if len(buff) != 0:
                            CompDat = buff.split()
                            wname = (
                                CompDat[0].strip('"').strip("'")
                            )  # remove quotas (" and ')
                            if (
                                len(CompDat) != 0 and '/' != wname
                            ):  # skip the empty line and '/' line
                                # define perforation
                                i1 = int(CompDat[1])
                                j1 = int(CompDat[2])
                                k1 = int(CompDat[3])
                                k2 = int(CompDat[4])

                                well_index = None
                                if len(CompDat) > 7:
                                    assert (
                                        '*' not in CompDat[5] and '*' not in CompDat[6]
                                    ), (
                                        'Reading SCH with default params is not supported:'
                                        + buff
                                    )
                                    c7 = CompDat[7]
                                    if c7 not in ['*', '/']:
                                        well_index = float(c7)

                                for k in range(k1, k2 + 1):
                                    # TODO support time>0
                                    self.add_perforation(
                                        name=wname,
                                        time=0.0,
                                        loc_ijk=(i1, j1, k),
                                        status='open',
                                        well_radius=well_radius,
                                        well_index=well_index,
                                        well_indexD=0.0,
                                        multi_segment=False,
                                    )
                            if len(CompDat) != 0 and '/' == CompDat[0]:
                                keep_reading = False
                                break
        print('WELLS read from SCH file:', len(self.wells))

    def add_control(
        self,
        name: str,
        time: float,
        type: str,
        mode: str,
        rate: float,
        bhp: float,
        bhp_constraint: float,
        inj_temp: float,
        phase_name: str,
    ):
        """
        :param name: well name
        :param time: simulation timestep, [days]
        :param type: 'inj' or 'prod'
        :param mode: 'rate' or 'bhp'
        :param rate: well rate, unit depends on physics, can be None if bhp-controlled
        :param bhp: bottom hole pressure, can be None if rate-controlled
        :param bhp_constraint: bottom hole pressure constraint (min for prod and max for inj wells)
        :param inj_temp: injection temperature, [K]
        :param phase_name # injected phase name, [str], for Compositional physics
        :return:
        """
        self.wells[name].controls.append(
            (
                time,
                WellControl(
                    type=type,
                    mode=mode,
                    rate=rate,
                    bhp=bhp,
                    bhp_constraint=bhp_constraint,
                    inj_temp=inj_temp,
                    phase_name=phase_name,
                ),
            )
        )

    def add_prd_rate_control(
        self, name, rate, rate_type, bhp_constraint=None, phase_name=None, time=0
    ):
        wctrl = WellControl()
        wctrl.prod_rate_control(
            rate=rate,
            rate_type=rate_type,
            bhp_constraint=bhp_constraint,
            phase_name=phase_name,
        )
        self.wells[name].controls.append((time, wctrl))

    def add_prd_bhp_control(self, name, bhp, time=0):
        wctrl = WellControl()
        wctrl.prod_bhp_control(bhp=bhp)
        self.wells[name].controls.append((time, wctrl))

    def add_inj_rate_control(
        self,
        name,
        rate,
        rate_type,
        bhp_constraint=None,
        temperature=None,
        phase_name=None,
        inj_composition=[],
        time=0,
    ):
        wctrl = WellControl()
        wctrl.inj_rate_control(
            rate=rate,
            rate_type=rate_type,
            bhp_constraint=bhp_constraint,
            temperature=temperature,
            phase_name=phase_name,
        )
        self.wells[name].controls.append((time, wctrl))

    def add_inj_bhp_control(
        self, name, bhp, temperature=None, phase_name=None, inj_composition=[], time=0
    ):
        wctrl = WellControl()
        wctrl.inj_bhp_control(bhp=bhp, temperature=temperature, phase_name=phase_name)
        self.wells[name].controls.append((time, wctrl))


class OBLParams:
    """
    OBL range, number of points
    """

    def __init__(self):
        self.zero = None
        self.n_points = None
        self.min_p = None
        self.max_p = None
        self.min_t = None
        self.max_t = None
        self.min_z = None
        self.max_z = None


class Simulation:
    """ """

    def __init__(self):
        self.time_steps = None


class OtherProps:
    """
    Other user defined properties
    """

    def __init__(self):
        pass


class InputData:
    """
    Class for initial values
    """

    def __init__(self, type_hydr, type_mech, init_type):
        self.type_hydr = type_hydr
        self.type_mech = type_mech
        self.rock = RockProps(type_hydr, type_mech)
        self.fluid = FluidProps()
        self.obl = OBLParams()
        self.initial = InitialSolution(init_type)
        self.well_data = WellData()
        self.mesh = MeshData()
        self.boundary = None
        self.sim = Simulation()
        self.other = OtherProps()

    def check(self):
        assert self.type_hydr in [
            'isothermal',
            'thermal',
        ], 'input_data: Unknown type_hydr'
        assert self.type_mech in [
            'poroelasticity',
            'thermoporoelasticity',
            'none',
        ], 'input_data: Unknown type_mech'
        for (
            k
        ) in (
            self.__dict__.keys()
        ):  #  loop over the attributes (self.rock, self.fluid, ..)
            sub_obj = self.__getattribute__(k)
            if not hasattr(sub_obj, '__dict__'):
                continue
            if k in [
                'initial',
                'mesh',
                'sim',
                'other',
            ]:  # do not check initial currently #TODO
                continue
            for (
                k2
            ) in sub_obj.__dict__.keys():  #  loop over the attributes in sub object
                value = sub_obj.__getattribute__(k2)
                if value is None:
                    # either perm or permx+permy+permx should be specified
                    if k2 == 'permx' or k2 == 'permy' or k2 == 'permz':
                        if sub_obj.__dict__['perm'] is not None:
                            continue
                    if k2 == 'perm':
                        if (
                            sub_obj.__dict__['permx'] is not None
                            and sub_obj.__dict__['permy'] is not None
                            and sub_obj.__dict__['permz'] is not None
                        ):
                            continue
                    # if stiffness specified, then allow E and nu non-specified
                    if k2 == 'E' or k2 == 'nu':
                        if sub_obj.__dict__['stiffness'] is not None:
                            continue
                    print(
                        'Error in InputData check: property',
                        k,
                        k2,
                        'is not initialized!',
                    )
                    assert False

    def make_prop_arrays(self):
        """
        one can specify different property values with a numpy array, and another property as a scalar value
        in that case, this function can replace scalar properties with a uniform array, so the props
        can be later used in operations. If some of props are not initialized (i.e. =None) they will be skipped.
        :return:
        """
        array_obj = ['rock']  # list of items which can be defined by regons
        # count number of regions (one value per region)
        max_n_regions = 1
        for k in self.__dict__.keys():  #  loop over the attributes (self.rock, ..)
            if k not in array_obj:
                continue
            sub_obj = self.__getattribute__(k)
            if not hasattr(sub_obj, '__dict__'):
                continue
            for (
                k2
            ) in sub_obj.__dict__.keys():  #  loop over the attributes in sub object
                value = sub_obj.__getattribute__(k2)
                if value is None:
                    continue
                if not np.isscalar(value):  # if np.array
                    max_n_regions = value.size
        # make arrays from scalar fields
        for (
            k
        ) in (
            self.__dict__.keys()
        ):  # loop over the attributes (self.rock, self.fluid, ..)
            if k not in array_obj:
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
                    self.__dict__[k].__dict__[k2] = (
                        np.zeros(max_n_regions, dtype=type(value)) + value
                    )
