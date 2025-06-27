from darts.engines import *
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.engines import sim_params
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.flash import ConstantK
from darts.physics.properties.density import DensityBasic

from darts.models.opt.opt_module_settings import OptModuleSettings
from darts.tools.keyword_file_tools import get_table_keyword


class Model(CICDModel, OptModuleSettings):
    def __init__(self, T, report_step=120, perm=300, poro=0.2, customize_new_operator=False, Peaceman_WI=False):
        # call base class constructor
        CICDModel.__init__(self)
        OptModuleSettings.__init__(self)

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.T = T
        self.report_step = report_step
        self.customize_new_operator = customize_new_operator

        # initialize global data to record the well location in vtk output file
        self.global_data = {'well location': 0}  # will be updated later in "run"

        self.set_reservoir(perm, poro)
        self.Peaceman_WI = Peaceman_WI
        self.set_physics()

        self.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=1, runtime=1000,
                            tol_newton=1e-6, tol_linear=1e-3, it_newton=10, it_linear=50,
                            newton_type=sim_params.newton_local_chop)

        self.timer.node["initialization"].stop()

    def set_reservoir(self, perm, poro):
        """Reservoir construction"""
        # nx = 20
        # ny = 10
        # nz = 2

        nx = 5
        ny = 5
        nz = 2

        # reservoir geometry： for realistic case, one just needs to load the data and input it
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=30, dy=30, dz=12,
                                         permx=perm, permy=perm, permz=perm, poro=poro, depth=2000)

        return

    def set_wells(self):
        # self.inj_list = [[5, 5]]
        # self.prod_list = [[15, 3], [15, 8]]

        self.inj_list = [[1, 5], [5, 1]]
        self.prod_list = [[1, 1], [5, 5]]

        # well index setting
        if self.Peaceman_WI:
            WI = -1  # use Peaceman function; check the function "add_perforation" for more details
        else:
            WI = 200

        n_perf = self.reservoir.nz
        for i, inj in enumerate(self.inj_list):
            self.reservoir.add_well('I' + str(i + 1))

            for k in range(n_perf):
                self.reservoir.add_perforation('I' + str(i + 1), cell_index=(inj[0], inj[1], k + 1),
                                               well_radius=0.1, well_index=WI)

        for p, prod in enumerate(self.prod_list):
            self.reservoir.add_well('P' + str(p + 1))

            for k in range(n_perf):
                self.reservoir.add_perforation('P' + str(p + 1), cell_index=(prod[0], prod[1], k + 1),
                                               well_radius=0.1, well_index=WI)

    def set_physics(self):
        """Physical properties"""
        # Create property containers:
        zero = 1e-8
        components = ['CO2', 'C1', 'H2O']
        phases = ['gas', 'oil']
        Mw = [44.01, 16.04, 18.015]
        nc = len(components)

        self.inj_composition = [1.0 - 2 * zero, zero]
        self.ini_stream = [0.1, 0.2]

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=Mw,
                                               min_z=zero / 10, temperature=1.)
        property_container.flash_ev = ConstantK(nc, [4, 2, 1e-1], zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('oil', DensityBasic(compr=1e-5, dens0=600))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.05)),
                                                ('oil', ConstFunc(0.5))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('oil', PhaseRelPerm("oil"))])

        """ Activate physics """
        thermal = False
        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, self.timer, state_spec=state_spec,
                                     n_points=200, min_p=1, max_p=300, min_z=zero/10, max_z=1-zero/10)
        self.physics.add_property_region(property_container)

        return

    def set_initial_conditions(self):
        input_distribution = {self.physics.vars[0]: 50.,
                              self.physics.vars[1]: self.ini_stream[0],
                              self.physics.vars[2]: self.ini_stream[1],
                              }
        return self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh,
                                                              input_distribution=input_distribution)


    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if "I" in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=140., inj_composition=self.inj_composition)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=50.)

    def set_op_list(self):
        if self.customize_new_operator:
            customized_component_etor = customized_etor_specific_component()
            axes_min = self.physics.axes_min
            axes_max = self.physics.axes_max
            customized_component_itor = self.physics.create_interpolator(customized_component_etor,
                                                                         axes_min=self.physics.axes_min,
                                                                         axes_max=self.physics.axes_max,
                                                                         n_ops=1,
                                                                         platform='cpu', algorithm='multilinear',
                                                                         mode='adaptive', precision='d',
                                                                         timer_name='customized component interpolation')
            self.physics.engine.customize_operator = self.customize_new_operator

            self.op_list = [self.physics.acc_flux_itor[0], customized_component_itor]

            # specify the index of blocks of customized operator
            idx_in_op_list = 1
            op_num_new = np.array(self.reservoir.mesh.op_num, copy=True)
            op_num_new[:] = idx_in_op_list  # set the second interpolator (i.e. "customized_component_itor") from "self.op_list" to all blocks
            self.physics.engine.idx_customized_operator = idx_in_op_list
            self.physics.engine.customize_op_num = index_vector(op_num_new)
        else:
            # self.op_list = [self.physics.acc_flux_itor]

            self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
            n_res = self.reservoir.mesh.n_res_blocks
            self.op_num[n_res:] = 1
            self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor]

    def run(self, export_to_vtk=False, file_name='data'):
        import random
        # use seed to generate the same random values every run
        random.seed(3)
        if export_to_vtk:

            well_loc = np.zeros(self.reservoir.n)
            for inj in self.inj_list:
                well_loc[(inj[1] - 1) * self.reservoir.nx + inj[0] - 1] = -1

            for prod in self.prod_list:
                well_loc[(prod[1] - 1) * self.reservoir.nx + prod[0] - 1] = 1

            self.global_data = {'well location': well_loc}

            self.export_vtk(file_name, global_cell_data=self.global_data)

        # now we start to run for the time report--------------------------------------------------------------
        time_step = self.report_step
        even_end = int(self.T / time_step) * time_step
        time_step_arr = np.ones(int(self.T / time_step)) * time_step
        if self.T - even_end > 0:
            time_step_arr = np.append(time_step_arr, self.T - even_end)

        for ts in time_step_arr:
            from darts.engines import well_control_iface
            for i, w in enumerate(self.reservoir.wells):
                if "I1" in w.name:
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.MOLAR_RATE,
                                                   is_inj=True, target=40., phase_name='gas', inj_composition=self.inj_composition)
                elif "I" in w.name:
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                                   is_inj=True, target=140., inj_composition=self.inj_composition)
                else:
                    self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                                   is_inj=False, target=50.)

            CICDModel.run(self, ts, save_well_data=False, save_reservoir_data=False, verbose=export_to_vtk)
            self.physics.engine.report()
            if export_to_vtk:
                self.export_vtk(file_name)


class customized_etor_specific_component(operator_set_evaluator_iface):
    n_ops = 1

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """

        # temp = self.temperature.evaluate(state)
        vec_values_as_np = values.to_numpy()
        vec_values_as_np[:] = 0

        # values[0] = state[0]  # pressure
        values[0] = 1 - state[1]  # comp_1
        return 0
