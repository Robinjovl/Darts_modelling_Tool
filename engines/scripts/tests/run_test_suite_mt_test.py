from darts.engines import *
from darts.physics import *
from darts.nonlinear_solvers import NewtonSolver
import numpy as np
import os, sys
from pathlib import Path
import importlib
from darts.tools.cicd_tools import check_performance as cicd_check_performance, get_performance_data


model_dir = r'../../../darts-models'
exception_dirs = ['darts', 'SPE10']
accepted_dirs = ['1D_4comp','SPE1_fine','Brugge','Geothermal_full', '2comp_geotherm','SPE10_2l_comp']
#accepted_dirs = ['SPE1_fine','Brugge','Geothermal_full']
#accepted_dirs = ['SPE10']
null = open(os.devnull, 'w')
orig_stdout = sys.stdout

# set working directory to folder which contains tests
os.chdir(model_dir)

p = Path('.')

# save original system path and loaded modules
sys_path_init = sys.path.copy()
sys_modules_init = sys.modules.copy()

# switch off darts output
redirect_darts_output('')

# iterate over dirs in 'model_dir'
for x in p.iterdir():
    if x.is_dir() and (str(x)[0] != '.') and str(x) in accepted_dirs:
        print("Running {:<30}".format(str(x) + ': '), end = '', flush= True)

        # step into folder with model
        os.chdir(x)
        # add it also to system path to load modules
        sys.path.append(os.path.abspath('.'))

        # import model and run it for default time
        try:
            mod = importlib.import_module('model')

            if 0:
              sys.stdout = null
              m = mod.Model()
              m.nonlinear_solver = NewtonSolver(tolerance=1e-1)
              m.params.tolerance_linear = 1e-10
              m.init()
              m.run()
              sys.stdout = orig_stdout
              cicd_check_performance(m)
            else:
              print ()
              for nt in [1, 2, 4, 6, 8, 10]:
                sys.stdout = null
                set_num_threads(nt)
                m = mod.Model()
                m.nonlinear_solver = NewtonSolver(tolerance=1e-1)
                m.init()
                m.run()
                sys.stdout = orig_stdout
                perf_data = get_performance_data(m)
                #print ()
                print ('nt=',nt, 'total: %f, lin it: %d, lin: %f, jac: %f (%f), interp: %f (%f), gen: %f' % (m.timer.node['simulation'].get_timer(), perf_data["linear iterations"] + perf_data['wasted newton iterations'],
                                                                      m.timer.node['simulation'].node["linear solver setup"].get_timer() +
                                                                      m.timer.node['simulation'].node["linear solver solve"].get_timer(),
                                                                      m.timer.node['simulation'].node["jacobian assembly"].get_timer(),  m.timer.node['simulation'].node["jacobian assembly"].get_timer() - perf_data['interpolation incl. generation time'],
                                                                      perf_data['interpolation incl. generation time'], perf_data['interpolation excl. generation time'],
                                                                      m.timer.node['simulation'].node["jacobian assembly"].node["interpolation"].node["acc flux interpolation"].node["body generation"].get_timer()))


            #m.print_timers()

        except Exception as err:
            sys.stdout = orig_stdout
            print (err)

        # model has been run, prepare for next one

        # un-import imported modules (next model could have same modules names)
        keys = list(sys.modules.keys())
        for m in keys:
            if m not in sys_modules_init.keys():
                del sys.modules[m]

        # remove current folder from system path
        sys.path.pop()
        # step back to parent folder with models
        os.chdir('..')
