from darts.engines import *
from for_each_model import *

model_dir = r'.'
exception_dirs = ['darts', 'SPE10']
accepted_dirs = ['1D_4comp', 'SPE1_fine', 'Brugge', '2comp_geotherm', 'SPE10_2l_comp',
                 'Unstructured_fine', 'Egg', 'ChemicalEqui', 'ChemicalKin', 'displaced_fault',
                 'SPE9', 'Geothermal_2D_high_enth', 'Geothermal_3D_low_enth']

log_file = os.path.abspath('log.txt')

# accepted_dirs = ['SPE10_2l_comp']

def check_performance(mod):
    x = os.path.basename(os.getcwd())
    print("Running {:<30}".format(x + ': '), end='', flush=True)
    # append model output to log file
    log_stream = redirect_all_output(log_file)
    # create model instance
    m = mod.Model()

    m.init()
    m.run()
    m.print_stat()
    abort_redirection(log_stream)
    passed = m.check_performance(overwrite=0)

    return passed

if __name__ == '__main__':
    # erase previous log file if existed
    f = open(log_file, "w")
    f.close()
    for_each_model(model_dir, check_performance, accepted_dirs)
    input("Press Enter to continue...")
