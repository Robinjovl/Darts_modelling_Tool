from sys import stdout, stderr
import time
import sys, os, shutil
import subprocess

from darts.models.cicd_model import get_platform, is_iter_solvers
from darts.engines import print_build_info as engines_pbi


def run_testing(platform, redirect_output=True):
    iter_solvers = is_iter_solvers()

    # define a model list to run
    accepted_dirs = [ # 2 phase (Compositional engine)
                     '2ph_comp',
                     '2ph_comp_solid',
                     '2ph_do',
                     '2ph_do_thermal',
                     '2ph_geothermal',
                     '2ph_geothermal_mass_flux',
                     # 3 phase (Compositional engine)
                     '3ph_comp_w',
                     '3ph_do',
                     '3ph_bo',
                     # ?
                     'Uniform_Brugge',
                     # chemistry (Compositional engine)
                     'Chem_benchmark_new',
                     # Geothermal engine
                     'GeoRising',
                     'CoaxWell',
                     # with flash (Compositional engine)
                     'CCS',
                     'SPE11b',
                     'effect_of_potential_energy',
                     # 'CO2_foam_CCS',
                     # models with multiple cases
                     'cpg_sloping_fault', # # Geothermal engine / Deadoil (Compositional engine)
                     'fracture_network',  # Geothermal engine
                     # adjoint gradients
                     'Adjoint_super_engine'
                     ]

    if platform == 'cpu':  # MPFA code is excluded from gpu build due to compilation issues (c++ std 20)
        accepted_dirs += ['2ph_do_thermal_mpfa']
        accepted_dirs += ['Adjoint_mpfa']

        # Tests for drift-flux well model (DFM) (implemented only for CPU)
        accepted_dirs += [
            # Coupled well-reservoir modeling using DFM wells is
            os.path.join('dfm_well', 'coupled_dfm_well_reservoir'),
            # Single-phase thermal well flow in a DFM well
            os.path.join('dfm_well', 'single_phase_thermal_dfm_well_flow'),
            # Two-phase isothermal well flow in a DFM well
            os.path.join('dfm_well', 'two_phase_isothermal_dfm_well_flow'),
        ]

        # mechanical models (with multiple cases)
        accepted_dirs += ['1ph_1comp_poroelastic_analytics']
        accepted_dirs += ['1ph_1comp_poroelastic_convergence']
        if iter_solvers:
            accepted_dirs += ['SPE10_mech']

    # RUN main.py files in the folder listed in the accepted_dirs
    n_total = 0
    failed_models = []
    for mdir in accepted_dirs:
        print('running main.py for model', mdir)
        n_total += 1
        os.chdir(mdir)
        shutil.rmtree("__pycache__", ignore_errors=True)
        if redirect_output:
            stdout_ = open('../_logs/' + mdir + '.log', 'w')
            stderr_ = open('../_logs/' + mdir + '_err.log', 'w')
        else:
            stdout_ = stdout
            stderr_ = stderr
        starting_time = time.time()
        mrun = subprocess.run(["python", "main.py"], stdout=stdout_, stderr=stderr_)
        ending_time = time.time()
        rcode = mrun.returncode
        if not rcode:
            print('OK', '\t%.2f s' % (ending_time - starting_time))
        else:
            print('FAIL, \t%.2f s' % (ending_time - starting_time))
            failed_models += [mdir]
            # duplicate 10 last lines to screen from the error log file
            with open('../_logs/' + mdir + '_err.log', 'r') as f:
                s = f.readlines()
                s = s if len(s) <= 10 else s[-10:]
                print('\t' + '\t'.join(s))
        os.chdir('..')

    n_failed = len(failed_models)
    n_passed = n_total - len(failed_models)
    print("Passed", n_passed, "of", n_total, "tests ")
    print('n_failed     =', n_failed)
    #print('n_failed_cpg =', n_failed_discr)
    #print('n_failed_dfn =', n_failed_dfn)
    #print('n_failed_mech=', n_failed_mech)

    print('Failed models   :\n\t', '\n\t'.join(failed_models))

    # exit with code equal to the number of failed models
    exit(n_failed)


if __name__ == '__main__':

    # print build info
    engines_pbi()

    # multithreaded run can be enabled by setting OMP_NUM_THREADS environment variable
    if os.getenv('OMP_NUM_THREADS') == None:
        os.environ['OMP_NUM_THREADS'] = '1'
    print('OMP_NUM_THREADS=', os.environ['OMP_NUM_THREADS'])

    # cpu/gpu
    platform = get_platform()
    print('platform=', platform)

    run_testing(platform)
