import os
import sys
import subprocess
import re
import shutil
import importlib
import time

valgrind_models = [ '2ph_comp', '2ph_comp_solid', '2ph_do',
                    '2ph_geothermal', 
                    '2ph_geothermal_mass_flux',
                    '3ph_comp_w', '3ph_do', '3ph_bo',
                    'Uniform_Brugge',
                    'Chem_benchmark_new',
                    #'CO2_foam_CCS',
                    'GeoRising',
                    'CoaxWell',
                    'phreeqc_dissolution',
                    '2ph_do_thermal_mpfa'
                ]       

# if the user passed extra models, append them (comma-separated):
extra = os.environ.get('APPEND_VALGRIND_MODEL')
if extra:
    for m in extra.split(','):
        m = m.strip()
        if m and m not in valgrind_models:
            valgrind_models.append(m)

# folder with logs to be reported
log_folder = os.path.join('models', '_valgrind_logs')

# valgrind suppression file
suppression_file = 'helper_scripts/valgrind-python.supp'

# patterns to extract leak info and errors from Valgrind log
MSG_PATTERNS = {
    'definitely_lost':    r'definitely lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'indirectly_lost':    r'indirectly lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'possibly_lost':      r'possibly lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'still_reachable':    r'still reachable:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'error_summary':      r'ERROR SUMMARY:\s+(\d+)'  # total errors
}

def run_model(model):
    # add it also to system path to load modules
    sys.path.insert(0, os.path.abspath(r'.'))

    # clear any cached bytecode
    shutil.rmtree("__pycache__", ignore_errors=True)

    # import the model definition and run
    try:
        # assume model.py defines a class Model
        mod = importlib.import_module('model')
        m = mod.Model()
        m.init(platform='cpu')
        m.set_output()
        m.run(days=m.data_ts.dt_first, save_well_data=False, save_reservoir_data=False)
        print(f"[OK] {model}")
        success = True
    except Exception as e:
        print(f"[FAIL] {model} → {e}")
        success = False

    return success

def analyze_log(log_path):
    """Parse Valgrind log and return a list of summary dicts plus accumulated error count."""
    try:
        content = open(log_path, 'r').read()
    except FileNotFoundError:
        return [], 1

    # split into per-process report blocks by HEAP SUMMARY
    blocks = re.split(r"(?m)==\d+==\s+HEAP SUMMARY:", content)
    summary_list = []
    total_errors = 0

    # first block may contain header before first HEAP SUMMARY; skip it
    for block in blocks[1:]:
        # Prepend the HEAP SUMMARY: marker to each block
        block = 'HEAP SUMMARY:' + block
        report = {}
        # Extract PID if present
        pid_match = re.search(r'==([0-9]+)==', block)
        report['pid'] = pid_match.group(1) if pid_match else None
        # For each pattern
        for key, pattern in MSG_PATTERNS.items():
            match = re.search(pattern, block)
            if match:
                if key == 'error_summary':
                    err = int(match.group(1))
                    report[key] = err
                    total_errors += err
                else:
                    report[key] = {
                        'bytes': match.group(1),
                        'blocks': match.group(2)
                    }
        summary_list.append(report)

    return summary_list, total_errors

def run_valgrind_for_model(model, timeout=1800):
    # model path
    model_path = os.path.join('models', model)
    if not os.path.isdir(model_path):
        print(f"[SKIP] Model directory not found: {model_path}")
        return True # failed

    # file paths
    vg_log = os.path.join(log_folder, f'{model}.vg.log')
    prog_out = os.path.join(log_folder, f'{model}.log')
    prog_err = os.path.join(log_folder, f'{model}_err.log')
    summary_file = os.path.join(log_folder, f'{model}.summary.txt')

    # Build the inline Python snippet to invoke run_model_direct
    py_snippet = (
        'import sys, os; '
        'from helper_scripts.valgrind_check import run_model; '
        f'model_path = os.path.join("models", "{model}"); '
        'os.chdir(model_path); '
        f'sys.exit(0 if run_model(model="{model}") else 1)'
    )

    # valgrind command profiling run_model_direct
    cmd = [
        'valgrind',
        '--trace-children=yes',
        '--error-exitcode=0',
        f'--suppressions={suppression_file}',
        '--gen-suppressions=all',
        f'--log-file={vg_log}',
        '--',
        'darts', '-c', py_snippet
    ]

    print(f'Running Valgrind for model {model}...')
    starting_time = time.time()

    # set environment: disable pymalloc so Valgrind sees everything
    env = os.environ.copy()
    env['PYTHONMALLOC'] = 'malloc'
    
    # set single thread for mpfa models
    if 'mpfa' in model.split('_'):
        env['OMP_NUM_THREADS'] = '1'

    # set environment: disable pymalloc so Valgrind sees everything
    try:
        darts_path = subprocess.check_output([
            'python', '-c',
            'import os, darts; print(os.path.dirname(darts.__file__))'
        ]).decode().strip()
        env['LD_LIBRARY_PATH'] = f"{darts_path}:{env.get('LD_LIBRARY_PATH','')}"
    except subprocess.CalledProcessError as e:
        print(f'Failed to determine DARTS path: {e}')
        return True # failed
    
    with open(prog_out, 'w') as out_f, open(prog_err, 'w') as err_f:
        try:
            proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f'ERROR: timeout profiling model {model}')
            return True # failed

    # analyze and write summary
    summary_list, total_errors = analyze_log(vg_log)
    with open(summary_file, 'w') as sf:
        sf.write(f'Valgrind summary for model: {model}\n')
        for report in summary_list:
            pid = report.get('pid', 'unknown')
            sf.write(f"\nReport for PID: {pid}\n")
            for k, v in report.items():
                if k == 'pid':
                    continue
                if isinstance(v, dict):
                    sf.write(f"  {k}: {v['bytes']} bytes in {v['blocks']} blocks\n")
                else:
                    sf.write(f"  {k}: {v}\n")
        sf.write(f"\nAccumulated ERROR SUMMARY: {total_errors}\n")

    ending_time = time.time()
    elapsed = ending_time - starting_time

    if proc.returncode != 0:
        print(f'[FAIL] {model} returned exit code {proc.returncode},\t\t{elapsed:.2f} s')
        return True  # failed, erros while model running 
    elif total_errors != 0:
        print(f'[FAIL] Valgrind detected {total_errors} errors for model {model},\t\t{elapsed:.2f} s')
        return True # failed, found memory errors
    else:
        print(f'[OK] profiling is finished for {model},\t\t{elapsed:.2f} s')
        return False # success

def main():
    # check for valgrind availability
    if not shutil.which('valgrind'):
        print('Error: valgrind not found in PATH')
        sys.exit(1)

    # create log folder
    os.makedirs(log_folder, exist_ok=True)

    # run models
    all_ok = True
    for model in valgrind_models:
        result = run_valgrind_for_model(model)
        if result:
            all_ok = False
        
    sys.exit(0 if all_ok else 1)

if __name__ == '__main__':
    main()