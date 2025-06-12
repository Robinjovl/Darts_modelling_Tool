import os
import sys
import subprocess
import re
import shutil
import importlib

valgrind_models = [
                    '2ph_comp',
                    'Chem_benchmark_new',
                    'GeoRising'
                   ]

# if the user passed extra models, append them (comma-separated):
extra = os.environ.get('APPEND_VALGRIND_MODEL')
if extra:
    for m in extra.split(','):
        m = m.strip()
        if m and m not in valgrind_models:
            valgrind_models.append(m)

log_folder = os.path.join('models', '_valgrind_logs')

# NOT USED: valgrind suppression file (adjust path as needed)
# suppression_file = 'helper_scripts/valgrind-python.supp'

def run_model(model, days):
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
        m.run(days=days, save_well_data=False, save_reservoir_data=False)
        print(f"[OK] {model}")
        success = True
    except Exception as e:
        print(f"[FAIL] {model} → {e}")
        success = False

    return success

# patterns to extract leak info and errors from Valgrind log
MSG_PATTERNS = {
    'definitely_lost':    r'definitely lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'indirectly_lost':    r'indirectly lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'possibly_lost':      r'possibly lost:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'still_reachable':    r'still reachable:\s+([\d,]+) bytes in ([\d,]+) blocks',
    'error_summary':      r'ERROR SUMMARY:\s+(\d+)'  # total errors
}

def analyze_log(log_path):
    """Parse Valgrind log and return a dict of summary values."""
    summary = {}
    try:
        content = open(log_path, 'r').read()
    except FileNotFoundError:
        return {'error': 'Log file not found'}

    ret_code = 0
    for key, pattern in MSG_PATTERNS.items():
        m = re.search(pattern, content)
        if m:
            if key == 'error_summary':
                ret_code = int(m.group(1))
                summary[key] = ret_code
            else:
                summary[key] = { 'bytes': m.group(1), 'blocks': m.group(2) }
    return summary, ret_code

def run_valgrind_for_model(model):
    # model path
    model_path = os.path.join('models', model)
    if not os.path.isdir(model_path):
        print(f"[SKIP] Model directory not found: {model_path}")
        return True # failed

    # file paths
    vg_log = os.path.join(log_folder, f'{model}.vg.log')
    prog_out = os.path.join(log_folder, f'{model}_mainpy.log')
    prog_err = os.path.join(log_folder, f'{model}_mainpy_err.log')
    summary_file = os.path.join(log_folder, f'{model}.summary.txt')

    # Build the inline Python snippet to invoke run_model_direct
    days = 1
    py_snippet = (
        'import sys, os; '
        'from helper_scripts.valgrind_profile import run_model; '
        f'model_path = os.path.join("models", "{model}"); '
        'os.chdir(model_path); '
        f'sys.exit(0 if run_model(model="{model}", days={days}) else 1)'
    )

    # valgrind command profiling run_model_direct
    cmd = [
        'valgrind',
        #'--quiet',
        '--error-exitcode=0',
        f'--log-file={vg_log}',
        '--',
        'python', '-c', py_snippet
    ]

    print(f'Running Valgrind for model {model}...')

    # set environment: disable pymalloc so Valgrind sees everything
    env = os.environ.copy()
    env['PYTHONMALLOC'] = 'malloc'
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
            proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, env=env, timeout=1800)
            # proc = subprocess.run(cmd, shell=True, env=env, timeout=1800)
        except subprocess.TimeoutExpired:
            print(f'ERROR: timeout profiling model {model}')
            return True # failed

    # analyze and write summary
    summary, ret_code = analyze_log(vg_log)
    with open(summary_file, 'w') as sf:
        sf.write(f'Valgrind summary for model: {model}\n')
        for k, v in summary.items():
            if isinstance(v, dict):
                sf.write(f'  {k}: {v["bytes"]} bytes in {v["blocks"]} blocks\n')
            else:
                sf.write(f'  {k}: {v}\n')

    if ret_code != 0:
        print(f'Valgrind detected errors for model {model}, exit code {ret_code}.')
    
    if proc.returncode != 0:
        print(f'[FAIL] {model} returned exit code {proc.returncode}.')
        return True  # failed
    else:
        print(f'[OK] profiling is finished for {model}.')
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