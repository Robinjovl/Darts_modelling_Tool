import os
import sys
import subprocess
import shutil
import importlib
import time
import glob

# List of models to profile with VTune
vtune_models = [
    'SPE11b'
]

# if the user passed extra models, append them (comma-separated):
extra = os.environ.get('APPEND_VTUNE_MODEL')
if extra:
    for m in extra.split(','):
        m = m.strip()
        if m and m not in vtune_models:
            vtune_models.append(m)

# folder with logs to be reported
log_folder = os.path.join('models', '_vtune_logs')

def run_vtune_for_model(model, timeout=3600):
    """Run VTune profiling for a specific model."""
    # model path
    model_path = os.path.join('models', model)
    if not os.path.exists(model_path):
        print(f"[SKIP] Model directory not found: {model_path}")
        return True  # failed

    # file paths
    vtune_result_dir = os.path.join(log_folder, f'vtune_mem_{model}')
    prog_out = os.path.join(log_folder, f'{model}.log')
    prog_err = os.path.join(log_folder, f'{model}_err.log')

    print(f'Running VTune profiling for model {model}...')
    starting_time = time.time()

    # VTune collection command - following user's specification
    vtune_collect_cmd = [
        'vtune',
        '-collect', 'hotspots',
        '-data-limit=0',
        '-r', vtune_result_dir,
        'darts', 'main.py'
    ]

    # Set environment
    env = os.environ.copy()
    # env['OMP_NUM_THREADS'] = '1'

    # Get DARTS path for library path
    try:
        darts_path = subprocess.check_output([
            'python', '-c',
            'import os, darts; print(os.path.dirname(darts.__file__))'
        ]).decode().strip()
        env['LD_LIBRARY_PATH'] = f"{darts_path}:{env.get('LD_LIBRARY_PATH','')}"
    except subprocess.CalledProcessError as e:
        print(f'Failed to determine DARTS path: {e}')
        return True  # failed

    # Run VTune collection
    with open(prog_out, 'w') as out_f, open(prog_err, 'w') as err_f:
        try:
            proc = subprocess.run(vtune_collect_cmd, stdout=out_f, stderr=err_f, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f'ERROR: timeout profiling model {model}')
            return True  # failed

    # Generate reports if collection was successful
    if proc.returncode == 0 and os.path.exists(vtune_result_dir):
        print(f'Generating VTune reports for {model}...')
        
        # Summary report
        summary_cmd = [
            'vtune',
            '-report', 'summary',
            '-report-knob', 'show-issues=true',
            '-report-output', os.path.join(log_folder, f'{model}_summary.csv'),
            '-format', 'csv',
            '-r', vtune_result_dir
        ]

        # Hotspots report - following user's specification
        hotspots_cmd = [
            'vtune',
            '-report', 'hotspots',
            '-report-knob', 'show-issues=true',
            '-report-output', os.path.join(log_folder, f'{model}_hotspots.csv'),
            '-format', 'csv',
            '-r', vtune_result_dir
        ]
        
        # Top-down report - following user's specification
        topdown_cmd = [
            'vtune',
            '-report', 'top-down',
            '-report-output', os.path.join(log_folder, f'{model}_topdown.csv'),
            '-format', 'csv',
            '-r', vtune_result_dir
        ]
        
        try:
            # Run summary report
            subprocess.run(summary_cmd, check=True, timeout=300)
            print(f'Generated summary report for {model}')

            # Run hotspots report
            subprocess.run(hotspots_cmd, check=True, timeout=300)
            print(f'Generated hotspots report for {model}')
            
            # Run top-down report
            subprocess.run(topdown_cmd, check=True, timeout=300)
            print(f'Generated top-down report for {model}')
            
        except subprocess.CalledProcessError as e:
            print(f'Failed to generate reports for {model}: {e}')
        except subprocess.TimeoutExpired:
            print(f'Timeout generating reports for {model}')

    ending_time = time.time()
    elapsed = ending_time - starting_time

    if proc.returncode != 0:
        print(f'[FAIL] {model} returned exit code {proc.returncode},\t\t{elapsed:.2f} s')
        return True  # failed, errors while model running
    else:
        print(f'[OK] VTune profiling finished for {model},\t\t{elapsed:.2f} s')
        return False  # success

def main():
    # check for vtune availability
    if not shutil.which('vtune'):
        print('Error: vtune not found in PATH')
        sys.exit(1)

    # create log folder
    os.makedirs(log_folder, exist_ok=True)

    # run models
    all_ok = True
    for model in vtune_models:
        result = run_vtune_for_model(model)
        if result:
            all_ok = False
        
    sys.exit(0 if all_ok else 1)

if __name__ == '__main__':
    main() 