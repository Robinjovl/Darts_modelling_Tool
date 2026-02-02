from darts.engines import *
from darts.tools.logging import redirect_all_output, abort_redirection
import os, sys, shutil
from pathlib import Path

from multiprocessing import Process, set_start_method, Value
import time
import importlib

import signal

original_stdout = os.dup(1)

def _ensure_parent_dir(path):
    """Create parent directory for the provided file path if missing."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def _sigterm_handler():
    print("received SIGABRT")
    sys.exit()

def spawn_process_function(model_path, model_procedure, ret_value):
    # step in target folder
    os.chdir(model_path)

    # add it also to system path to load modules
    sys.path.insert(0, os.path.abspath(r'.'))

    # import model and run it for default time
    try:
        # import model.py
        mod = importlib.import_module('model')
        # perform required procedures
        ret_value.value = model_procedure(mod)

    except Exception as err:
        # sys.stdout = orig_stdout
        print(err)

def spawn_process_function_adjoint(model_path, model_procedure, ret_value):
    # step in target folder
    os.chdir(model_path)

    # add it also to system path to load modules
    sys.path.append(os.path.abspath(r'.'))

    # import model and run it for default time
    try:
        # import model.py
        mod = importlib.import_module('adjoint_definition')
        # perform required procedures
        ret_value.value = model_procedure(mod)

    except Exception as err:
        # sys.stdout = orig_stdout
        print(err)

def for_each_model(root_path, model_procedure, accepted_paths=[], excluded_paths=[], timeout=120):
    set_start_method('spawn')

    # set working directory to folder which contains tests
    os.chdir(root_path)

    p = Path(root_path)
    # iterate over directories in 'root_path'
    parent = p.cwd()
    failed = []
    if len(accepted_paths) != 0:
        for x in accepted_paths:
            if 'mpfa' in  x:
                nt = os.environ['OMP_NUM_THREADS']
                os.environ['OMP_NUM_THREADS'] = '1'
            # set as failed by default - if model run fails with exception,ret_value remains equal to 1
            ret_value = Value("i", 1, lock=False)
            p = Process(target=spawn_process_function, args=(x, model_procedure,ret_value), )
            p.start()
            p.join(timeout=7200)
            p.terminate()
            if ret_value.value:
                failed.append(x)
            if 'mpfa' in x:
                os.environ['OMP_NUM_THREADS'] = nt
    else:
        for x in p.iterdir():
            if x.is_dir() and (str(x)[0] != '.'):
                if len(excluded_paths) != 0:
                    if str(x) in excluded_paths:
                        continue
                p = Process(target=spawn_process_function, args=(str(x), model_procedure), )
                p.start()
                p.join(timeout=7200)
                p.terminate()
    return failed


def for_each_model_adjoint(root_path, model_procedure, accepted_paths=[], excluded_paths=[], timeout=120):

    # set working directory to folder which contains tests
    os.chdir(root_path)

    p = Path(root_path)
    # iterate over directories in 'root_path'
    parent = p.cwd()
    failed = []
    if len(accepted_paths) != 0:
        for x in accepted_paths:
            if 'mpfa' in  x:
                nt = os.environ['OMP_NUM_THREADS']
                os.environ['OMP_NUM_THREADS'] = '1'
            # set as failed by default - if model run fails with exception,ret_value remains equal to 1
            ret_value = Value("i", 1, lock=False)
            starting_time = time.time()
            p = Process(target=spawn_process_function_adjoint, args=(x, model_procedure, ret_value), )
            p.start()
            p.join(timeout=7200)
            p.terminate()
            if ret_value.value:
                failed.append(x)
            ending_time = time.time()
            if not ret_value.value:
                print('OK, \t%.2f s' % (ending_time - starting_time))
            else:
                print('FAIL, \t%.2f s' % (ending_time - starting_time))
            if 'mpfa' in  x:
                os.environ['OMP_NUM_THREADS'] = nt
    else:
        for x in p.iterdir():
            if x.is_dir() and (str(x)[0] != '.'):
                if len(excluded_paths) != 0:
                    if str(x) in excluded_paths:
                        continue
                p = Process(target=spawn_process_function_adjoint, args=(str(x), model_procedure), )
                p.start()
                p.join(timeout=7200)
                p.terminate()
    return failed

def run_single_test(dir, module_name, args, ret_value, platform):

    # step in target folder
    os.chdir(dir)

    # add it also to system path to load modules
    # sys.path.append(os.path.abspath(r'.'))
    sys.path.insert(0, os.path.abspath(r'.'))

    # import model and run it for default time
    try:
        mod = importlib.import_module(module_name)
        if isinstance(args, dict) and 'name' in args:
            args_str = str(args['name'])
            del args['name']
        else:
            try:  # model-provided formatter
                args_str = mod.get_output_folder(args)
            except Exception:
                if isinstance(args, (list, tuple)) and len(args) > 0:
                    args_str = '_'.join(map(str, args[:-1])) if len(args) > 1 else str(args[0])
                elif isinstance(args, dict):
                    if 0 in args:
                        args_str = str(args[0])
                    else:
                        args_str = '_'.join(f'{k}-{v}' for k, v in args.items())
                else:
                    args_str = 'run'
        args_str = args_str.replace(os.sep, '_')
        # perform required procedures
        print("Running {:<30}".format(dir + ': ' + args_str), flush=True)
        log_file = os.path.join(os.path.join(os.path.abspath(os.pardir), '_logs'),
                                str(dir) + '_' + args_str + '.log')
        _ensure_parent_dir(log_file)
        f = open(log_file, 'w')
        f.close()
        #log_stream = redirect_all_output(log_file)
        shutil.rmtree("__pycache__", ignore_errors=True)
        # create model instance
        ret_value.value, test_time = mod.run_test(args, platform=platform)
        ###log_stream = redirect_all_output(log_file)
        #abort_redirection(log_stream)
        if ret_value.value:
            print('FAIL, \t%.2f s' % test_time)
        else:
            if test_time > 0.0:
                print('OK, \t%.2f s' % test_time)
            else:
                print('SAVED')
    except Exception as err:
        # sys.stdout = orig_stdout
        print(dir)
        print(err)


def run_tests(root_path, test_dirs=[], test_args=[], overwrite='0', platform='cpu'):
    # set working directory to folder which contains tests
    os.chdir(root_path)

    logs_folder = os.path.join(os.path.abspath(os.pardir), '_logs')
    os.makedirs(logs_folder, exist_ok=True)

    failed = []
    n_tot = 0
    assert(len(test_dirs) == len(test_args))
    for i, dir in enumerate(test_dirs):
        for arg in test_args[i]:
            # set as failed by default - if model run fails with exception,ret_value remains equal to 1
            ret_value = Value("i", 1, lock=False)

            # erase previous log file if existed
            if isinstance(arg, (list, tuple)) and len(arg) > 0:
                arg_label = str(arg[0])
            elif isinstance(arg, dict):
                if 0 in arg:
                    arg_label = str(arg[0])
                elif 'name' in arg:
                    arg_label = str(arg['name'])
                else:
                    arg_label = '_'.join(f'{k}-{v}' for k, v in arg.items())
            else:
                arg_label = 'run'
            log_file = os.path.join(logs_folder, str(dir) + '_' + arg_label + '.log')
            _ensure_parent_dir(log_file)
            f = open(log_file, "w")
            f.close()
            #log_stream = redirect_all_output(log_file)
            starting_time = time.time()
            arg_o = arg + [overwrite] if type(arg) == list else arg  # add overwrite [pkl] flag if a list
            p = Process(target=run_single_test, args=(dir, 'main', arg_o, ret_value, platform), )
            p.start()
            p.join(timeout=7200)
            p.terminate()
            #abort_redirection(log_stream)
            ending_time = time.time()
            str_status = 'OK' if not ret_value.value else 'FAIL'
            if isinstance(arg, list):
                arg_label_print = '_'.join(map(str, arg))
            elif isinstance(arg, dict):
                if 'name' in arg:
                    arg_label_print = str(arg['name'])
                else:
                    arg_label_print = '_'.join(f'{k}-{v}' for k, v in arg.items())
            else:
                arg_label_print = str(arg)
            print('Test ' + dir + ' ' + arg_label_print + ': ' + str_status + ', \t%.2f s' % (ending_time - starting_time))

            if ret_value.value:
                failed.append(dir + ' ' + arg_label_print)
            n_tot += 1

    return n_tot, failed
