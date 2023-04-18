from darts.engines import *
import os, sys, shutil
from pathlib import Path

from multiprocessing import Process, set_start_method, Value

import importlib

import signal

original_stdout = os.dup(1)

def _sigterm_handler():
    print("received SIGABRT")
    sys.exit()

def redirect_all_output(log_file, append = True):
    if append:
        log_stream = open(log_file, "a+")
    else:
        log_stream = open(log_file, "w")
    # this way truly all messages from both Python and C++, printf or std::cout, will be redirected
    os.dup2(log_stream.fileno(), sys.stdout.fileno())
    return log_stream

def abort_redirection(log_stream):
    os.dup2(original_stdout, sys.stdout.fileno())
    log_stream.close()

def spawn_process_function(model_path, model_procedure, ret_value):
    # step in target folder
    os.chdir(model_path)

    # add it also to system path to load modules
    sys.path.append(os.path.abspath(r'.'))

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
    # if __name__ == '__main__':

    set_start_method('spawn')

    # null = open(os.devnull, 'w')
    # orig_stdout = sys.stdout

    # set working directory to folder which contains tests
    os.chdir(root_path)

    p = Path(root_path)
    # iterate over directories in 'root_path'
    parent = p.cwd()
    n_fails = 0
    if len(accepted_paths) != 0:
        for x in accepted_paths:
            # set as failed by default - if model run fails with exception,ret_value remains equal to 1
            ret_value = Value("i", 1, lock=False)
            p = Process(target=spawn_process_function, args=(x, model_procedure,ret_value), )
            p.start()
            p.join(timeout=7200)
            p.terminate()
            n_fails += ret_value.value
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
    return n_fails


def for_each_model_adjoint(root_path, model_procedure, accepted_paths=[], excluded_paths=[], timeout=120):
    # if __name__ == '__main__':

    # set_start_method('spawn')  # it is already spawned in the function "for_each_model"

    # null = open(os.devnull, 'w')
    # orig_stdout = sys.stdout

    # set working directory to folder which contains tests
    os.chdir(root_path)

    p = Path(root_path)
    # iterate over directories in 'root_path'
    parent = p.cwd()
    n_fails = 0
    if len(accepted_paths) != 0:
        for x in accepted_paths:
            # set as failed by default - if model run fails with exception,ret_value remains equal to 1
            ret_value = Value("i", 1, lock=False)
            p = Process(target=spawn_process_function_adjoint, args=(x, model_procedure, ret_value), )
            p.start()
            p.join(timeout=7200)
            p.terminate()
            n_fails += ret_value.value
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
    return n_fails

def run_single_test(dir, module_name, args, ret_value):
    # step in target folder
    os.chdir(dir)

    # add it also to system path to load modules
    sys.path.append(os.path.abspath(r'.'))

    # import model and run it for default time
    try:
        mod = importlib.import_module(module_name)
        # perform required procedures
        print("Running {:<30}".format(dir + ': ' + "_".join(args)), flush=True)
        log_file = os.path.join(os.path.abspath(os.pardir), '_logs/' + str(dir) + '.log')
        f = open(log_file, "a")
        f.close()
        log_stream = redirect_all_output(log_file)
        shutil.rmtree("__pycache__", ignore_errors=True)
        # create model instance
        ret_value.value, test_time = mod.run_test(args)
        log_stream = redirect_all_output(log_file)
        abort_redirection(log_stream)
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


def run_tests(root_path, test_dirs=[], test_args=[]):
    # set_start_method('spawn')

    # set working directory to folder which contains tests
    os.chdir(root_path)

    n_failed = 0
    n_tot = 0
    assert(len(test_dirs) == len(test_args))
    for i, dir in enumerate(test_dirs):
        for arg in test_args[i]:
            # set as failed by default - if model run fails with exception,ret_value remains equal to 1
            ret_value = Value("i", 1, lock=False)
            p = Process(target=run_single_test, args=(dir, 'main', arg, ret_value), )
            p.start()
            p.join(timeout=7200)
            p.terminate()
            n_failed += ret_value.value
            n_tot += 1

    return n_tot, n_failed