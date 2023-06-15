from darts.engines import *
from darts.physics import *
import os, sys
from pathlib import Path

from multiprocessing import Process, set_start_method

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

def spawn_process_function(model_path, model_procedure):
    # step in target folder
    os.chdir(model_path)

    # add it also to system path to load modules
    sys.path.append(os.path.abspath(r'.'))

    # import model and run it for default time
    try:
        # import model.py
        mod = importlib.import_module('model')

        # perform required procedures
        model_procedure(mod)

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
    for x in p.iterdir():
        if x.is_dir() and (str(x)[0] != '.'):
            if len(accepted_paths) != 0:
                if str(x) in accepted_paths:
                    p = Process(target=spawn_process_function, args=(x, model_procedure), )
                    p.start()
                    p.join(timeout=7200)
                    p.terminate()
            else:
                if len(excluded_paths) != 0:
                    if x in excluded_paths:
                        continue
                p = Process(target=spawn_process_function, args=(x, model_procedure), )
                p.start()
                p.join(timeout=7200)
                p.terminate()
