import os, sys
original_stdout = os.dup(1)

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
    
