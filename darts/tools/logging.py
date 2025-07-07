import os
import sys

original_stdout = os.dup(1)


def redirect_all_output(log_file, append=True):
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


####################################################################


# Logging usage example
if __name__ == "__main__":
    from darts.engines import logging

    logging.log("screen only")
    logging.duplicate_output_to_file("log.log")
    logging.log("screen and log file")
    print("screen only")

    # Logging verbosity example
    # Default logging verbosity level is : INFO
    logging.debug("debug that should not be shown")
    logging.info("info")
    logging.set_logging_level(logging.LoggingLevel.ERROR)
    logging.info("info that should not be shown")
    logging.error("error")
    logging.critical("critical")

    """
    # Results

    ## Screen
    
        screen only
        screen and log file
        screen only
        info
        error
        critical

    ## Log file 'log.log'

        screen and log file
        info
        error
        critical

    """
