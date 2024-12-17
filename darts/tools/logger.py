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
  
####################################################################
  
import logging
import darts.engines as darts_cpp_logger

class CppLoggingHandler(logging.Handler):
    def __init__(self, level = 0) -> None:
        super().__init__(level)

    def emit(self, record):
        darts_cpp_logger.log(f"{record.levelname}: {record.getMessage()}")


logger = logging.getLogger("Darts")
cpp_handler = CppLoggingHandler()

logger.addHandler(cpp_handler)
logger.setLevel(logging.DEBUG)


if __name__ == "__main__": 
    darts_cpp_logger.duplicate_output_to_file("log.log")
    logger.info("yes")
    print("pizza")
    logger.error("no")
