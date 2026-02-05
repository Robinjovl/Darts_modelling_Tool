import sys
from contextlib import contextmanager

from darts.engines import redirect_darts_output

from model import Model


@contextmanager
def tee_stdout(filepath):
    """
    Context manager that redirects stdout to both console and file.

    :param filepath: Path to the log file.
    :type filepath: str
    """
    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    original_stdout = sys.stdout
    try:
        with open(filepath, "w", encoding="utf-8") as log_file:
            sys.stdout = Tee(original_stdout, log_file)
            yield
    finally:
        sys.stdout = original_stdout


if __name__ == "__main__":
    redirect_darts_output("run.log")

    with tee_stdout("simulation.log"):
        m = Model(
            mode="analytical",
            p_init=1e-3,
            t_init=230.0,
            energy_source=2e3,
            fixed_pressure=None,
            sv_init=0.535,
            volume=1.0,
        )
        m.init()

        ok = m.run(days=1, method="radau")
        if not ok:
            raise RuntimeError("Zerod run failed to reach the final time.")

        m.plot_ph_path(use_log_p=True)
        m.plot_state_history()
