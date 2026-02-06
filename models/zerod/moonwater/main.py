import sys
from contextlib import contextmanager
import os
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


def run_model(mode: str = 'analytical',
                p_init: float = 1e-5,
                energy_source: float = 1e5,
                sv_init: float = 0.535,
                fixed_pressure: bool = False,
                n_pres_points: int = 10000000,
                n_enth_points: int = 10000000,
                max_ts: float = 1e-3):
    if mode == 'analytical':
        output_folder = f'output_{mode}_p_{p_init}_q_{energy_source}_sv_{sv_init}'
    elif mode == 'obl':
        output_folder = f'output_{mode}_p_{p_init}_q_{energy_source}_sv_{sv_init}_n_pres_{n_pres_points}_n_enth_{n_enth_points}'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    redirect_darts_output(os.path.join(output_folder, 'log.txt'))

    with tee_stdout(os.path.join(output_folder, 'simulation.log')):
        m = Model(
            mode=mode,
            p_init=p_init,
            energy_source=energy_source,
            sv_init=sv_init,
            fixed_pressure=fixed_pressure,
            n_pres_points=n_pres_points,
            n_enth_points=n_enth_points,
            max_ts=max_ts,
        )
        m.init()

        ok = m.run(days=1.0, method="radau")
        if not ok:
            raise RuntimeError("Zerod run failed to reach the final time.")

        ph_path_file = os.path.join(output_folder, 'ph_diagram.png')
        m.plot_ph_path(use_log_p=True, output_path=ph_path_file)
        # PT flash is not initialized
        # pt_path_file = os.path.join(output_folder, 'pt_diagram.png')
        # m.plot_pt_path(use_log_p=True, output_path=pt_path_file)
        state_history_file = os.path.join(output_folder, 'state_history.png')
        m.plot_state_history(use_log_p=True, output_path=state_history_file)

if __name__ == "__main__":
    # analytical derivatives
    # run_model(mode='analytical', p_init=1e-3, energy_source=1e5, sv_init=0.535, fixed_pressure=False)

    # OBL derivatives
    run_model(mode='obl', p_init=1e-3, energy_source=1e5, sv_init=0.535, fixed_pressure=False,
                n_pres_points=1000000000, n_enth_points=1000000000, max_ts=1e-4)
