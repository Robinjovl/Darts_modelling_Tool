from darts.engines import redirect_darts_output

from model import Model


if __name__ == "__main__":
    redirect_darts_output("run.log")

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

    dt = 1e-4
    n_steps = 200
    ok = m.run(days=dt * n_steps, method="radau", restart_dt=dt)
    if not ok:
        raise RuntimeError("Zerod run failed to reach the final time.")

    m.plot_ph_path(use_log_p=True)
