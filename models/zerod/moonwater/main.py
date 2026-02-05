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

    ok = m.run(days=1, method="radau")
    if not ok:
        raise RuntimeError("Zerod run failed to reach the final time.")

    m.plot_ph_path(use_log_p=True)
    m.plot_state_history()
