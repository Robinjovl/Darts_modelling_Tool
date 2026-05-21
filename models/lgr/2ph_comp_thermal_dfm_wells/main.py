from pathlib import Path

from darts.engines import redirect_darts_output

from model import Model


if __name__ == "__main__":
    case_dir = Path(__file__).resolve().parent
    redirect_darts_output(str(case_dir / "run.log"))

    model = Model()
    model.init()
    model.set_output(output_folder=str(case_dir / "output"))
    model.run(model.runtime)
    model.print_timers()
    model.print_stat()
