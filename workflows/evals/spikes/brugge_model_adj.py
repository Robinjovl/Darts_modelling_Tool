"""Uniform_Brugge CI model with the OptModuleSettings mixin, for the adjoint feasibility check."""

from model import Model as BruggeModel

from darts.models.darts_model import DartsModel
from darts.models.opt.opt_module_settings import OptModuleSettings


class Model(BruggeModel, OptModuleSettings):
    """Brugge proxy run in report-step segments so that ``engine.report()`` populates report data."""

    def __init__(self, horizon_days: float, report_step_days: float):
        BruggeModel.__init__(self)
        OptModuleSettings.__init__(self)
        self.T = horizon_days
        self.report_step = report_step_days

    def set_solver(self):
        super().set_solver()
        self.ts_control.runtime = self.T

    def run(self, *args, **kwargs):
        n_full = int(self.T / self.report_step)
        steps = [self.report_step] * n_full
        remainder = self.T - n_full * self.report_step
        if remainder > 1e-12:
            steps.append(remainder)
        for step in steps:
            DartsModel.run(
                self, step, save_well_data=False, save_reservoir_data=False, verbose=0
            )
            self.physics.engine.report()
