"""3ph_bo — SPE-style 3-phase black-oil benchmark (10x10x3 grid).

Migrated to the ``ModelConfig`` / ``build_config()`` pattern: no InputData,
no ``set_input_data()`` override.  The model script declares its
configuration up front and passes it to the physics family directly.
"""

import numpy as np
from darts.engines import well_control_iface

from darts.api.model_config import ModelConfig
from darts.models.cicd_model import CICDModel
from darts.models.darts_model import (
    InitialConditionsConfig,
    OutputConfig,
    SimParamsConfig,
    WellConfig,
    WellPerforationConfig,
    WellsConfig,
)
from darts.physics.blackoil import BlackOil, BlackOilConfig
from darts.reservoirs.struct_reservoir import StructReservoir, StructReservoirConfig


class Model(CICDModel):
    """3-phase black-oil 10x10x3 layered grid, one injector and one producer."""

    def __init__(self):
        """Construct the 3ph_bo model and apply its ModelConfig.

        :return: None
        """
        super().__init__()

        self.timer.node["initialization"].start()

        self.config = self.build_config()
        self.set_reservoir()
        self.set_physics()
        # configure() applies sim_params which needs self.physics.n_vars —
        # safe to call now that physics is constructed.
        self.configure(self.config)

        self.set_sim_params(
            first_ts=1e-6,
            mult_ts=2,
            max_ts=10,
            runtime=100,
            tol_newton=1e-3,
            tol_linear=1e-7,
            it_newton=10,
            it_linear=50,
        )

        self.timer.node["initialization"].stop()

    # ------------------------------------------------------------------
    # Declarative configuration
    # ------------------------------------------------------------------

    def build_config(self) -> ModelConfig:
        """Return a fully-populated :class:`ModelConfig` for this model.

        Replaces the legacy ``set_input_data()`` override.
        """
        reservoir_cfg = StructReservoirConfig(
            type="structured",
            nx=10,
            ny=10,
            nz=3,
            dx=3000.0 / 10,
            dy=3000.0 / 10,
            dz=[6.0] * 100 + [9.0] * 100 + [15.0] * 100,
            permx=[500.0] * 100 + [50.0] * 100 + [200.0] * 100,
            permy=[500.0] * 100 + [50.0] * 100 + [200.0] * 100,
            permz=[80.0] * 100 + [42.0] * 100 + [20.0] * 100,
            poro=0.3,
            depth=[2540.0] * 100 + [2548.0] * 100 + [2560.0] * 100,
        )
        physics_cfg = BlackOilConfig(
            pvt_path="physics.in",
            thermal=False,
            n_points=5001,
            zero=1e-12,
            epsilon_z=1e-13,
            min_p=1.0,
            max_p=450.0,
            min_t=-10.0,
            max_t=100.0,
            min_z=0.0,
            max_z=1.0,
        )
        wells_cfg = WellsConfig(
            wells=[
                WellConfig(
                    name="I1",
                    perforations=[WellPerforationConfig(ijk=[1, 1, 1])],
                ),
                WellConfig(
                    name="P1",
                    perforations=[WellPerforationConfig(ijk=[10, 10, 3])],
                ),
            ]
        )
        return ModelConfig(
            reservoir=reservoir_cfg,
            physics=physics_cfg,
            wells=wells_cfg,
            initial_conditions=InitialConditionsConfig(
                by_array={
                    "pressure": 330.0,
                    "z0": 0.001225901537,
                    "z1": 0.7711341309,
                }
            ),
            sim_params=SimParamsConfig(
                first_ts=1e-6,
                mult_ts=2.0,
                max_ts=10.0,
                runtime=100.0,
                tol_newton=1e-3,
                tol_linear=1e-7,
                it_newton=10,
                it_linear=50,
            ),
            output=OutputConfig(),
        )

    # ------------------------------------------------------------------
    # Reservoir / physics / wells hooks (called by DartsModel lifecycle)
    # ------------------------------------------------------------------

    def set_reservoir(self):
        """Build the StructReservoir from the Config's reservoir section.

        :return: None
        """
        c = self.config.reservoir
        self.reservoir = StructReservoir(
            self.timer,
            nx=c.nx,
            ny=c.ny,
            nz=c.nz,
            dx=np.array(c.dx) if isinstance(c.dx, list) else c.dx,
            dy=np.array(c.dy) if isinstance(c.dy, list) else c.dy,
            dz=np.array(c.dz) if isinstance(c.dz, list) else c.dz,
            permx=np.array(c.permx) if isinstance(c.permx, list) else c.permx,
            permy=np.array(c.permy) if isinstance(c.permy, list) else c.permy,
            permz=np.array(c.permz) if isinstance(c.permz, list) else c.permz,
            poro=c.poro,
            depth=np.array(c.depth) if isinstance(c.depth, list) else c.depth,
        )

    def set_wells(self):
        """Add the two wells + their perforations from the Config.

        Called by ``DartsModel.init()`` after the reservoir is built.

        :return: None
        """
        if self.config.wells is not None:
            self.set_wells_from_dict(self.config.wells.model_dump())

    def set_physics(self):
        """Instantiate BlackOil physics from the Config's physics section.

        :return: None
        """
        self.physics = BlackOil(self.config.physics, self.timer)
        zero = 1e-12
        self.inj_composition = [1 - 2 * zero, zero]
        self.ini_stream = [0.001225901537, 0.7711341309]

    def set_initial_conditions(self):
        """Apply initial-condition arrays to the mesh via the physics.

        :return: result of ``physics.set_initial_conditions_from_array``
        :rtype: Any
        """
        input_distribution = {
            self.physics.vars[0]: 330.0,
            self.physics.vars[1]: self.ini_stream[0],
            self.physics.vars[2]: self.ini_stream[1],
        }
        return self.physics.set_initial_conditions_from_array(
            mesh=self.reservoir.mesh, input_distribution=input_distribution
        )

    def set_well_controls(self):
        """Apply injector/producer BHP controls for the two wells.

        :return: None
        """
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.BHP,
                    is_inj=True,
                    target=400.0,
                    inj_composition=self.inj_composition,
                )
            else:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=70.0,
                )
