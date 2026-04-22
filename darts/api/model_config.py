"""Aggregate ``ModelConfig`` root — single source of truth for a darts model.

``ModelConfig`` is the Pydantic aggregate that replaces the legacy
``darts.input.input_data.InputData`` god-struct.  A model now consumes ONE
configuration object that composes all sub-configs (reservoir, physics,
wells, initial conditions, sim params, output, extensions) instead of
mutating a free-form attribute bag.

Both Python workflows (``ModelConfig(reservoir=..., physics=..., ...)``) and
JSON/MCP workflows (``ModelConfig.model_validate(json_payload)``) end up
with the same validated object, which ``DartsModel.configure()`` then
applies to a model instance.

The ``physics`` field is a discriminated union on ``kind``:
``black_oil | compositional | dead_oil | geothermal``.  Each concrete
physics Config carries a literal ``kind`` tag for Pydantic's dispatch.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from darts.models.darts_model import (
    ExtensionsConfig,
    InitialConditionsConfig,
    OutputConfig,
    SimParamsConfig,
    WellsConfig,
)
from darts.physics.blackoil import BlackOilConfig
from darts.physics.deadoil import DeadOilConfig
from darts.physics.geothermal.geothermal import GeothermalConfig
from darts.physics.super.physics import CompositionalConfig
from darts.reservoirs.struct_reservoir import StructReservoirConfig

# Discriminated union of known physics Configs.  Each concrete Config
# declares a literal ``kind`` field; Pydantic dispatches JSON payloads to
# the right Config by ``kind``.
PhysicsConfig = Annotated[
    BlackOilConfig | CompositionalConfig | DeadOilConfig | GeothermalConfig,
    Field(discriminator="kind"),
]


class ModelConfig(BaseModel):
    """Single-root aggregate configuration for a darts model.

    Replaces ``darts.input.input_data.InputData``.  Every sub-config is a
    typed Pydantic model with ``extra="forbid"`` so JSON workflows fail
    fast on typos and no undocumented fields leak into live models.
    """

    model_config = ConfigDict(extra="forbid")

    reservoir: StructReservoirConfig = Field(
        description="Reservoir grid + geomechanics configuration"
    )
    physics: PhysicsConfig = Field(
        description="Physics family and OBL/PVT parameters (union on 'kind')"
    )
    wells: WellsConfig | None = Field(
        default=None,
        description="Well perforations and control schedule (optional — "
        "single-cell sandbox models may omit)",
    )
    initial_conditions: InitialConditionsConfig | None = Field(
        default=None, description="Initial state (pressure, compositions, …)"
    )
    sim_params: SimParamsConfig | None = Field(
        default=None, description="Solver and timestepping parameters"
    )
    output: OutputConfig | None = Field(
        default=None, description="Output folder, precision, initial-state toggle"
    )
    extensions: ExtensionsConfig = Field(
        default_factory=ExtensionsConfig,
        description="Typed escape hatch for model-specific state (geom, "
        "stress, fracture, other)",
    )


def load_model_config(path: str | Any) -> ModelConfig:
    """Load and validate a ``ModelConfig`` from a JSON file path or open file.

    :param path: filesystem path or open file handle to JSON
    :type path: str | pathlib.Path | IO
    :return: validated ModelConfig
    :rtype: ModelConfig
    """
    import json
    from pathlib import Path

    if isinstance(path, str | Path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = json.load(path)
    return ModelConfig.model_validate(raw)


__all__ = [
    "ModelConfig",
    "PhysicsConfig",
    "load_model_config",
]
