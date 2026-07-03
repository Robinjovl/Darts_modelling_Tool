"""Typed configuration classes for geomechanics (poro-/thermoporoelastic) models.

These replace the free-form ``darts.input.input_data.InputData`` god-struct
(and its ``RockProps`` / ``FluidProps`` / ``OBLParams`` / ``InitialSolution`` /
``MeshData`` / ``OtherProps`` sub-bags) that the poromechanics stack used to
consume.  :class:`MechModelConfig` is the single aggregate a geomechanics model
builds and hands to :class:`darts.reservoirs.unstruct_reservoir_mech.UnstructReservoirMech`
and :class:`darts.models.thmc_model.THMCModel`, exactly where the old ``idata``
object went.

Unlike the flow configs (``GeothermalConfig``, ``DeadOilConfig``, ...), the
mechanics configs are deliberately **mutable** and permit arbitrary objects:
rock material fields are polymorphic ``scalar | numpy array | Stiffness`` values,
some are broadcast from scalars to per-region arrays at runtime
(:meth:`MechRockConfig.make_region_arrays`), and a few models overwrite rock
arrays in place after mesh interpolation.  ``validate_assignment`` is therefore
off and ``arbitrary_types_allowed`` is on, so the configs stay typed containers
(named fields, defaults, ``extra="forbid"``) without fighting the numerics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# A rock/fluid property value: a scalar, a python list, or a numpy array
# (per-region or per-cell).  Kept permissive because the poromechanics engines
# accept heterogeneous per-cell arrays as readily as uniform scalars.
ScalarOrArray = float | int | list | np.ndarray

# Shared model_config: typed but mutable, arbitrary objects allowed (numpy
# arrays, Stiffness tensors, well_control_iface enums), no re-validation on
# in-place assignment, and unknown fields rejected so typos fail fast.
_MECH_MODEL_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    validate_assignment=False,
    extra="forbid",
)


class MechRockConfig(BaseModel):
    """Rock (matrix) material properties for poro-/thermoporoelasticity.

    Replaces the legacy ``RockProps`` bag.  Every field defaults to ``None`` and
    accepts a scalar or a per-region/per-cell array; mechanics tensors
    (``perm``, ``biot``, ``th_expn``, ``thermal_conductivity``) may be a flat
    9-element (3x3) or full tensor, and ``stiffness`` may be a ``Stiffness``
    object or a Voigt list.
    """

    model_config = _MECH_MODEL_CONFIG

    porosity: ScalarOrArray | None = Field(None, description="Matrix porosity [-]")
    perm: ScalarOrArray | None = Field(
        None, description="Isotropic perm scalar or full permeability tensor [mD]"
    )
    permx: ScalarOrArray | None = Field(None, description="Permeability x [mD]")
    permy: ScalarOrArray | None = Field(None, description="Permeability y [mD]")
    permz: ScalarOrArray | None = Field(None, description="Permeability z [mD]")
    compressibility: ScalarOrArray | None = Field(
        None, description="Rock compressibility [1/bar]"
    )
    density: ScalarOrArray | None = Field(None, description="Rock density [kg/m3]")
    E: ScalarOrArray | None = Field(None, description="Young modulus [bar]")
    nu: ScalarOrArray | None = Field(None, description="Poisson ratio [-]")
    kd: ScalarOrArray | None = Field(None, description="Drained bulk modulus [bar]")
    stiffness: Any = Field(
        None, description="Stiffness tensor (Stiffness object or Voigt list)"
    )
    biot: ScalarOrArray | None = Field(None, description="Biot coefficient/tensor [-]")
    heat_capacity: ScalarOrArray | None = Field(
        None, description="Volumetric heat capacity [kJ/m3/K]"
    )
    thermal_conductivity: ScalarOrArray | None = Field(
        None, description="Thermal conductivity scalar/tensor [kJ/m/day/K]"
    )
    th_expn: ScalarOrArray | None = Field(
        None, description="Thermal expansion coefficient/tensor"
    )
    th_expn_poro: ScalarOrArray | None = Field(
        None, description="Porosity thermal-expansion term"
    )
    # Optional non-reservoir (proxy / burden) material fields used by some models.
    poro_non_rsv: ScalarOrArray | None = Field(
        None, description="Porosity of non-reservoir (burden) layers [-]"
    )
    perm_non_rsv: ScalarOrArray | None = Field(
        None, description="Permeability of non-reservoir (burden) layers [mD]"
    )
    E_non_rsv: ScalarOrArray | None = Field(
        None, description="Young modulus of non-reservoir (burden) layers [bar]"
    )
    th_expn_orig: ScalarOrArray | None = Field(
        None, description="Original (linear) thermal expansion kept for proxy models"
    )

    def get_permxyz(self):
        """Return the permeability as ``(kx, ky, kz)`` or a full tensor.

        :return: ``(permx, permy, permz)`` when a full tensor is not set, the
            triple ``(perm, perm, perm)`` for an isotropic scalar ``perm``, or
            the ``perm`` tensor otherwise.
        :rtype: tuple | Any
        """
        if self.perm is None:
            return self.permx, self.permy, self.permz
        if np.isscalar(self.perm):
            return self.perm, self.perm, self.perm
        return self.perm

    def make_region_arrays(self, skip=("compressibility", "density")):
        """Broadcast scalar rock fields to per-region arrays in place.

        Reproduces the legacy ``InputData.make_prop_arrays`` behaviour: if any
        rock field is already an array, its size defines the number of regions
        and every other non-``None`` scalar field (except those in ``skip``) is
        replaced by a uniform array of that length.

        :param skip: field names to leave as scalars.
        :type skip: tuple[str, ...]
        :return: ``None``; fields are mutated in place.
        """
        max_n_regions = 1
        for name in type(self).model_fields:
            value = getattr(self, name)
            if value is not None and not np.isscalar(value):
                max_n_regions = np.asarray(value).size
        for name in type(self).model_fields:
            if name in skip:
                continue
            value = getattr(self, name)
            if value is None or not np.isscalar(value):
                continue
            setattr(self, name, np.zeros(max_n_regions, dtype=type(value)) + value)


class MechFluidConfig(BaseModel):
    """Single-phase fluid properties for the poromechanics property container.

    Replaces the legacy ``FluidProps`` bag.
    """

    model_config = _MECH_MODEL_CONFIG

    compressibility: float | None = Field(
        None, description="Fluid compressibility [1/bar]"
    )
    density: float | None = Field(
        None, description="Fluid density at reference [kg/m3]"
    )
    viscosity: float | None = Field(None, description="Fluid viscosity [cP]")
    Mw: float | None = Field(None, description="Molar weight [kg/kmol]")
    heat_capacity: float | None = Field(
        None, description="Fluid heat capacity [kJ/kmol/K]"
    )
    thermal_conductivity: float | None = Field(
        None, description="Fluid thermal conductivity [kJ/m/day/K]"
    )


class MechOBLConfig(BaseModel):
    """OBL (operator-based linearization) grid range and resolution.

    Replaces the legacy ``OBLParams`` bag; consumed when building the
    ``Poroelasticity`` physics.
    """

    model_config = _MECH_MODEL_CONFIG

    zero: float | None = Field(None, description="Small composition epsilon")
    n_points: int | None = Field(None, description="Number of OBL points per axis")
    min_p: float | None = Field(None, description="Min pressure [bar]")
    max_p: float | None = Field(None, description="Max pressure [bar]")
    min_t: float | None = Field(None, description="Min temperature [K]")
    max_t: float | None = Field(None, description="Max temperature [K]")
    min_z: float | None = Field(None, description="Min composition [-]")
    max_z: float | None = Field(None, description="Max composition [-]")
    epsilon_z: float | None = Field(None, description="Composition axis epsilon")


class MechInitialConfig(BaseModel):
    """Initial state (uniform or gradient) for a geomechanics model.

    Replaces the legacy ``InitialSolution`` bag.  ``type`` selects between a
    uniform state and a depth-gradient state; the relevant subset of fields is
    populated accordingly.
    """

    model_config = _MECH_MODEL_CONFIG

    type: str = Field("uniform", description="'uniform' or 'gradient'")
    initial_pressure: ScalarOrArray | None = Field(
        None, description="Initial pressure [bar]"
    )
    initial_temperature: ScalarOrArray | None = Field(
        None, description="Initial temperature [K]"
    )
    initial_displacements: Any = Field(
        None, description="Initial displacements [Ux, Uy, Uz] [m]"
    )
    initial_composition: Any = Field(None, description="Initial composition")
    reference_depth_for_pressure: float | None = Field(
        None, description="Ref depth p [m]"
    )
    pressure_gradient: float | None = Field(
        None, description="Pressure gradient [bar/m]"
    )
    pressure_at_ref_depth: float | None = Field(
        None, description="Pressure at ref depth [bar]"
    )
    reference_depth_for_temperature: float | None = Field(
        None, description="Ref depth T [m]"
    )
    temperature_gradient: float | None = Field(
        None, description="Temperature gradient [K/m]"
    )
    temperature_at_ref_depth: float | None = Field(
        None, description="Temperature at ref depth [K]"
    )


class MechMeshConfig(BaseModel):
    """Mesh identity and tag maps for a geomechanics model.

    Replaces the legacy ``MeshData`` bag.
    """

    model_config = _MECH_MODEL_CONFIG

    mesh_filename: str | None = Field(None, description="Path to the mesh file")
    bnd_tags: dict | None = Field(None, description="Boundary-face name -> gmsh tag")
    matrix_tags: list | None = Field(None, description="Matrix region gmsh tags")
    tags: dict | None = Field(
        None, description="Full name -> tag map (mesh generation)"
    )


class MechOtherConfig(BaseModel):
    """Model-specific auxiliary inputs (geometry, controls, analytics).

    Replaces the legacy free-form ``OtherProps`` bag.  Every attribute observed
    across the geomechanics models is declared here explicitly so the container
    stays typed (``extra="forbid"``) rather than accepting arbitrary attributes.
    """

    model_config = _MECH_MODEL_CONFIG

    # analytics (poroelastic reference solutions)
    case_name: str | None = Field(None, description="Case identifier")
    F: float | None = Field(None, description="Applied load [bar*m]")
    Fa: float | None = Field(None, description="Mandel load parameter [bar*m]")
    load_vertic: float | None = Field(None, description="Vertical load [bar]")
    load_horiz: float | None = Field(None, description="Horizontal load [bar]")
    h: ScalarOrArray | None = Field(None, description="Layer thickness fractions")
    M: ScalarOrArray | None = Field(None, description="Biot modulus")
    m: ScalarOrArray | None = Field(None, description="Confined compressibility term")
    skempton: ScalarOrArray | None = Field(None, description="Skempton coefficient")
    c: ScalarOrArray | None = Field(None, description="Consolidation coefficient")
    kd: ScalarOrArray | None = Field(
        None, description="Drained bulk modulus (per-region) [bar]"
    )
    # geometry (THM proxy / field models)
    nx: int | None = Field(None, description="Grid cells x")
    ny: int | None = Field(None, description="Grid cells y")
    nz: int | None = Field(None, description="Grid cells z")
    rsv_top: float | None = Field(None, description="Reservoir top depth [m]")
    rsv_bottom: float | None = Field(None, description="Reservoir bottom depth [m]")
    rsv_xy: float | None = Field(None, description="Reservoir half-size in x-y [m]")
    rsv_x1: float | None = Field(None, description="Reservoir x min [m]")
    rsv_x2: float | None = Field(None, description="Reservoir x max [m]")
    rsv_y1: float | None = Field(None, description="Reservoir y min [m]")
    rsv_y2: float | None = Field(None, description="Reservoir y max [m]")
    frac_width: float | None = Field(None, description="Fracture width [m]")
    Xc: Any = Field(None, description="Cell-boundary x coordinates")
    Yc: Any = Field(None, description="Cell-boundary y coordinates")
    Zc: Any = Field(None, description="Cell-boundary z coordinates")
    # wells / controls (THM proxy)
    prod_well_coords: list | None = Field(None, description="Producer [X, Y, Z1, Z2]")
    inj_well_coords: list | None = Field(None, description="Injector [X, Y, Z1, Z2]")
    delta_temp_inj: float | None = Field(
        None, description="Injection temperature offset [K]"
    )
    delta_p: float | None = Field(
        None, description="Well BHP offset from initial [bar]"
    )
    well_rate: float | None = Field(
        None, description="Well rate (mass/volume) [kg/day]"
    )
    wctrl_type: Any = Field(None, description="well_control_iface control-type enum")
    # fractures (displaced fault / DFN)
    frac_apers: ScalarOrArray | None = Field(
        None, description="Fracture aperture(s) [m]"
    )
    perm_frac: Any = Field(
        None, description="Fracture permeability [mD] or geometry flag"
    )
    friction: ScalarOrArray | None = Field(
        None, description="Fault/fracture friction [-]"
    )


class MechSimConfig(BaseModel):
    """Simulation timing and solver-parameter handles.

    Replaces the legacy ``Simulation`` bag.  ``time_steps`` is a numpy array of
    report times; ``data_ts`` holds a :class:`darts.models.darts_model.DataTS`
    object with nonlinear/linear solver tolerances.
    """

    model_config = _MECH_MODEL_CONFIG

    time_steps: Any = Field(None, description="Report time steps [days]")
    data_ts: Any = Field(None, description="DataTS solver-parameters object")


class MechModelConfig(BaseModel):
    """Aggregate geomechanics model configuration.

    Single-root replacement for ``darts.input.input_data.InputData``.  A model
    builds one ``MechModelConfig`` and passes it to the poromechanics reservoir
    and physics builders in place of the old ``idata`` god-struct.
    """

    model_config = _MECH_MODEL_CONFIG

    type_hydr: str = Field(..., description="'isothermal' or 'thermal'")
    type_mech: str = Field(
        ..., description="'poroelasticity', 'thermoporoelasticity' or 'none'"
    )
    rock: MechRockConfig = Field(default_factory=MechRockConfig)
    fluid: MechFluidConfig = Field(default_factory=MechFluidConfig)
    obl: MechOBLConfig = Field(default_factory=MechOBLConfig)
    initial: MechInitialConfig = Field(default_factory=MechInitialConfig)
    mesh: MechMeshConfig = Field(default_factory=MechMeshConfig)
    other: MechOtherConfig = Field(default_factory=MechOtherConfig)
    sim: MechSimConfig = Field(default_factory=MechSimConfig)
    boundary: Any = Field(None, description="Boundary-condition spec (tag -> bc dict)")

    @classmethod
    def create(cls, type_hydr: str, type_mech: str, init_type: str = "uniform"):
        """Build an empty config mirroring ``InputData(type_hydr, type_mech, init_type)``.

        :param type_hydr: ``'isothermal'`` or ``'thermal'``.
        :type type_hydr: str
        :param type_mech: ``'poroelasticity'``, ``'thermoporoelasticity'`` or ``'none'``.
        :type type_mech: str
        :param init_type: initial-condition kind, ``'uniform'`` or ``'gradient'``.
        :type init_type: str
        :return: a fresh aggregate config with empty sub-configs.
        :rtype: MechModelConfig
        """
        return cls(
            type_hydr=type_hydr,
            type_mech=type_mech,
            initial=MechInitialConfig(type=init_type),
        )

    def make_prop_arrays(self):
        """Broadcast scalar rock properties to per-region arrays.

        Thin wrapper over :meth:`MechRockConfig.make_region_arrays`, preserving
        the legacy ``InputData.make_prop_arrays()`` entry point.

        :return: ``None``; ``self.rock`` is mutated in place.
        """
        self.rock.make_region_arrays()

    def check(self):
        """Validate the two structural rock invariants of the legacy struct.

        Mirrors the load-bearing part of the old ``InputData.check()``:
        permeability must be given either as ``perm`` or as the full
        ``permx``/``permy``/``permz`` triple, and (for a mechanics run) the
        elastic moduli must be given either as ``stiffness`` or as ``E`` and
        ``nu``.  The legacy per-field ``None`` scan is deliberately not
        reproduced field-by-field: unlike the old struct (which created thermal
        fields only conditionally), these configs always declare every field, so
        an exhaustive ``None`` check would spuriously reject valid isothermal
        cases.  Genuinely missing values still surface at the physics/reservoir
        builder that consumes them.

        :raises AssertionError: if the permeability or elastic-moduli invariant
            is violated, or the ``type_*`` tags are unknown.
        :return: ``None``.
        """
        assert self.type_hydr in ("isothermal", "thermal"), "Unknown type_hydr"
        assert self.type_mech in (
            "poroelasticity",
            "thermoporoelasticity",
            "none",
        ), "Unknown type_mech"
        rock = self.rock
        # ``any(v is None ...)`` (identity), not ``None in (...)`` (elementwise
        # ``==``), so array-valued heterogeneous perm/E fields are handled.
        if rock.perm is None and any(
            v is None for v in (rock.permx, rock.permy, rock.permz)
        ):
            raise AssertionError(
                "MechModelConfig.check: rock permeability (perm or permx/permy/permz)"
                " is not initialized"
            )
        if (
            self.type_mech != "none"
            and rock.stiffness is None
            and any(v is None for v in (rock.E, rock.nu))
        ):
            raise AssertionError(
                "MechModelConfig.check: rock stiffness or (E and nu) is not initialized"
            )
