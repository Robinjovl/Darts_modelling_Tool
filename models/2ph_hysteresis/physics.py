import darts.engines as darts_engines
import numpy as np

from darts.engines import *
from darts.engines import index_vector, operator_set_evaluator_iface, timer_node, value_vector
from darts.physics.base.physics_base import PhysicsBase
from darts.physics.super.physics import Compositional

import warnings
class CompositionalCapillary(Compositional):
    def __init__(self, components: list, phases: list, timer: timer_node, n_points: int,
                 min_p: float, max_p: float, min_z: float, max_z: float, min_t: float = None, max_t: float = None,
                 state_spec: PhysicsBase.StateSpecification = PhysicsBase.StateSpecification.P,
                 cache: bool = False, axes_min=None, axes_max=None, n_axes_points=None,
                 epsilon_z: float = 1e-8):

        if axes_min is None or axes_max is None or n_axes_points is None:
            super().__init__(
                components=components,
                phases=phases,
                timer=timer,
                n_points=n_points,
                min_p=min_p,
                max_p=max_p,
                min_z=min_z,
                max_z=max_z,
                epsilon_z=epsilon_z,
                min_t=min_t,
                max_t=max_t,
                state_spec=state_spec,
                cache=cache,
                axes_min=axes_min,
                axes_max=axes_max,
                n_axes_points=n_axes_points,
            )
            self.PT_axes_min = value_vector(list(self.PT_axes_min) + [0.0])
            self.PT_axes_max = value_vector(list(self.PT_axes_max) + [1.0])
            self.n_axes_points = index_vector(list(self.n_axes_points) + [n_points])
        else:
            super().__init__(
                components=components,
                phases=phases,
                timer=timer,
                n_points=n_points,
                min_p=min_p,
                max_p=max_p,
                min_z=min_z,
                max_z=max_z,
                epsilon_z=epsilon_z,
                min_t=min_t,
                max_t=max_t,
                state_spec=state_spec,
                cache=cache,
                axes_min=axes_min,
                axes_max=axes_max,
                n_axes_points=n_axes_points,
            )

    def create_interpolator(self, evaluator: operator_set_evaluator_iface, axes_min: value_vector, axes_max: value_vector,
                            timer_name: str, n_ops: int, algorithm: str = 'multilinear', mode: str = 'adaptive',
                            platform: str = 'cpu', precision: str = 'd', region: str = '',
                            is_barycentric: bool = False):
        """
        Create interpolator object according to specified parameters

        :param evaluator: State operators to be interpolated. Evaluator object is used to generate supporting points
        :type evaluator: darts.engines.operator_set_evaluator_iface
        :param timer_name: Name of timer object
        :type timer_name: str
        :param algorithm: interpolator type:
            'multilinear' (default) - piecewise multilinear generalization of piecewise bilinear interpolation on rectangles;
            'linear' - a piecewise linear generalization of piecewise linear interpolation on triangles
        :type algorithm: str
        :param mode: interpolator mode:
            'adaptive' (default) - only supporting points required to perform interpolation are evaluated on-the-fly;
            'static' - all supporting points are evaluated during itor object construction
        :type mode: str
        :param platform: platform used for interpolation calculations :
            'cpu' (default) - interpolation happens on CPU;
            'gpu' - interpolation happens on GPU
        :type platform: str
        :param precision: precision used in interpolation calculations:
            'd' (default) - supporting points are stored and interpolation is performed using double precision;
            's' - supporting points are stored and interpolation is performed using single precision
        :type precision: str
        :type region: str
        :param region: str(region index) for reservoir operator, str(-1) for well operator, '' for others
        needed to make different filenames for cache as self.well_operators has the same type ReservoirOperators
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        """
        n_dims = len(self.n_axes_points)
        assert n_dims == self.n_vars + 1
        assert len(axes_min) == n_dims
        assert len(axes_max) == n_dims
        for n_p in self.n_axes_points:
            assert n_p > 1

        itor_name = "%s_%s_%s_interpolator_i_%s_%d_%d" % (algorithm,
                                                          mode,
                                                          platform,
                                                          precision,
                                                          n_dims,
                                                          n_ops)
        itor = None
        try:
            itor_cls = getattr(darts_engines, itor_name)
            if algorithm == 'linear':
                itor = itor_cls(evaluator, self.n_axes_points, axes_min, axes_max, is_barycentric)
            else:
                itor = itor_cls(evaluator, self.n_axes_points, axes_min, axes_max)
        except (AttributeError, ValueError):
            if np.prod(np.array(self.n_axes_points), dtype=np.float64) < np.iinfo(np.int64).max:
                itor_name = itor_name.replace("interpolator_i", "interpolator_l")
            else:
                itor_name = itor_name.replace("interpolator_i", "interpolator_ll")
            try:
                itor_cls = getattr(darts_engines, itor_name)
                if algorithm == 'linear':
                    itor = itor_cls(evaluator, self.n_axes_points, axes_min, axes_max, is_barycentric)
                else:
                    itor = itor_cls(evaluator, self.n_axes_points, axes_min, axes_max)
            except (AttributeError, ValueError) as exc:
                raise ValueError(
                    f"No compiled interpolator found for n_dims={n_dims}, n_ops={n_ops}"
                ) from exc

        self.create_itor_timers(itor, timer_name)
        itor.init()
        return itor, n_ops

    def init_physics(
        self,
        discr_type: str = "tpfa",
        platform: str = "cpu",
        itor_type: str = "multilinear",
        itor_mode: str = "adaptive",
        itor_precision: str = "d",
        verbose: bool = False,
        is_barycentric: bool = False,
        n_solid: int = None,
    ):
        super().init_physics(
            discr_type=discr_type,
            platform=platform,
            itor_type=itor_type,
            itor_mode=itor_mode,
            itor_precision=itor_precision,
            verbose=verbose,
            is_barycentric=is_barycentric,
            n_solid=n_solid,
        )
        self.engine.hysteresis_enabled = True

    def init_wells(self, wells):
        super().init_wells(wells)
        for well in wells:
            well.hysteresis_enabled = True
            if hasattr(well, "control"):
                well.control.hysteresis_enabled = True
            if hasattr(well, "constraint"):
                well.constraint.hysteresis_enabled = True

    def set_well_controls(
        self,
        wctrl,
        control_type,
        is_inj,
        target,
        phase_name=None,
        inj_composition=None,
        inj_temp=None,
    ):
        super().set_well_controls(
            wctrl=wctrl,
            control_type=control_type,
            is_inj=is_inj,
            target=target,
            phase_name=phase_name,
            inj_composition=inj_composition,
            inj_temp=inj_temp,
        )
        wctrl.hysteresis_enabled = True

