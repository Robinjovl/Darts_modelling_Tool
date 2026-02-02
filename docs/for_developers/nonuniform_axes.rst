Non-uniform axis support
========================

Overview
--------

The multilinear interpolators now accept explicitly specified supporting-point
coordinates for every axis.  Axis nodes are stored in flattened arrays
(`axis_nodes_flat`, `axis_inv_dx_flat`) together with:

* per-axis offsets for fast `index -> coordinate` conversion
* per-cell inverse spacing used during derivative reduction
* compact coarse bin LUTs (up to 1024 bins/axis) that provide an *O(1)*
  interval guess followed by at most four correction steps

This keeps point/hypercube lookup complexity at :math:`O(N_\mathrm{dims})`
while allowing arbitrary spacing in each dimension.


Python API
----------

All ``multilinear_adaptive_cpu_interpolator`` instantiations now have an
additional constructor that accepts nested lists with axis nodes:

.. code-block:: python

    import numpy as np
    import darts.engines as engines
    from darts.engines import index_vector, value_vector, timer_node

    axis_nodes = [
        [0.0, 0.02, 0.2, 0.5, 1.2],
        [2.0, 2.5, 3.5, 5.0],
        [-1.0, -0.5, 0.0, 0.05, 0.1, 0.2],
    ]

    axes_points = [len(nodes) for nodes in axis_nodes]
    axes_min = [nodes[0] for nodes in axis_nodes]
    axes_max = [nodes[-1] for nodes in axis_nodes]

    evaluator = MyEvaluator(n_dim=len(axis_nodes), n_ops=24)
    ctor_name = f"multilinear_adaptive_cpu_interpolator_l_d_{evaluator.n_dim}_{evaluator.n_ops}"
    ctor = getattr(engines, ctor_name)
    interpolator = ctor(
        evaluator,
        index_vector(axes_points),
        value_vector(axes_min),
        value_vector(axes_max),
        axis_nodes,  # <-- non-uniform nodes
    )
    interpolator.init()
    interpolator.init_timer_node(timer_node())

The C++ constructors follow the same pattern and accept
``std::vector<std::vector<double>>`` with axis nodes.


Regression tests
----------------

``tests/engines/src/interpolation/test_nonuniform_axes.py`` exercises:

* perfect reconstruction (values + gradients) on clustered non-uniform nodes
* parity with the legacy uniform-grid constructor
* boundary clamping/extrapolation safety

Use ``python3 -m pytest tests/engines/src/interpolation/test_nonuniform_axes.py``
after rebuilding the ``darts.engines`` extension to run the suite.


Benchmark helper
----------------

``tests/engines/src/interpolation/bench_nonuniform_axes.py`` measures:

* throughput for uniform vs. non-uniform grids
* the distribution of correction steps required after the coarse-bin guess

The script requires the rebuilt extension module. Execute it from the
repository root with:

.. code-block:: bash

    python3 tests/engines/src/interpolation/bench_nonuniform_axes.py

The output reports wall-clock timings and a histogram of correction steps so
you can validate that the bounded adjustment loop remains tight.


GPU follow-up
-------------

The CUDA implementation can reuse the same data layout with a few additions:

* Mirror ``axis_nodes_flat``/``axis_inv_dx_flat``/``axis_bin_left_idx_flat`` into
  ``thrust::device_vector`` objects and bind them to read-only/texture caches to
  maximize bandwidth reuse.
* Keep coarse-bin tables in ``__constant__`` memory when the per-axis bin count
  is small (``<= 1024``).  For larger tables, stage per-axis chunks into shared
  memory per thread block.
* Implement ``get_axis_interval_index_nonuniform`` on the device with the same
  two-level strategy.  The correction loop should stay branch-bounded
  (``<= 4`` steps) to avoid warp divergence; use ``__shfl_sync`` to broadcast
  successful corrections when processing batched points.
* Expose a device-side constructor that uploads user-provided axis nodes once
  and reuses them for all kernels (similar to how host adaptive interpolators
  reuse their flattened buffers).

This plan keeps the GPU kernels warp-friendly and requires no changes to the
public Python API once the device-side buffers mirror the host equivalents.
