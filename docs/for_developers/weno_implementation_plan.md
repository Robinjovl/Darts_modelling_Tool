# WENO2 implementation plan: discretizer-precomputed geometry

This document specifies the proposed implementation of the fully implicit,
second-order WENO transport reconstruction described by Lie, Mykkeltvedt, and
Møyner. It is an implementation plan, not documentation of an already available
feature.

The central design decision is that `discretizer/src` owns every calculation that
depends only on the mesh. The discretizer exports compact, sorted geometry
coefficients through `conn_mesh`. The Super engine combines those coefficients
with cell-centred, state-dependent OBL operators during every nonlinear assembly
and computes the nonlinear WENO weights, residual, and complete Jacobian.

The initial production target is the Super CPU TPFA engine on matrix cells. The
spatial reconstruction must support hexahedra, tetrahedra, wedges/prisms, and
pyramids, including mixed-topology meshes. WENO reconstruction for MPFA, GPU,
fracture transport, mechanics, wells, and adjoints is a later extension, but the
static data contract should not prevent those ports.

## 1. Method boundary and naming

The target scheme should be exposed as `WENO2`. The primary paper uses linear
candidate polynomials and midpoint face quadrature and is formally second order
in space. It is higher order than single-point upstream mobility (SPU), but it is
not the classical fifth-order Jiang--Shu finite-difference scheme.

The existing pressure/phase-potential approximation remains unchanged. WENO
replaces only the piecewise-constant, one-sided cell value used for advective
mobility and component mobility at a face.

```text
mesh and rock data
    -> discretizer/src: all static WENO geometry and stencil preprocessing
    -> conn_mesh: one-time transfer, ownership, remapping, and assembly ordering

cell primary states
    -> existing adaptive OBL interpolation: operator values and derivatives
    -> engine_super_cpu: derived transport fields, WENO weights and face values
    -> engine_super_cpu: phase-potential selection, residual, and full Jacobian
```

The adaptive OBL interpolator remains a state-space interpolator. It must not gain
physical cell, face, or WENO-stencil knowledge, and WENO must not trigger extra
OBL evaluations at reconstructed face states.

## 2. Ownership of work

| Layer | Owns | Must not own |
| --- | --- | --- |
| `mesh::Mesh` in `discretizer/src/mesh` | Raw topology, nodes, element types, cell and face centroids, face areas, face-node incidence, adjacency, boundary classification | OBL values or nonlinear WENO weights |
| `dis::Discretizer` | Compact-face selection, local SVD coordinates, candidate construction, matrix inversion, positive linear weights, face query vectors, support/dependency lists, quality and fallback flags | State-dependent smoothness indicators or residual/Jacobian entries |
| Python reservoir adapters | Select WENO2, create or retain the C++ discretizer mesh, provide an explicit discretizer-to-engine cell map, and pass exported vectors once | Numerical reconstruction loops |
| `conn_mesh` | Own exported static arrays, map canonical one-way faces to the final directed connection order, create sorted row dependencies and hot-loop lookup slots | SVD, candidate construction, OBL evaluation, or nonlinear WENO arithmetic |
| OBL operator evaluators and interpolators | Cell-centred thermodynamic/transport operator values and derivatives | Mesh reconstruction |
| `engine_super_cpu` | Derived cell transport fields, nonlinear weights, one-sided face values and sensitivities, upwind selection, residual, and complete Jacobian | Geometry discovery, SVD, matrix inversion, or runtime map searches |

The separation is important for both correctness and cost. Geometry is immutable
over a simulation, whereas WENO smoothness and weights change with the current
Newton state.

## 3. Existing discretizer facilities to reuse

The C++ discretizer already contains most of the required raw information and
several suitable implementation patterns:

* `mesh::Mesh` stores `nodes`, `elems`, `elem_nodes`, `centroids`, `conns`,
  `conn_nodes`, `adj_matrix`, `adj_matrix_cols`, and `adj_matrix_offset`.
* `mesh::ElemType` already distinguishes `TETRA`, `HEX`, `PRISM`, and
  `PYRAMID`.
* `mesh::Connection` stores the connection type, two element IDs, face-node
  offset/count, normal, centroid `c`, CPG second-side centroid `c_2`, and area.
* Gmsh construction obtains face nodes by intersecting element-node sets and
  computes face centroids, areas, and normals.
* CPG construction already receives cell/face centroids, face areas, face nodes,
  face-cell incidence, and `cell_facetag`. It also computes side-specific original
  face centres for split faces near faults.
* `LinearApproximation::sort()` sorts a stencil and permutes its coefficient
  columns consistently. WENO export should follow the same deterministic
  stencil/coefficient discipline.
* `Discretizer::reconstruct_pressure*_gradients_per_cell()` demonstrates
  cell-local least-squares construction, SVD/pseudoinverse fallback, and storage
  of a cell stencil.
* `Discretizer::calc_mpfa_transmissibilities()` demonstrates a flat,
  connection-ordered export through `flux_stencil`, `flux_offset`,
  `flux_vals`, and `flux_rhs`.
* The currently unused connection-based gradient helper
  `find_connections_to_reconstruct_gradient()` already evaluates local
  three-direction geometry and determinant quality. Its selection objective is
  MPFA-specific and should not be reused directly, but its small-matrix and
  combination machinery is relevant.

WENO should therefore add a separate `WenoStaticData` product to `Discretizer`.
It should not overload `p_grads`, `flux_stencil`, or MPFA transmissibilities:
pressure-gradient MPFA and nonlinear transport reconstruction have different
mathematical meanings even though both use flattened stencil arrays.

## 4. Static and state-dependent data

### 4.1 Precompute in the discretizer

The following quantities are invariant for a fixed mesh and must be calculated
once before simulation:

* eligible matrix cells and geometric face sides;
* compact logical-face selection and largest-subface selection;
* cell-local SVD scaling/rotation;
* transformed support-point coordinates;
* candidate support cell IDs and boundary value-source kinds;
* each candidate's inverse `3 x 3` interpolation matrix;
* positive linear weight proportional to reference tetrahedron volume;
* transformed query vector from the target centroid to every physical face side;
* per-cell union of WENO-dependent cell IDs;
* canonical one-way flow-connection to its two geometric face-side records;
* cell/connection eligibility, rank, conditioning, and fallback reason;
* deterministic candidate, face, and support ordering.

Permeability is static but is deliberately not part of the WENO smoothness
geometry. It remains in the precomputed TPFA/MPFA flux coefficient. Candidate
construction follows cell/face geometry, as in the WENO paper, rather than an
MPFA co-normal objective.

### 4.2 Finalize in `conn_mesh` before engine initialization

The following operations are also static but depend on the final engine block and
connection ordering, including wells:

* duplicate one-way reservoir face-side references into directed connections;
* reorder those references with `reverse_and_sort()`;
* mark well, DFM, fracture, and geometry-less NNC connections as SPU;
* apply the static `op_num` region mask to candidate eligibility;
* form each residual row's fixed dependency union for both possible upstream
  directions;
* sort/unique every row's BCSR column list;
* compute local BCSR slots for direct neighbours and every WENO support cell;
* set the final `n_links` used to allocate the Jacobian.

No raw nodes, SVD objects, or dynamic matrices should be copied into the engine.

### 4.3 Evaluate on every nonlinear assembly

The following quantities depend on the current cell state and remain in the
engine:

* OBL `LAMBDA`, `FLUX`, density, capillary pressure, permeability multiplier,
  and all corresponding state derivatives;
* derived phase mobility `L_p = lambda_p`;
* derived component/energy mobility `C_p,c = lambda_p * beta_p,c`;
* candidate gradients, smoothness indicators, nonlinear weights, and face
  values;
* WENO sensitivities with respect to every support-cell scalar;
* phase-potential difference and binary upstream side;
* residual, CFL/output fluxes, and all Jacobian contributions.

The split prevents a geometry-dependent multiplier from being represented as an
OBL operator and prevents a state-dependent multiplier from being frozen in the
discretizer.

## 5. Static numerical representation

For target cell `i`, candidate `k` uses three geometry support points. An internal
support point is at a neighbouring cell centroid and obtains its scalar value from
that neighbour. A no-flow boundary support point is at the boundary-face centroid
but obtains its scalar value from cell `i`. Other boundary source kinds are
discussed below.

Let `X_i` contain the vectors from the cell centroid to the selected geometric
face centroids. The discretizer computes

```text
X_i = U_i Sigma_i V_i^T,
y = Sigma_i^-1 V_i^T (x - x_i).
```

The SVD transform is used for support points, face query points, and the
smoothness indicator. It does not alter the value of a linear candidate at a
physical point.

For candidate support points `s1`, `s2`, and `s3`, define the reference-coordinate
matrix with rows `y_s1^T`, `y_s2^T`, and `y_s3^T`. The discretizer stores its
inverse `A_i,k`. At runtime the engine only performs

```text
d_i,k = [q_s1 - q_i, q_s2 - q_i, q_s3 - q_i]^T
sigma_i,k = A_i,k d_i,k
IS_i,k = sigma_i,k^T sigma_i,k
q_i,k(x_f) = q_i + y_i,f^T sigma_i,k.
```

The unnormalised and normalised WENO weights are

```text
alpha_i,k = gamma_i,k / (epsilon + IS_i,k)^p
w_i,k = alpha_i,k / sum_l alpha_i,l,
```

with `p = 2` and initially `epsilon = 1e-12`. The discretizer stores
`gamma_i,k` proportional to `abs(det(A_i,k^-1)) / 6` in reference coordinates
and normalises the positive weights per target cell. Degenerate candidates are
removed before normalisation.

Storing `A_i,k` and `y_i,f` is preferable to storing candidate coefficients for
every candidate/face pair: all expensive geometry operations are precomputed,
while the static memory remains proportional to candidates plus cell-face sides
rather than their Cartesian product.

## 6. Candidate construction and topology support

### 6.1 Unified geometric-face view

Add a small geometry-only face view to `mesh::Mesh`, separate from the physical
flow connection list. It must represent internal and boundary face sides and
contain:

* target element;
* neighbour element or boundary marker;
* global face/node incidence;
* centroid as viewed from the target side;
* area;
* connection type;
* logical-face tag where available;
* canonical physical connection ID, or an invalid value for geometry-only
  boundary faces.

This extra view is necessary because CPG currently omits external faces from
`Mesh::conns`, while a robust SVD and a boundary candidate need all geometric
faces. It also avoids changing existing TPFA/MPFA connection semantics.

For ordinary Gmsh faces, initialise `Connection::c_2 = Connection::c` so the two
mesh sources expose the same side-centroid contract. For CPG split/fault faces,
retain the existing distinct `c` and `c_2` values. The WENO face-side query vector
must use the centroid belonging to the target cell.

### 6.2 Compact face selection

For conforming Gmsh grids, retain one support per geometric cell face. For CPG,
group subfaces by `cell_facetag` and select the largest-area subface as the
polynomial support in each logical direction. All subface flow connections remain
in the physical flux loop and each still receives a face-side query vector.

The selected faces also define `X_i` for the SVD. This reproduces the compact
stencil intent of the primary paper and prevents small fault subfaces from
dominating candidate count or smoothness scaling.

### 6.3 Primary candidates from face-node incidence

For every target-cell vertex, collect selected faces incident on that vertex. A
set of three incident faces gives a primary 3D candidate. If more than three faces
meet at a vertex, enumerate its full-rank three-face combinations, then deduplicate
support tuples.

Expected interior counts on regular cells are:

| Element | Faces | Vertices | Nominal candidates |
| --- | ---: | ---: | ---: |
| Hexahedron | 6 | 8 | 8 |
| Tetrahedron | 4 | 4 | 4 |
| Wedge/prism | 5 | 6 | 6 |
| Pyramid | 5 | 5 | 8 |

The pyramid has four base-vertex candidates and four full-rank combinations of
the side faces meeting at its apex.

Reject a candidate when its reference-coordinate matrix is rank deficient,
nonfinite, or exceeds the configured condition threshold. If face-node incidence
is unavailable for a general polyhedron, enumerate all selected-face triplets,
rank by reference tetrahedron volume and conditioning, and retain a configured
maximum. This fallback must not replace the explicit standard-topology tests.

### 6.4 Boundary and interface policy

Each support stores a value-source kind:

* `CELL`: read the neighbour's OBL-derived scalar;
* `TARGET_COPY`: read the target value for a no-flow boundary support;
* `BOUNDARY_DOF`: read an existing fixed boundary state when that state has the
  required operator value and derivative contract;
* `INVALID`: exclude the candidate.

The first implementation should support `CELL` and `TARGET_COPY`. Prescribed
inflow/Robin boundaries without a compatible boundary OBL state use a counted
local SPU fallback.

Candidates crossing different `op_num` regions are disabled once `conn_mesh`
knows the final region array. The geometry remains valid and does not need to be
recomputed; the remaining positive `gamma` values are renormalised once before
simulation. A target with too few candidates is marked SPU.

## 7. Proposed flattened data contract

Add a `WenoStaticData` member to `dis::Discretizer`, exposed as standard vectors
in the same style as MPFA. Suggested arrays are:

```text
cell_candidate_offset       n_reservoir_cells + 1
candidate_support_cell      3 * n_candidates
candidate_support_kind      3 * n_candidates
candidate_inverse           9 * n_candidates
candidate_gamma             n_candidates
candidate_condition         n_candidates          (diagnostics/setup only)

cell_face_side_offset       n_reservoir_cells + 1
face_side_connection        n_face_sides
face_side_reference_dx      3 * n_face_sides
face_side_status            n_face_sides

one_way_face_side_m         n_one_way_reservoir_connections
one_way_face_side_p         n_one_way_reservoir_connections

cell_dependency_offset      n_reservoir_cells + 1
cell_dependency_cell        sum(unique target/support dependencies)
cell_weno_status            n_reservoir_cells
```

`candidate_inverse` uses a documented row-major layout. Support order and matrix
columns must always be permuted together.

The discretizer sorts data as follows:

1. target cells in engine-cell order;
2. cell dependencies by ascending engine cell ID;
3. candidates lexicographically by their three value-source cell IDs and source
   kinds;
4. support IDs within a candidate with the inverse-matrix columns permuted to
   match;
5. canonical one-way face records in the same order as the `cell_m/cell_p`
   arrays exported to `conn_mesh`;
6. face-side records by target cell and canonical connection ID.

The raw discretizer element ID is not assumed to equal the engine block ID. The
reservoir supplies a mandatory `discretizer_cell_to_engine` mapping. CPG active
cells use an identity/local mapping. The plain Gmsh unstructured path currently
orders matrix and fracture cells differently from `mesh::Mesh`, so it requires an
explicit map and a permutation of every support/target ID before export.

Because `darts.discretizer` and `darts.engines` are separate extension modules,
the initial transfer should follow the existing MPFA precedent: expose standard
vectors on `Discretizer` and call one `conn_mesh.init_weno_static(...)` method
from Python. This incurs one setup-time copy but no per-iteration Python work.
Moving the structure to a shared cross-module C++ library is only justified if
profiling shows setup copies to be material.

## 8. `conn_mesh` lifecycle and assembly order

`conn_mesh.init_weno_static(...)` is called after the normal reservoir
`conn_mesh.init(...)`/`init_mpfa(...)` and before wells are appended. It validates
array lengths, canonical connection alignment, target IDs, support IDs, positive
weights, and sorted offsets, then owns the data.

`ReservoirBase.init_wells()` later calls `add_wells()` and
`reverse_and_sort()`. Extend `reverse_and_sort()` so that it:

1. keeps candidate arrays grouped by target reservoir cell;
2. maps each canonical reservoir connection to forward and reverse directed
   connection IDs using the existing `one_way_to_conn_index_*` arrays;
3. attaches both target-side face records to each directed connection, swapping
   them for the reverse direction;
4. marks appended well/DFM connections as WENO-ineligible;
5. constructs the fixed row dependency graph;
6. stores physical-connection-to-BCSR local slots in the same directed order as
   `block_m`, `block_p`, `tran`, and `grav_coef`.

The row graph for residual row `i` contains `i`, every physical neighbour `j`,
and the dependency cells for both target `i` and target `j` on every incident
eligible face. Both sides are required because phase-potential upwind direction
can change between Newton iterations.

The current TPFA engine assumes CSR off-diagonals and physical connections are the
same list. WENO requires two separate views:

* a directed physical-connection CSR used to evaluate fluxes;
* an expanded BCSR dependency graph used to store derivatives.

The engine must iterate the first and use precomputed slots into the second. It
must never treat a WENO-only dependency column as a physical diffusion,
conduction, or advective connection.

The MPFA `cell_stencil`/`n_links` Jacobian setup is a useful model for the expanded
row graph. It should be generalised into shared fixed-structure infrastructure
instead of copied into the WENO kernel.

## 9. State-dependent engine assembly

The existing OBL batch remains the first assembly phase and produces
`op_vals_arr` and `op_ders_arr` for every current cell state.

For phase `p` and conserved quantity `c`, the engine forms cell values

```text
L_p = LAMBDA_p
C_p,c = LAMBDA_p * FLUX_p,c
```

and their local state derivatives

```text
dC_p,c/dX = FLUX_p,c * dLAMBDA_p/dX
             + LAMBDA_p * dFLUX_p,c/dX.
```

For each directed physical face and phase:

1. evaluate the existing two-point phase-potential difference;
2. select the upstream target cell using the existing sign convention;
3. if the connection/target is SPU, run the existing path;
4. otherwise load the target candidate range and face-side reference vector;
5. reconstruct `L_p` and all `C_p,c` fields using static inverse matrices and
   current cell values;
6. calculate analytic sensitivities to every support-cell scalar;
7. chain the scalar sensitivities with that support cell's OBL derivatives;
8. assemble the residual and scatter derivatives through precomputed BCSR slots.

The component/energy advective flux is assembled directly as

```text
F_p,c = dt * T_f * m_f * DeltaPhi_p * C_p,c,face.
```

Here `T_f` and the gravity coefficient are static discretizer/mesh quantities,
while `m_f`, `DeltaPhi_p`, and `C_p,c,face` are state-dependent. The harmonic
permeability-porosity multiplier remains in the engine because its `MULT` values
and derivatives come from OBL.

The phase volumetric rate uses reconstructed `L_p`. For thermal potential energy,
add a reconstructed `L_p * rho_p` field and a static face-specific potential
energy computed from the discretizer face centroid. Diffusion, fluid/rock
conduction, and dispersion remain on their existing discretizations unless a
separate higher-order design is approved.

### 9.1 Analytic WENO derivative

For a scalar candidate, static `A_k` maps the three value differences to
`sigma_k`. The engine obtains

```text
dIS_k/dq_l = 2 sigma_k^T (d sigma_k/dq_l).
```

For `alpha_k = gamma_k / (epsilon + IS_k)^p`, define

```text
r_k,l = -p / (epsilon + IS_k) * dIS_k/dq_l.
```

Then the stable normalised-weight derivative is

```text
dw_k/dq_l = w_k * (r_k,l - sum_m w_m r_m,l).
```

The face-value derivative combines the derivative of the candidate value and the
derivative of its weight. This scalar derivative is finally multiplied by the
support cell's `dL/dX` or `dC/dX` and accumulated in the corresponding BCSR block.

The binary upstream indicator is treated semismoothly as in current SPU assembly:
the selected branch is differentiated, but the derivative of the sign switch is
not taken.

### 9.2 Bounds and fallback

For nonnegative fields such as phase mobility and mass component mobility, a
nonfinite or negative reconstructed value reverts locally to the target-cell SPU
value and derivative. Energy coefficients must not be assumed nonnegative because
their sign can depend on the enthalpy reference. Use a finite/local-range safeguard
for energy.

Every fallback increments a reason-specific counter. Silent per-face degradation
is not acceptable.

## 10. Mesh-path integration

| Reservoir path | Static WENO geometry source | Required work |
| --- | --- | --- |
| CPG | Existing C++ `Mesh` and `Discretizer` in `cpg_reservoir.py` | Retain `cell_facetag`/boundary face-side data in `Mesh`, call `prepare_weno_static()`, and export with the active-cell mapping |
| Gmsh unstructured | `Mesh::gmsh_mesh_processing()` and `Discretizer` | Make this C++ mesh the WENO geometry source for plain `UnstructReservoir`; provide the explicit permutation to its current fracture-first engine ordering |
| Structured Cartesian | New lightweight C++ structured-mesh/face builder in `discretizer/src/mesh` | Build hex elements, all internal/boundary geometric face sides, node incidence, centroids, and identity engine mapping; then use the same `prepare_weno_static()` path |

WENO geometry must not be calculated independently in
`darts/reservoirs/mesh/unstruct_discretizer.py` or
`struct_discretizer.py`. Those classes may provide input arrays and mappings, but
candidate construction, SVD, inverses, and static coefficient export belong to
the C++ discretizer.

If a model constructs `conn_mesh` directly and does not retain a compatible
discretizer mesh, selecting WENO2 must raise an actionable initialization error.

## 11. API and diagnostics

Keep SPU as the default and add an explicit scheme enum:

```python
model.params.transport_scheme = sim_params.transport_scheme_t.weno2
```

Initial options should include:

* `weno_epsilon = 1e-12`;
* `weno_power = 2`;
* `weno_condition_limit`;
* `weno_max_candidates` for generic/high-valence cells;
* `weno_compact_stencil = True`;
* `weno_bound_fallback = True`.

`DartsModel.init()` should request WENO preprocessing before
`reservoir.init_wells()` when the enum is active. The default SPU path should not
pay the startup-time or memory cost of WENO static arrays.

Expose at least these diagnostics:

* cells and physical face sides eligible for WENO;
* candidate count distribution by element type;
* fallback counts by rank, condition, boundary, region, unsupported connection,
  nonfinite value, and bound violation;
* minimum/maximum candidate condition and nonlinear weight;
* physical connection count versus expanded BCSR nonzeros;
* bytes used by static WENO data and the expanded Jacobian;
* separate interpolation, WENO reconstruction, assembly, linear setup, and
  linear solve timings.

## 12. Implementation sequence

### Phase 1: static data types and geometry view

* Add the geometry-only face-side representation to
  `discretizer/src/mesh/mesh.h` and populate it for Gmsh and CPG in `mesh.cpp`.
* Normalise the `c`/`c_2` contract.
* Add the structured C++ geometry builder.
* Add topology/face-incidence unit fixtures for hex, tet, prism, and pyramid.

Exit gate: every cell reports the expected selected faces and deterministic
face-node incidence, including boundary faces.

### Phase 2: `Discretizer::prepare_weno_static()`

* Add `WenoStaticData` in `discretizer.h`.
* Implement compact face selection, SVD, reference coordinates, primary
  candidates, inverse matrices, `gamma`, dependency unions, and fallback flags.
* Reuse the existing matrix/SVD implementation, but add focused rank and
  conditioning tests.
* Export vectors in deterministic engine order through `py_discretizer.cpp`.

Exit gate: constant and arbitrary linear scalar fields reconstruct exactly on
rotated and stretched instances of all four required topologies using only the
exported arrays.

### Phase 3: `conn_mesh` transfer and fixed graph

* Add `init_weno_static(...)` and pybind exposure.
* Validate and own the exported vectors.
* Extend `reverse_and_sort()` to remap face sides and build sorted row
  dependencies/BCSR slots.
* Generalise Jacobian allocation from hard-coded `n_conns + n_blocks` to the
  final static graph size.

Exit gate: every finite-difference dependency is present in the graph, and
physical connections remain distinguishable from WENO-only columns.

### Phase 4: physical-connection assembly refactor

Refactor `engine_super_cpu.tpp` so SPU assembly iterates directed physical
connections independently of the Jacobian column list. Do not add WENO arithmetic
in this phase.

Exit gate: the default SPU model suite and flux outputs remain bitwise identical
where ordering permits, otherwise within the established regression tolerance.

### Phase 5: WENO2 runtime and complete Jacobian

* Add a fixed-size, allocation-free reconstruction helper under `engines/src`.
* Form `L_p`, `C_p,c`, and derivatives from existing OBL arrays.
* Implement nonlinear weights, scalar sensitivities, residual, CFL, velocity,
  and BCSR scatter.
* Add thermal advective and potential-energy handling.
* Keep well, DFM, fracture, and unsupported connections on explicit SPU paths.

Exit gate: full residual directional derivatives match finite differences away
from branch-switch surfaces.

### Phase 6: API, output, and compatibility

* Add the enum/options in `globals.h` and `py_globals.cpp`.
* Trigger preprocessing from the model initialization lifecycle.
* Make `darts/models/output.py` consume canonical engine WENO fluxes instead of
  independently recomputing SPU values.
* Add restart, history-axis, multi-region, and error-message coverage.

### Phase 7: performance selection

Start with memory-lean on-the-fly reconstruction using precomputed `A_i,k` and
`y_i,f`. Benchmark, but do not assume, that caching blended slopes or nonlinear
weights is faster. Any cache is valid for one assembly epoch only and must not lag
weights across timesteps.

Select the production layout using end-to-end runtime, Jacobian memory, Newton
iterations, linear iterations, and achieved error. OBL supporting-point evaluation
counts must remain unchanged solely from enabling WENO.

### Phase 8: later engine ports

After CPU TPFA acceptance, extend the same static data contract to MPFA, GPU,
mechanics, fractures, and adjoints. Until implemented, these combinations must
fail explicitly rather than claim WENO while silently using a different global
scheme.

## 13. Validation and acceptance

### 13.1 Static geometry tests

Add C++ unit tests under `tests/cpp/unit/weno` for:

* nominal candidate counts on hex, tet, prism, and pyramid;
* constant and exact linear reconstruction;
* translation, rotation, uniform scaling, and high-aspect-ratio invariance;
* face/node ordering and deterministic export;
* pyramid apex combinations;
* boundary `TARGET_COPY` supports;
* compact CPG largest-subface selection;
* rank/condition rejection and generic-triplet fallback;
* engine-cell permutation for fracture-first Gmsh ordering.

### 13.2 Runtime/Jacobian tests

* Compare scalar WENO sensitivities with central finite differences.
* Compare full residual directional derivatives with finite differences for every
  support cell, avoiding upwind and limiter switching surfaces.
* Verify equal and opposite face residuals.
* Test both phase-potential directions, gravity, capillarity, permeability-porosity
  multiplication, phase appearance, thermal energy, and multiple OBL regions.
* Verify SPU fallback derivatives use the selected SPU branch.

### 13.3 Convergence and topology tests

* Smooth advection with `dt` proportional to `h^2`: approximately first-order
  SPU and second-order WENO2 spatial convergence.
* Discontinuous transport: bounded/nonoscillatory behaviour and the expected
  reduced convergence order.
* Pure and mixed meshes containing every required topology.
* Skewed, nonmatching, inactive-cell, boundary, and high-aspect-ratio cases.
* Add an explicit pyramid mesh; the repository currently has reusable hex, tet,
  and wedge examples but no dedicated pyramid regression fixture.

### 13.4 Repository verification

After implementation, run the mandatory build and verification stack in one
session conda environment, including:

* C++ tests with `ENABLE_TESTING=ON` and `ctest`;
* `models/run_test_suite2.py LOG` with SPU as the default;
* interpolator tests, which should require no WENO-specific change;
* discretizer comparison tests;
* focused WENO model/convergence tests;
* pre-commit on all edited Python files;
* Sphinx documentation build.

### 13.5 Performance acceptance

Report, for representative `10^5`--`10^6` cell cases:

* preprocessing time and static WENO bytes per cell;
* BCSR nonzeros and matrix bytes versus SPU;
* OBL interpolation, WENO reconstruction, residual/Jacobian, solver setup, and
  solve time;
* single-thread and OpenMP scaling;
* Newton and linear iteration counts;
* total time required to reach a fixed error, including comparison with refined
  SPU meshes.

No performance claim should be based only on the reconstruction microkernel.

## 14. Important limitations and decisions

* WENO2 improves the transport reconstruction but does not repair TPFA pressure
  inconsistency on non-K-orthogonal grids. Tetrahedral and mixed-topology support
  means the reconstruction and Jacobian are valid on those cells, not that TPFA
  becomes an MPFA-quality pressure discretization.
* Volume-proportional `gamma` is positive. The Shi--Hu--Shu negative-weight split
  is unnecessary for WENO2 and should only be added behind a precomputed
  `has_negative_gamma` flag if a future optimal higher-order scheme introduces
  negative weights.
* Per-field nonlinear weights are the correctness baseline. Shared phase weights
  may reduce arithmetic but can miss a component discontinuity and require
  separate validation before becoming an option.
* Fully implicit means differentiating the nonlinear weights. Freezing weights is
  an approximate Jacobian and is not the initial implementation.
* The expanded Jacobian is likely the dominant memory cost. Candidate caps and
  compact CPG selection must be justified by both accuracy and matrix-memory
  measurements.

## References

* K.-A. Lie, T. S. Mykkeltvedt, and O. Møyner, "A Fully Implicit WENO Scheme on
  Stratigraphic and Unstructured Polyhedral Grids," *Computational Geosciences*,
  2019. <https://doi.org/10.1007/s10596-019-9829-x>
* G.-S. Jiang and C.-W. Shu, "Efficient Implementation of Weighted ENO Schemes,"
  *Journal of Computational Physics*, 1996.
  <https://doi.org/10.1006/jcph.1996.0130>
* J. Shi, C. Hu, and C.-W. Shu, "A Technique of Treating Negative Weights in WENO
  Schemes," *Journal of Computational Physics*, 2002.
  <https://doi.org/10.1006/jcph.2001.6892>
