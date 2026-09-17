# THM vs Geomechanical Proxy

Comparison of a fully-coupled **Thermo-Hydro-Mechanical (THM)** reservoir simulation against a fast analytical **geomechanical proxy** for thermoporoelastic responses.

The THM simulation runs inside [open-DARTS](https://gitlab.com/open-porous-media/open-darts) on a **3D** unstructured hexahedral mesh. The proxy evaluates the same displacements and stresses analytically from the pressure/temperature changes produced by the THM solver, making it orders of magnitude faster while remaining accurate for the far-field mechanical response.

Both the THM solver and the proxy are verified against the closed-form solution for a **laterally infinite** (uniaxial-strain) reservoir layer.

**Poroelastic horizontal total stress change**:

$$\Delta\sigma_H^\text{total} = \frac{\alpha(1-2\nu)}{1-\nu}\,\Delta P$$

where α is the Biot coefficient and ν is Poisson's ratio. With the default parameters (α = 0.7, ν = 0.25) the ratio is ≈ 0.47; with α = 1 it reduces to the often-cited value 2/3. Vertical total stress is unchanged (Δσv = 0) because the reservoir is free to deform vertically.

**Thermoelastic horizontal total stress change**:

$$\Delta\sigma_H^\text{total} = \frac{E\,\alpha_T}{1-\nu}\,\Delta T$$

where E is Young's modulus and α_T is the linear thermal expansion coefficient (1×10⁻⁵ K⁻¹ by default). Heating produces compressive horizontal stress; cooling produces tension.

The `print_at_point` mode in `main_proxy.py` prints both the THM and proxy values at the reservoir mid-point alongside these limits.

---

## Overview

| Step | Tool | What it does |
|------|------|--------------|
| 1 | `main.py` | Runs the full THM simulation; writes VTK output |
| 2 | `main_proxy.py` | Reads THM VTK output, runs the proxy, plots comparison |

**Physics options**

| `physics_type` | Description |
|---------------|-------------|
| `single_phase` | Isothermal poromechanics (pressure + mechanics) |
| `single_phase_thermal` | Thermoporoelasticity (pressure + temperature + mechanics) |

**Well configurations**

| `wells_type` | Description |
|-------------|-------------|
| `inj` | Single injection well (BHP control) |
| `prod` | Single production well (BHP control) |
| `doublet` | Injection + production pair (rate control) |

---

## Repository layout

Only source files tracked in git are listed below. Generated artefacts (`results/`, `meshes/NX_NY_NZ/`, `.spyproject/`) are local and not committed.

```
THM_vs_geomech_proxy/
├── main.py                      # Entry point: run THM simulation
├── main_proxy.py                # Entry point: run proxy & compare vs THM
├── model.py                     # THMCModel subclass (physics, wells, IC/BC)
├── reservoir.py                 # UnstructReservoirMech subclass (mesh, VTK output)
├── set_case.py                  # Dispatches case-specific input data
├── gen_msh.py                   # Gmsh structured 3-D box mesh generation
├── plot_vtk.py                  # PyVista / matplotlib VTK visualization
├── geomechanics.py              # Proxy utilities (deriv, HAS_GPU flag, fault/Mohr-Coulomb)
├── tools.py                     # Miscellaneous helpers
├── cases/
│   ├── base.py                  # input_data_base(), input_data_struct_like(), mesh coordinate tables
│   └── case_1.py                # input_data_case_1/2/3() definitions
└── meshes/
    └── case_1/
        └── mesh.msh             # Pre-generated mesh for the named cases (case_1/2/3)
```

**Proxy extension** (`_proxygeomech.pyd` on Windows / `_proxygeomech.so` on Linux) is **not included** in this repository. It must be compiled from the C++ source at:

> <https://gitlab.com/open-darts/subsidence-esmda/-/tree/main/proxy_geomech_cpp>

Place the compiled `.pyd`/`.so` (and optionally the CUDA variant `_proxygeomech_cuda.*`) in this directory before running `main_proxy.py`.

---

## Geometry cases

Two mesh strategies are supported, selected by the case name passed to `run()` / `run_geomech_proxy()`.

### Rectangular hexahedra, generated on the fly (`NX_NY_NZ` naming)

Cases named as `NX_NY_NZ` (e.g. `17_17_15`, `41_41_66`, `71_71_66`, `83_83_90`, `97_97_90`) use `cases/base.py::input_data_struct_like`. The mesh is built at runtime by Gmsh (`gen_msh.py`) as a structured grid of regular hexahedra with `generate_mesh=True` on the first run; subsequent runs reuse the saved `.msh` file.

- **Single mesh tag** (`99991`) covers the entire domain.
- Rock properties (permeability, porosity, Young's modulus) are assigned per cell by coordinate-based interpolation: reservoir-layer values inside the bounding box `[rsv_top, rsv_bottom] × [rsv_x1, rsv_x2] × [rsv_y1, rsv_y2]`, non-reservoir values elsewhere.
- Cell coordinates are refined near the reservoir and near well locations; the refinement pattern for each `NX` value is defined in `cases/base.py::_xc_from_nx`.

### Pre-generated mesh with multiple tags (`case_1`, `case_2`, `case_3`)

These cases share a committed mesh (`meshes/case_1/mesh.msh`) with **three Gmsh physical tags**: tag 1 = overburden, tag 2 = reservoir, tag 3 = underburden. Properties per tag can differ independently.

The input-data functions are layered on top of each other:

```
cases/base.py :: build_input_data()       ← shared geometry, fluid, OBL, well defaults
    └── cases/case_1.py :: input_data_case_1()   ← sets rsv bounds, perm=1000 mD, matrix_tags=(1,2,3)
            └── input_data_case_2()   ← overrides props per tag (set_props_by_tags=True)
                    └── input_data_case_3()   ← also sets per-tag heat capacity and thermal conductivity
```

| Case | Property assignment | Rock property regions |
|------|---------------------|-----------------------|
| `case_1` | Coordinate interpolation (`set_props_by_tags=False`) | Two regions: reservoir (cells inside `rsv_top/bottom/xy` bounds) and surroundings; each region gets its own perm, poro, E set in `build_input_data` |
| `case_2` | Per mesh tag (`set_props_by_tags=True`) | Three regions matching the Gmsh tags: overburden (tag 1), reservoir (tag 2), underburden (tag 3); vertical permeability is ×0.1 of horizontal |
| `case_3` | Per mesh tag | Same as case_2 + different heat capacity and thermal conductivity per tag (sand vs shale values) |

Using `set_props_by_tags=True` lets each tag carry its own full rock-property set, making it straightforward to represent any number of geological layers as long as the mesh is tagged accordingly.

---

## Running

### 1. THM simulation only

Edit the `if __name__ == '__main__':` block in `main.py` to select a case and physics, then:

```bash
python main.py
```

Key parameters at the top of that block:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cases` | `['case_3']` | Mesh/case names to run |
| `thermal` | `True` | `True` → `single_phase_thermal` + doublet wells |
| `n_years` | `30` | Simulation duration |
| `report_step` | `365.25/4` | VTK output frequency (days) |
| `generate_mesh` | `False` | Re-generate mesh with Gmsh if `True` |
| `decouple_geomech` | `True` | Disable mechanics→porosity feedback |
| `plot_vtk_timesteps` | `[0, -1]` | Timestep indices to visualise after run |

Output is written to `results/sol_cpp_<physics_type>_<wells_type>_<case>/`.

### Fault stability (FSP) post-processing

`run()` in `main.py` does this automatically at the end (`plot_fault=True`, function `postprocess_fault`): FSP appended to the fault files, slip time series, 2D `FSP`/`delta_FSP` maps per report step and dip profiles through every well. Cases without a fault mesh skip it. The 2D slices of `plot_vtk_timesteps` pass through `idata.other.plot_slice_origin` (`None`: mesh center). In `no_damage_zone*` the domain is `DOMAIN_W` × `DOMAIN_H` = 20 × 20 km (depth 5 km, 500 m extrusion layers along Y, set in `gen_fault_msh_no_damage_zone.py`) with the fault crossing mid-depth at X = `DOMAIN_W`/2. The doublet straddles the fault: injector 500 m west of it, producer 500 m east (1 km apart; `well_offset` in `cases/no_damage_zone.py`), both at Y = `DOMAIN_H`/2. Both request the depth range 2000–3000 m, and perforations are restricted to the reservoir tag. Each well is therefore perforated over the whole local reservoir layer, which the fault offsets: ~2450–2750 m at the injector, ~2250–2550 m at the producer; slices go through the injector row at 2575 m depth. Both wells are on mass-rate control at 8000 m³/day (`idata.other.well_rate_m3_day`, balanced doublet, no BHP limits), and the injected water is 40 K colder than the reservoir.

For cases with a fault (`no_damage_zone*`), `gen_fault_msh_no_damage_zone.py` writes `mesh.msh` (simulation) and `mesh_fault.msh` (same geometry with the fault surfaces tagged `FAULT = 9991`). During the run, `reservoir.save_fault_traction` writes the fault surface with the total-stress traction for every report step (`fault<N>.vtu`, `fault.pvd`). The traction is taken from the engine connection forces (Hooke + Biot + thermal) on the mesh faces lying on the fault. Then run:

```bash
python fault.py results/sol_cpp_single_phase_inj_no_damage_zone --friction 0.6 --cohesion 0
```

Per fault face, with `t` the traction, `n` the unit normal and `p` the pore pressure (`fault.fault_stability`):

```
sigma_n     = t . n                                  (compression positive)
tau         = |t - sigma_n n|
sigma_n_eff = sigma_n - p                            (full pore pressure; the fault has its own constitutive
                                                      behavior, so the matrix Biot coefficient is not used)
mcc         = tau - (cohesion + friction * sigma_n_eff)
FSP         = tau / sigma_n_eff                      (NaN where sigma_n_eff <= 0)
```

The run prints this computation step by step for the face with the largest FSP (`print_face_computation`).

This appends to every `fault<N>.vtu`: `sigma_n` (compression positive), `sigma_n_eff = sigma_n - p`, `tau`, `mcc = tau - (cohesion + friction * sigma_n_eff)` (> 0 means slip), `FSP = tau / sigma_n_eff` (NaN where `sigma_n_eff <= 0`), `delta_FSP` (change since the first step) and `slip`. It also writes `fault_slip.csv` and `fault_slip_vs_time.png` (FSP mean/max and slipping area [km², %] vs time), and 2D maps per report step `fault_plots/<field>_step<NNNN>.png` (fault faces unfolded onto the fault plane, along strike vs down dip, one color range per field over all steps). `--plot FSP mcc tau` selects the mapped fields (default `FSP`); `--plot` with no fields skips the maps. From Python: `plot_fault_case(case_dir, fields=('FSP', 'delta_FSP'))` or `plot_fault_field(vtu_filename, 'FSP')`. 1D profiles vs depth along the fault dip (pore pressure, temperature for thermal runs, normal stress total/effective, shear stress, Coulomb stress `tau - (c + mu sigma_n_eff)`, FSP) at one along-strike position, for several report steps: `--profile 4250 5750` (no value: at the face with the largest FSP), `--profile-steps 0 30`; from Python `plot_fault_dip_profiles(case_dir, strike=4250., steps=[0, 30])` or `point=[x, y, z]` (e.g. a well) instead of `strike`. Saves `fault_plots/dip_profile_strike<S>m.png` and `.csv`. To compare runs (files are only read): `compare_cases([dir1, dir2], labels=['homogeneous', 'heterogeneous'], frictions=(0.6, 1.2), png_filename='fault_slip_comparison.png')` plots slipping area and FSP max/mean vs time per case and friction.

### 2. THM + proxy comparison

Edit the `if __name__ == '__main__':` block in `main_proxy.py`, then:

```bash
python main_proxy.py
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `cases` | Same as `main.py` |
| `run_thm` | `True` to re-run the THM solver before proxy |
| `timestep_list` | VTK timestep indices to pass through the proxy |
| `modes` | List of analysis modes (see below) |
| `n_threads` | CPU thread count for proxy evaluation |
| `use_gpu` | `True` to use `_proxygeomech_cuda` |
| `read_from_cache` | `True` to reload proxy results from `.pkl` without recomputing |

**`modes` options**

| Mode | Output |
|------|--------|
| `plot_vertic_line` | 1-D vertical profiles: displacements, strains, stresses (THM vs proxy) |
| `plot_2d_slices` | 2-D XZ contour maps: displacements, effective/total stresses |
| `plot_horiz_line` | ΔP along the X axis at fixed Y, Z, over multiple timesteps |
| `print_at_point` | Console comparison at reservoir mid-point (includes analytical reference) |
| `check_initial` | Validate initial pressure and vertical stress against density-based gradients |

HTML and PDF summary reports (`proxy_vs_thm_2d.html`, `proxy_vs_thm_report.pdf`) are saved automatically in the timestep output folder.

---

## Key physics details

### THM model (`model.py`)

- Inherits `THMCModel` from open-DARTS.
- Fluid: single phase, single component (water). No phase equilibrium or compositional flash is needed; the single component fills the pore space with saturation = 1.
- Solver: CPR-preconditioned GMRES (`cpu_gmres_fs_cpr`).
- Mechanical equilibrium is found by running a large fictitious timestep (`dt = 1e8` days) before the actual simulation starts; subsequent displacements are reported relative to that initial state.
- Wells are placed by finding the mesh cell whose centroid is closest to the requested (X, Y, Z) perforation coordinate; well index is computed using the Peaceman formula.

**Optional: temperature-dependent fluid properties** (`model.py`)
The two classes below are defined in `model.py` but are **not active by default**. They can be enabled by overriding the property containers after `set_physics()` is called (see `set_physics_dummy` in `model.py` for an example):

| Class | Replaces | What it adds |
|-------|----------|--------------|
| `MaoDuan2009Shifted` | constant viscosity | State-dependent water viscosity via the Mao-Duan 2009 correlation. Wraps `MaoDuan2009` and shifts the model's relative temperature axis (baseline 0) to absolute Kelvin by adding `t_abs0=373.15 K` (100 °C), so the correlation always receives a physical temperature. |
| `DensityBasicTdep` | `DensityBasic` | Adds a linear thermal-expansion correction: `ρ(p,T) = ρ₀ · (1 + c·(p−p₀) − β·(T−T₀))`, where `β` is the volumetric thermal expansion coefficient. |

### Reservoir (`reservoir.py`)

- Inherits `UnstructReservoirMech`.
- Boundary conditions: roller (no normal displacement) on all lateral and bottom faces; free top surface.
- Thermal BC: aquifer (constant temperature) on all boundaries.
- Rock properties are assigned either by nearest-neighbour interpolation from a structured grid or per Gmsh physical tag.

### Geomechanical proxy (`_proxygeomech.pyd`)

The proxy solves the thermoporoelastic response analytically by integrating the Green's function for a rectangular prism source (Mindlin-type). It accepts:
- Source cell bounding boxes (prisms) and their ΔP, ΔT values from the THM VTK output.
- Evaluation point coordinates.

And returns displacements, strains, and effective/total stresses split into poroelastic and thermoelastic contributions. GPU acceleration is available via `_proxygeomech_cuda.pyd`.

---

## Output files

All output is written under `results/sol_cpp_<physics_type>_<wells_type>_<case>/`.

### THM simulation (`main.py`)

| File | Description |
|------|-------------|
| `solution<N>.vtu` | Per-timestep VTK unstructured grid (written by `reservoir.py`). Cell fields: `pressure` [bar], `temperature` [K], `ux`/`uy`/`uz` (displacements relative to the geomechanical equilibrium state) [m], `tot_stress` (total stress, 6-component Voigt) [bar], `eff_stress`, `delta_tot_stress`, `delta_eff_stress`, `delta_pressure`, `delta_temperature`, `strain`, `perm` (full 3×3 permeability tensor) [mD], `E` [bar], `poisson`, `poro`, `viscosity` [cP]. |
| `solution.pvd` | ParaView collection file referencing all `.vtu` timesteps. |
| `fault<N>.vtu`, `fault.pvd` | Fault surface (cases with `mesh_fault.msh` only). Cell fields: `traction` (total stress, compression positive) [bar], `normal`, `pressure` [bar], `temperature` [K] (thermal runs); after `fault.py` also `sigma_n`, `sigma_n_eff`, `tau`, `mcc` [bar], `FSP`, `slip`. |
| `fault_slip.csv`, `fault_slip_vs_time.png` | FSP (mean/max) and slipping fault area per report step, written by `fault.py`. |
| `wells.vtk` | Well tube geometry (one cylinder per perforation) for overlay in ParaView. |
| `plots_timestep_<N>/<arr>_slice_<xz,yz,xy>.png` (slices through the mesh center, or through `slice_origin=[x, y, z]` passed to `plot_vtk_pyvista`, e.g. a well at reservoir depth) | XZ slice plots produced by `plot_vtk.py` for each timestep index in `plot_vtk_timesteps`. One PNG per array: `u_x,m`, `u_y,m`, `u_z,m`, `temperature,K`, `pressure,MPa`, `delta_temperature,K`, `delta_pressure,MPa`, `viscosity,cP`, `perm_XX,mD`, `perm_ZZ,mD`, `delta_eff_stress_XX/YY/ZZ,MPa`, `delta_tot_stress_XX/YY/ZZ,MPa`. Arrays absent from the VTK file (e.g. `temperature` for isothermal runs) are skipped. |

### Proxy comparison (`main_proxy.py`)

Output goes into `results/.../plots_timestep_<N>/` for each processed VTK timestep index `N`.

| File | Mode | Description |
|------|------|-------------|
| `mesh_skeleton.png` | always | XZ view of mesh cell boundaries with reservoir outline and well locations. |
| `delta_pressure_along_x.png` | `plot_horiz_line` | ΔP [MPa] along the X axis at fixed Y, Z, one curve per reported timestep. |
| `dp_contour.png`, `dt_contour.png` | `plot_2d_slices` | ΔP and ΔT XZ contour maps interpolated from THM cell centres. |
| `<variable> - Proxy_contour.png` | `plot_2d_slices` | Proxy 2-D XZ field (displacements, effective/total stresses). |
| `<variable> - THM_contour.png` | `plot_2d_slices` | Same field from the THM VTK solution, interpolated to the same grid. |
| `<variable> - Difference_contour.png` | `plot_2d_slices` | Point-wise difference THM − Proxy on the XZ grid. |
| `<array>_<loc>_all.png` | `plot_vertic_line` | 1-D vertical profile of a field (displacement, stress, strain) at a reference XY location: THM (blue) vs proxy (red dashed). |
| `<array>_<loc>_all_diff.png` | `plot_vertic_line` | Absolute difference THM − Proxy for the same profile. |
| `proxy_vs_thm_2d.html` | `plot_2d_slices` / `plot_vertic_line` | Self-contained HTML report embedding all 2-D contour images and 1-D profile plots in a single page. |
| `proxy_vs_thm_report.pdf` | `plot_2d_slices` / `plot_vertic_line` | PDF version of the same report (one row per variable, columns: Proxy / THM / Difference for 2-D; one page per variable for 1-D). |
| `displs_stresses_prx.pkl` | `plot_2d_slices` | Cached proxy displacement and stress arrays. Reloaded on the next run when `read_from_cache=True`, skipping proxy recomputation. |

---

## Assumptions & known limitations

### THM solver

- **Single perforation per well** (`model.py`).
  The well-placement loop finds all cells that intersect the perforation interval but then breaks after adding the first one. Multi-perforation support is in progress on the `ilshat/mech_mswell` branch.

- **Geomechanics decoupled from flow by default** (`main.py`, `decouple_geomech=True`).
  The volumetric-strain transmissibility terms that feed mechanics back into porosity (and therefore pressure) are zeroed out. This is a common approximation for field-scale problems where the mechanical compaction signal is small compared to the fluid driving force.

- **Temperature scale is relative, not absolute** (`cases/base.py`, `MaoDuan2009Shifted`).
  The OBL range spans −50 to +50 K around a baseline of 0. `MaoDuan2009Shifted` maps that baseline to 373.15 K (100 °C) before calling the correlation. If the actual reservoir temperature differs significantly from 100 °C, `t_abs0` in `MaoDuan2009Shifted.__init__` must be adjusted.

- **Single-phase water physics** (`model.py`, `ModelProperties.evaluate`).
  `ModelProperties.evaluate` uses a trivial flash: water is assumed to be always single-phase liquid.

- **Initial temperature is uniform and equal to zero** (`cases/base.py`).
  `idata.initial.temperature_at_ref_depth = 0` and `idata.initial.temperature_gradient = 0`, so there is no geothermal gradient. The temperature field starts flat and evolves only through injection/production. Enabling a gradient requires setting both parameters and adjusting `t_abs0` in `MaoDuan2009Shifted` so the correlation receives a physically meaningful absolute temperature at every depth.

### Geomechanical proxy

- **Rectangular prism cell shapes** — the proxy integrates the nucleus-of-strain solution over axis-aligned rectangular prisms. Each source cell must be representable as a cuboid with faces perpendicular to X, Y, Z axes.

- **Mesh aligned with coordinate axes** — the proxy assumes the mesh is not rotated; tilted or curvilinear grids are not supported.

- **Homogeneous rock geomechanical properties** — a single set of elastic constants (Young's modulus, Poisson's ratio, Biot coefficient, thermal expansion coefficient) is used for the entire domain. Spatially varying elastic properties are not accounted for in the proxy.

- **2-D proxy comparison is evaluated at Y = 0 only** (`main_proxy.py`, `plot_2d_slices` mode).
  The XZ slice through the centre of the domain is the only plane compared. Off-centre or XY slices are not plotted.

- **Proxy source cells**: reservoir-only vs. full domain (`main_proxy.py`, `perm_non_rsv` / `poro_non_rsv`).
  When `perm_non_rsv ≤ 0` (truly impermeable overburden) the proxy integrates only over cells with `poro > poro_non_rsv`. When the surrounding rock has any permeability, pressure diffuses into it and all cells are used as sources. The threshold is currently hardcoded to `0`.
