import numpy as np
import pandas as pd

MW_CO2 = 44.01  # kg / kmol

def calculate_total_co2_in_reservoir_single_phase(model):
    n_vars = model.physics.n_vars
    n_res_blocks = model.reservoir.mesh.n_res_blocks

    X = np.array(model.physics.engine.X, copy=False).reshape((-1, n_vars))
    X_res = X[:n_res_blocks, :]

    volumes = np.array(model.reservoir.mesh.volume, copy=False)[:n_res_blocks]
    poro = np.array(model.reservoir.mesh.poro, copy=False)[:n_res_blocks]

    pc = model.physics.property_containers[0]

    total_co2_kmol = 0.0

    for i in range(n_res_blocks):
        state = X_res[i, :]
        pc.evaluate(state)

        # only one fluid phase and only one component: CO2
        molar_density = pc.dens_m[0]   # kmol/m3 fluid
        pv = poro[i] * volumes[i]      # m3 pore volume

        total_co2_kmol += molar_density * pv

    total_co2_mass_kg = total_co2_kmol * MW_CO2
    return total_co2_kmol, total_co2_mass_kg



class LGRInterfaceTransAnalyzer:
    """
    Analyze lateral coarse-fine interface transmissibilities in an assembled LGR model.

    Supported sides
    ---------------
    left, right, up, down

    Assumptions
    -----------
    1. model has already been assembled and initialized:
         - model.reservoir.mesh exists
         - model.lgrs exists
         - model.lgr_meta exists
         - model.level0 exists
    2. LGR is column-style patch refinement as in your current implementation
    3. Lateral interface is extracted exactly as in assemble_lgr_connections()
    """

    VALID_SIDES = ("left", "right", "up", "down")

    def __init__(self, model):
        self.model = model

        if not hasattr(model, "lgrs"):
            raise AttributeError("model has no attribute 'lgrs'")
        if not hasattr(model, "lgr_meta"):
            raise AttributeError("model has no attribute 'lgr_meta'")
        if not hasattr(model, "level0"):
            raise AttributeError("model has no attribute 'level0'")
        if not hasattr(model, "reservoir") or not hasattr(model.reservoir, "mesh"):
            raise AttributeError("model.reservoir.mesh is not available")

        # make sure level0 discretization/meta exist
        self.model.level0.discretize()
        self.disc0 = self.model.level0.discretizer
        self.g2l0 = np.asarray(self.disc0.global_to_local, dtype=int)
        self.l2g0 = np.asarray(self.disc0.local_to_global, dtype=int)

        self.nx0 = int(self.model.level0.nx)
        self.ny0 = int(self.model.level0.ny)
        self.nz0 = int(self.model.level0.nz)

        self.block_m = np.asarray(self.model.reservoir.mesh.block_m, dtype=int)
        self.block_p = np.asarray(self.model.reservoir.mesh.block_p, dtype=int)
        self.tran = np.asarray(self.model.reservoir.mesh.tran, dtype=float)

        self.conn_df = pd.DataFrame({
            "conn_id": np.arange(len(self.block_m), dtype=int),
            "block_m": self.block_m,
            "block_p": self.block_p,
            "tran": self.tran,
        })

    # ------------------------------------------------------------------
    # basic index helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _side_to_offset(side, nx):
        if side == "left":
            return -1
        elif side == "right":
            return +1
        elif side == "up":
            return -nx
        elif side == "down":
            return +nx
        else:
            raise ValueError(f"Unknown side: {side}")

    @staticmethod
    def _validate_side(side):
        if side not in ("left", "right", "up", "down"):
            raise ValueError(f"side must be one of left/right/up/down, got: {side}")

    @staticmethod
    def convert_ijk_to_gindex_1based(i_1b, j_1b, k_1b, nx, ny):
        """
        Convert 1-based structured (i,j,k) to 0-based flattened global index.
        """
        return (k_1b - 1) * nx * ny + (j_1b - 1) * nx + (i_1b - 1)

    def _coarse_global_neighbor_index(self, ic_1b, jc_1b, k_1b, side):
        """
        Return global structured index (before actnum squeeze) of the coarse neighbor
        adjacent to the refined parent cell at (ic_1b, jc_1b, k_1b).
        """
        if side == "left":
            return self.convert_ijk_to_gindex_1based(ic_1b - 1, jc_1b, k_1b, self.nx0, self.ny0)
        elif side == "right":
            return self.convert_ijk_to_gindex_1based(ic_1b + 1, jc_1b, k_1b, self.nx0, self.ny0)
        elif side == "up":
            return self.convert_ijk_to_gindex_1based(ic_1b, jc_1b - 1, k_1b, self.nx0, self.ny0)
        elif side == "down":
            return self.convert_ijk_to_gindex_1based(ic_1b, jc_1b + 1, k_1b, self.nx0, self.ny0)
        else:
            raise ValueError(f"Unknown side: {side}")

    def _coarse_local_neighbor_index(self, ic_1b, jc_1b, k_1b, side):
        """
        Convert coarse global structured index to level0 local active index.
        """
        g = self._coarse_global_neighbor_index(ic_1b, jc_1b, k_1b, side)
        l = int(self.g2l0[g])
        if l < 0:
            raise ValueError(
                f"Neighbor coarse block became inactive/unmapped: "
                f"(i,j,k)=({ic_1b},{jc_1b},{k_1b}), side={side}, global={g}, local={l}"
            )
        return l

    def _get_lgr_offset(self, lgr_name):
        if lgr_name not in self.model.lgr_meta["lgr_offsets"]:
            raise KeyError(f"{lgr_name} not found in model.lgr_meta['lgr_offsets']")
        return int(self.model.lgr_meta["lgr_offsets"][lgr_name])

    def _get_lgr_cfg(self, lgr_name):
        if lgr_name not in self.model.lgrs:
            raise KeyError(f"{lgr_name} not found in model.lgrs")
        return self.model.lgrs[lgr_name]["lgr_coords_in_parent_grid"]

    # ------------------------------------------------------------------
    # fine-face cell indices in assembled numbering
    # ------------------------------------------------------------------
    def fine_face_indices(self, lgr_name, side):
        """
        Return assembled global indices of fine cells on one lateral face of the LGR.

        Output
        ------
        list[dict] with keys:
            layer_local   : 0-based local layer inside the LGR
            k_1b          : 1-based global k in level0 indexing
            face_pos      : 0..(nface-1)
            fine_local    : local index inside this LGR
            fine_global   : assembled index in reservoir.mesh numbering
        """
        self._validate_side(side)

        cfg = self._get_lgr_cfg(lgr_name)
        offset = self._get_lgr_offset(lgr_name)

        i1 = int(cfg["i_range"][0])   # parent coarse cell center, 1-based
        j1 = int(cfg["j_range"][0])
        k1, k2 = map(int, cfg["k_range"])
        rx, ry, rz = map(int, cfg["refine"])

        if rz != 1:
            raise NotImplementedError("This class currently assumes refine[2] == 1")

        nk = k2 - k1 + 1
        plane_size = rx * ry
        rows = []

        for kk in range(nk):
            k_1b = k1 + kk
            base_fine = kk * plane_size

            if side == "left":
                ii = 0
                for jj in range(ry):
                    fine_local = base_fine + jj * rx + ii
                    rows.append({
                        "layer_local": kk,
                        "k_1b": k_1b,
                        "face_pos": jj,
                        "fine_local": fine_local,
                        "fine_global": offset + fine_local,
                    })

            elif side == "right":
                ii = rx - 1
                for jj in range(ry):
                    fine_local = base_fine + jj * rx + ii
                    rows.append({
                        "layer_local": kk,
                        "k_1b": k_1b,
                        "face_pos": jj,
                        "fine_local": fine_local,
                        "fine_global": offset + fine_local,
                    })

            elif side == "up":
                jj = 0
                for ii in range(rx):
                    fine_local = base_fine + jj * rx + ii
                    rows.append({
                        "layer_local": kk,
                        "k_1b": k_1b,
                        "face_pos": ii,
                        "fine_local": fine_local,
                        "fine_global": offset + fine_local,
                    })

            elif side == "down":
                jj = ry - 1
                for ii in range(rx):
                    fine_local = base_fine + jj * rx + ii
                    rows.append({
                        "layer_local": kk,
                        "k_1b": k_1b,
                        "face_pos": ii,
                        "fine_local": fine_local,
                        "fine_global": offset + fine_local,
                    })

        return rows

    # ------------------------------------------------------------------
    # interface connection extraction
    # ------------------------------------------------------------------
    def extract_face_connections(self, lgr_name, side, require_unique=True):
        """
        Extract all coarse-fine lateral connections on one face of one LGR from the
        assembled reservoir.mesh connection list.

        Returns
        -------
        DataFrame with columns:
            conn_id, block_m, block_p, tran,
            coarse_local, fine_global,
            layer_local, k_1b, face_pos,
            coarse_is_m, fine_is_p
        """
        self._validate_side(side)

        cfg = self._get_lgr_cfg(lgr_name)
        i1 = int(cfg["i_range"][0])
        j1 = int(cfg["j_range"][0])

        face_rows = self.fine_face_indices(lgr_name, side)
        out_rows = []

        for row in face_rows:
            k_1b = int(row["k_1b"])
            fine_global = int(row["fine_global"])
            coarse_local = self._coarse_local_neighbor_index(i1, j1, k_1b, side)

            mask = (
                ((self.conn_df["block_m"] == coarse_local) & (self.conn_df["block_p"] == fine_global)) |
                ((self.conn_df["block_m"] == fine_global) & (self.conn_df["block_p"] == coarse_local))
            )
            hit = self.conn_df.loc[mask].copy()

            # if require_unique and len(hit) != 1:
            #     raise ValueError(
            #         f"Expected exactly one connection for "
            #         f"LGR={lgr_name}, side={side}, k_1b={k_1b}, "
            #         f"coarse_local={coarse_local}, fine_global={fine_global}, "
            #         f"but found {len(hit)}"
            #     )

            if hit.empty:
                continue

            # hit["lgr_name"] = lgr_name
            # hit["side"] = side
            # hit["coarse_local"] = coarse_local
            # hit["fine_global"] = fine_global
            # hit["layer_local"] = int(row["layer_local"])
            # hit["k_1b"] = k_1b
            # hit["face_pos"] = int(row["face_pos"])
            # hit["coarse_is_m"] = (hit["block_m"] == coarse_local).astype(bool)
            # hit["fine_is_p"] = (hit["block_p"] == fine_global).astype(bool)
            # hit = hit.copy().reset_index(drop=True)

            hit["lgr_name"] = lgr_name
            hit["side"] = side
            hit["coarse_local"] = coarse_local
            hit["fine_global"] = fine_global
            hit["layer_local"] = int(row["layer_local"])
            hit["k_1b"] = k_1b
            hit["face_pos"] = int(row["face_pos"])
            hit["coarse_is_m"] = (hit["block_m"] == coarse_local).astype(bool)
            hit["fine_is_p"] = (hit["block_p"] == fine_global).astype(bool)
            hit["pair_conn_idx"] = np.arange(len(hit), dtype=int)

            out_rows.append(hit)

        if not out_rows:
            return pd.DataFrame(columns=[
                "conn_id", "block_m", "block_p", "tran",
                "lgr_name", "side",
                "coarse_local", "fine_global",
                "layer_local", "k_1b", "face_pos",
                "coarse_is_m", "fine_is_p",
            ])

        out = pd.concat(out_rows, axis=0, ignore_index=True)
        out = out.sort_values(["k_1b", "face_pos"]).reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    # summaries
    # ------------------------------------------------------------------
    def summarize_face(self, lgr_name, side):
        """
        Summary statistics for one LGR face.

        Returns dict with:
            - connection_df
            - per_layer_df
            - summary
        """
        df = self.extract_face_connections(lgr_name, side)

        if df.empty:
            return {
                "connection_df": df,
                "per_layer_df": pd.DataFrame(),
                "summary": {
                    "lgr_name": lgr_name,
                    "side": side,
                    "count": 0,
                    "sum_T": 0.0,
                    "mean_T": np.nan,
                    "min_T": np.nan,
                    "max_T": np.nan,
                }
            }

        per_layer = (
            df.groupby("k_1b", as_index=False)
              .agg(
                  count=("tran", "size"),
                  sum_T=("tran", "sum"),
                  mean_T=("tran", "mean"),
                  min_T=("tran", "min"),
                  max_T=("tran", "max"),
              )
              .sort_values("k_1b")
              .reset_index(drop=True)
        )

        summary = {
            "lgr_name": lgr_name,
            "side": side,
            "count": int(len(df)),
            "sum_T": float(df["tran"].sum()),
            "mean_T": float(df["tran"].mean()),
            "min_T": float(df["tran"].min()),
            "max_T": float(df["tran"].max()),
        }

        return {
            "connection_df": df,
            "per_layer_df": per_layer,
            "summary": summary,
        }

    def summarize_lgr(self, lgr_name):
        """
        Summarize all four lateral faces of one LGR.
        """
        rows = []
        per_face = {}

        for side in self.VALID_SIDES:
            out = self.summarize_face(lgr_name, side)
            per_face[side] = out
            rows.append(out["summary"])

        summary_df = pd.DataFrame(rows)
        return {
            "lgr_name": lgr_name,
            "faces": per_face,
            "summary_df": summary_df,
        }

    def summarize_all_lgrs(self):
        """
        Summarize all LGR patches in the model.
        """
        all_rows = []
        out = {}

        for lgr_name in self.model.lgr_meta["lgr_orders"]:
            res = self.summarize_lgr(lgr_name)
            out[lgr_name] = res
            all_rows.append(res["summary_df"])

        if all_rows:
            combined = pd.concat(all_rows, axis=0, ignore_index=True)
        else:
            combined = pd.DataFrame()

        out["summary_df"] = combined
        return out


class FineEffectiveTransAnalyzer:
    """
    Compute per-layer effective transmissibility for a fine structured model
    around a square patch, then aggregate over layers.

    Definitions
    -----------
    For each side and each layer k0:
      - interface region: 1 column/row on the patch boundary
      - neighbor region : n_nb_cols columns/rows outside the interface
      - flux            : only immediate interface links (1 column/row)
      - dp_macro        : p_if_avg(layer) - p_nb_avg(layer)
      - T_eff_layer     : - total_flux_layer / (lambda_ref_layer * dp_macro_layer)

    Structured indexing
    -------------------
    g = i0 + j0 * nx + k0 * nx * ny
    """

    VALID_SIDES = ("left", "right", "up", "down")
    VALID_AGG = ("average", "sum", "weighted_average")

    def __init__(
        self,
        darts_model,
        nx,
        ny,
        nz,
        patch_size=5,
        patch_center_1b=None,
        reservoir_k0_range=None,
        n_nb_cols=5,
    ):
        self.model = darts_model
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.patch_size = int(patch_size)
        self.n_nb_cols = int(n_nb_cols)

        if self.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")
        if self.n_nb_cols < 1:
            raise ValueError("n_nb_cols must be >= 1")
        if patch_center_1b is None:
            raise ValueError("patch_center_1b must be provided as (i_center_1b, j_center_1b)")
        if reservoir_k0_range is None:
            raise ValueError("reservoir_k0_range must be provided, e.g. range(1, 8)")

        self.i_center_1b = int(patch_center_1b[0])
        self.j_center_1b = int(patch_center_1b[1])
        self.reservoir_k0_range = list(reservoir_k0_range)

    # ------------------------------------------------------------------
    # basic arrays
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(vec, dtype=None):
        arr = np.asarray(vec)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def get_connection_arrays(self):
        mesh = self.model.reservoir.mesh
        block_m = self._to_numpy(mesh.block_m, dtype=int)
        block_p = self._to_numpy(mesh.block_p, dtype=int)
        tran = self._to_numpy(mesh.tran, dtype=float)
        return block_m, block_p, tran

    def get_pressure_array(self, reservoir_only=True):
        n_vars = len(self.model.physics.vars)
        X = self._to_numpy(self.model.physics.engine.X, dtype=float).reshape((-1, n_vars))
        pressure = X[:, 0].copy()

        if reservoir_only:
            if hasattr(self.model.reservoir.mesh, "n_res_blocks"):
                n_res = int(self.model.reservoir.mesh.n_res_blocks)
            else:
                n_res = int(self.model.reservoir.n)
            pressure = pressure[:n_res]

        return pressure

    def build_connection_dataframe(self):
        block_m, block_p, tran = self.get_connection_arrays()
        return pd.DataFrame({
            "conn_id": np.arange(len(block_m), dtype=int),
            "block_m": block_m,
            "block_p": block_p,
            "tran": tran,
        })

    # ------------------------------------------------------------------
    # indexing helpers
    # ------------------------------------------------------------------
    def lin_index(self, i0, j0, k0):
        return i0 + j0 * self.nx + k0 * self.nx * self.ny

    def _patch_bounds_0b(self):
        ic = self.i_center_1b - 1
        jc = self.j_center_1b - 1
        half = self.patch_size // 2

        i0_min = ic - half
        i0_max = ic + half
        j0_min = jc - half
        j0_max = jc + half

        return i0_min, i0_max, j0_min, j0_max

    def interface_cells_layer(self, side, k0):
        if side not in self.VALID_SIDES:
            raise ValueError(f"side must be one of {self.VALID_SIDES}")

        i0_min, i0_max, j0_min, j0_max = self._patch_bounds_0b()
        ids = []

        if side == "left":
            i0 = i0_min
            for j0 in range(j0_min, j0_max + 1):
                ids.append(self.lin_index(i0, j0, k0))

        elif side == "right":
            i0 = i0_max
            for j0 in range(j0_min, j0_max + 1):
                ids.append(self.lin_index(i0, j0, k0))

        elif side == "up":
            j0 = j0_min
            for i0 in range(i0_min, i0_max + 1):
                ids.append(self.lin_index(i0, j0, k0))

        elif side == "down":
            j0 = j0_max
            for i0 in range(i0_min, i0_max + 1):
                ids.append(self.lin_index(i0, j0, k0))

        return ids

    def neighbor_cells_layer(self, side, k0):
        """
        Per-layer neighbor support region:
        outside the interface, with thickness = n_nb_cols.
        """
        if side not in self.VALID_SIDES:
            raise ValueError(f"side must be one of {self.VALID_SIDES}")

        i0_min, i0_max, j0_min, j0_max = self._patch_bounds_0b()
        ids = []
        s = self.n_nb_cols

        if side == "left":
            for j0 in range(j0_min, j0_max + 1):
                for i0 in range(i0_min - s, i0_min):
                    ids.append(self.lin_index(i0, j0, k0))

        elif side == "right":
            for j0 in range(j0_min, j0_max + 1):
                for i0 in range(i0_max + 1, i0_max + 1 + s):
                    ids.append(self.lin_index(i0, j0, k0))

        elif side == "up":
            for j0 in range(j0_min - s, j0_min):
                for i0 in range(i0_min, i0_max + 1):
                    ids.append(self.lin_index(i0, j0, k0))

        elif side == "down":
            for j0 in range(j0_max + 1, j0_max + 1 + s):
                for i0 in range(i0_min, i0_max + 1):
                    ids.append(self.lin_index(i0, j0, k0))

        return ids

    def immediate_neighbor_cells_layer(self, side, k0):
        """
        Immediate cells across the interface, one-to-one with interface cells.
        These are the cells used for linkwise flux.
        """
        if_cells = self.interface_cells_layer(side, k0)

        if side == "left":
            return [c - 1 for c in if_cells]
        elif side == "right":
            return [c + 1 for c in if_cells]
        elif side == "up":
            return [c - self.nx for c in if_cells]
        elif side == "down":
            return [c + self.nx for c in if_cells]
        else:
            raise ValueError(f"Unknown side: {side}")

    # ------------------------------------------------------------------
    # properties and mobility
    # ------------------------------------------------------------------
    def get_pressure_of_cells(self, cell_indices):
        pressure = self.get_pressure_array(reservoir_only=True)
        cell_indices = np.asarray(cell_indices, dtype=int)

        if np.any(cell_indices < 0) or np.any(cell_indices >= len(pressure)):
            raise IndexError("cell_indices contains out-of-range indices")

        return pd.DataFrame({
            "cell_idx": cell_indices,
            "pressure": pressure[cell_indices]
        })

    def _eval_mu_of_cell(self, cell_idx):
        n_vars = len(self.model.physics.vars)
        X = np.asarray(self.model.physics.engine.X, dtype=float).reshape((-1, n_vars))
        state = X[cell_idx, :].copy()

        pc = self.model.physics.property_containers[0]
        pc.evaluate(state)

        if len(pc.mu) < 1:
            raise RuntimeError("No viscosity found in property container.")

        return float(pc.mu[0])

    def _connection_mobility(self, cell_a, cell_b, mode="cell_a"):
        mu_a = self._eval_mu_of_cell(cell_a)
        mu_b = self._eval_mu_of_cell(cell_b)

        if mode == "cell_a":
            mu = mu_a
        elif mode == "cell_b":
            mu = mu_b
        elif mode == "arithmetic":
            mu = 0.5 * (mu_a + mu_b)
        elif mode == "harmonic":
            mu = 2.0 / (1.0 / mu_a + 1.0 / mu_b)
        else:
            raise ValueError(f"Unknown mobility mode: {mode}")

        lam = 1.0 / mu
        return lam, mu_a, mu_b, mu

    # ------------------------------------------------------------------
    # connection lookup
    # ------------------------------------------------------------------
    def find_connection_between_two_cells(self, cell_a, cell_b):
        df = self.build_connection_dataframe()

        mask = (
            ((df["block_m"] == cell_a) & (df["block_p"] == cell_b)) |
            ((df["block_m"] == cell_b) & (df["block_p"] == cell_a))
        )
        return df.loc[mask].reset_index(drop=True)

    def flux_between_two_cells(self, cell_a, cell_b, mobility_mode="cell_a"):
        df = self.find_connection_between_two_cells(cell_a, cell_b)

        if df.empty:
            raise ValueError(f"Cells {cell_a} and {cell_b} are not connected.")

        if len(df) > 1:
            tran_vals = df["tran"].to_numpy(dtype=float)
            if not np.allclose(tran_vals, tran_vals[0], rtol=1e-10, atol=1e-12):
                raise ValueError(
                    f"Cells {cell_a} and {cell_b} have multiple connections "
                    f"with different tran values: {tran_vals}"
                )

        tran = float(df["tran"].iloc[0])

        pressure = self.get_pressure_array(reservoir_only=True)
        p_a = float(pressure[cell_a])
        p_b = float(pressure[cell_b])
        dp = p_a - p_b

        # if dp < 0:
        #     raise ValueError(
        #         f"Expected cell_a (interface) to be high-pressure side, "
        #         f"but got dp = {dp} < 0 for cells {cell_a}, {cell_b}"
        #     )

        lam, mu_a, mu_b, mu_ref = self._connection_mobility(cell_a, cell_b, mode=mobility_mode)
        q = - tran * dp * lam

        return {
            "cell_a": int(cell_a),
            "cell_b": int(cell_b),
            "pressure_a": p_a,
            "pressure_b": p_b,
            "dp_a_minus_b": dp,
            "tran": tran,
            "lambda_ref": lam,
            "mu_a": mu_a,
            "mu_b": mu_b,
            "mu_ref": mu_ref,
            "flux": q,
            "n_connections_found": int(len(df)),
        }

    # ------------------------------------------------------------------
    # per-layer calculations
    # ------------------------------------------------------------------
    def layer_pressures(self, side, k0):
        nb_cells = self.neighbor_cells_layer(side, k0)
        if_cells = self.interface_cells_layer(side, k0)

        p_nb = self.get_pressure_of_cells(nb_cells)
        p_if = self.get_pressure_of_cells(if_cells)

        return {
            "k0": int(k0),
            "side": side,
            "neighbor_cells": nb_cells,
            "interface_cells": if_cells,
            "p_nb_df": p_nb,
            "p_if_df": p_if,
            "p_nb_avg": float(p_nb["pressure"].mean()),
            "p_if_avg": float(p_if["pressure"].mean()),
        }

    def collect_layer_face_connections(self, side, k0):
        if_cells = self.interface_cells_layer(side, k0)
        nb_cells = self.immediate_neighbor_cells_layer(side, k0)

        rows = []
        for c_nb, c_if in zip(nb_cells, if_cells):
            conn_df = self.find_connection_between_two_cells(c_nb, c_if)

            if conn_df.empty:
                rows.append({
                    "k0": int(k0),
                    "side": side,
                    "cell_nb": int(c_nb),
                    "cell_if": int(c_if),
                    "connected": False,
                    "tran": np.nan,
                })
            else:
                rows.append({
                    "k0": int(k0),
                    "side": side,
                    "cell_nb": int(c_nb),
                    "cell_if": int(c_if),
                    "connected": True,
                    "tran": float(conn_df["tran"].iloc[0]),
                })

        return pd.DataFrame(rows)

    def layer_total_flux(self, side, k0, mobility_mode="cell_a"):
        conn_df = self.collect_layer_face_connections(side, k0)
        conn_df = conn_df[conn_df["connected"]].copy()

        flux_rows = []
        total_q = 0.0

        for _, row in conn_df.iterrows():
            out = self.flux_between_two_cells(
                int(row["cell_if"]),
                int(row["cell_nb"]),
                mobility_mode=mobility_mode,
            )
            total_q += out["flux"]

            flux_rows.append({
                "k0": int(k0),
                "side": side,
                "cell_nb": int(row["cell_nb"]),
                "cell_if": int(row["cell_if"]),
                "tran": out["tran"],
                "pressure_if": out["pressure_a"],
                "pressure_nb": out["pressure_b"],
                "dp_if_minus_nb": out["dp_a_minus_b"],
                "mu_ref_link": out["mu_ref"],
                "lambda_ref_link": out["lambda_ref"],
                "flux": out["flux"],
            })

        return {
            "k0": int(k0),
            "side": side,
            "n_links": int(len(flux_rows)),
            "total_flux": float(total_q),
            "link_df": pd.DataFrame(flux_rows),
        }

    def effective_trans_of_layer(
        self,
        side,
        k0,
        mobility_mode="cell_a",
        ref_mu_mode="interface_avg",
    ):
        pinfo = self.layer_pressures(side, k0)
        finfo = self.layer_total_flux(side, k0, mobility_mode=mobility_mode)

        dp_macro = pinfo["p_if_avg"] - pinfo["p_nb_avg"]
        if abs(dp_macro) < 1e-14:
            raise ZeroDivisionError(
                f"Macro pressure drop too small for side={side}, k0={k0}."
            )

        link_df = finfo["link_df"]
        if link_df.empty:
            raise RuntimeError(f"No valid interface links found for side={side}, k0={k0}.")

        if ref_mu_mode == "neighbor_avg":
            mu_ref = float(np.mean([self._eval_mu_of_cell(c) for c in link_df["cell_nb"]]))
        elif ref_mu_mode == "interface_avg":
            mu_ref = float(np.mean([self._eval_mu_of_cell(c) for c in link_df["cell_if"]]))
        else:
            raise ValueError(f"Unknown ref_mu_mode: {ref_mu_mode}")

        lam_ref = 1.0 / mu_ref
        total_flux = finfo["total_flux"]

        T_layer_total = - total_flux / (lam_ref * dp_macro)
        T_layer_avglink = T_layer_total / finfo["n_links"]

        return {
            "k0": int(k0),
            "side": side,
            "n_links": int(finfo["n_links"]),
            "p_nb_avg": float(pinfo["p_nb_avg"]),
            "p_if_avg": float(pinfo["p_if_avg"]),
            "dp_macro": float(dp_macro),
            "mu_ref": float(mu_ref),
            "lambda_ref": float(lam_ref),
            "total_flux": float(total_flux),
            "T_layer_total": float(T_layer_total),
            "T_layer_avglink": float(T_layer_avglink),
            "link_df": link_df,
            "neighbor_pressure_df": pinfo["p_nb_df"],
            "interface_pressure_df": pinfo["p_if_df"],
        }

    def effective_trans_of_face_per_layer(
        self,
        side,
        mobility_mode="cell_a",
        ref_mu_mode="neighbor_avg",
    ):
        rows = []
        detail = {}

        for k0 in self.reservoir_k0_range:
            out = self.effective_trans_of_layer(
                side=side,
                k0=k0,
                mobility_mode=mobility_mode,
                ref_mu_mode=ref_mu_mode,
            )
            detail[int(k0)] = out
            rows.append({
                "k0": int(k0),
                "side": side,
                "n_links": out["n_links"],
                "p_nb_avg": out["p_nb_avg"],
                "p_if_avg": out["p_if_avg"],
                "dp_macro": out["dp_macro"],
                "mu_ref": out["mu_ref"],
                "lambda_ref": out["lambda_ref"],
                "total_flux": out["total_flux"],
                "T_layer_total": out["T_layer_total"],
                "T_layer_avglink": out["T_layer_avglink"],
            })

        per_layer_df = pd.DataFrame(rows).sort_values("k0").reset_index(drop=True)
        return {
            "side": side,
            "per_layer": detail,
            "per_layer_df": per_layer_df,
        }

    # ------------------------------------------------------------------
    # aggregation over layers
    # ------------------------------------------------------------------
    def aggregate_layers(self, per_layer_df, mode="average"):
        if mode not in self.VALID_AGG:
            raise ValueError(f"mode must be one of {self.VALID_AGG}")

        df = per_layer_df.copy()
        if df.empty:
            raise RuntimeError("per_layer_df is empty.")

        if mode == "average":
            return {
                "agg_mode": mode,
                "n_layers": int(len(df)),
                "n_links": int(df["n_links"].sum()),
                "p_nb_avg": float(df["p_nb_avg"].mean()),
                "p_if_avg": float(df["p_if_avg"].mean()),
                "dp_macro": float(df["dp_macro"].mean()),
                "mu_ref": float(df["mu_ref"].mean()),
                "lambda_ref": float(df["lambda_ref"].mean()),
                "total_flux": float(df["total_flux"].mean()),
                "T_face_total": float(df["T_layer_total"].mean()),
                "T_face_avglink": float(df["T_layer_avglink"].mean()),
            }

        elif mode == "sum":
            return {
                "agg_mode": mode,
                "n_layers": int(len(df)),
                "n_links": int(df["n_links"].sum()),
                "p_nb_avg": float(df["p_nb_avg"].sum()),
                "p_if_avg": float(df["p_if_avg"].sum()),
                "dp_macro": float(df["dp_macro"].sum()),
                "mu_ref": float(df["mu_ref"].sum()),
                "lambda_ref": float(df["lambda_ref"].sum()),
                "total_flux": float(df["total_flux"].sum()),
                "T_face_total": float(df["T_layer_total"].sum()),
                "T_face_avglink": float(df["T_layer_avglink"].sum()),
            }


    def effective_trans_of_face(
        self,
        side,
        mobility_mode="cell_a",
        ref_mu_mode="neighbor_avg",
        agg_mode="average",
    ):
        face_out = self.effective_trans_of_face_per_layer(
            side=side,
            mobility_mode=mobility_mode,
            ref_mu_mode=ref_mu_mode,
        )
        agg = self.aggregate_layers(face_out["per_layer_df"], mode=agg_mode)

        return {
            "side": side,
            **agg,
            "per_layer_df": face_out["per_layer_df"],
            "per_layer": face_out["per_layer"],
        }

    def effective_trans_all_faces(
        self,
        mobility_mode="cell_a",
        ref_mu_mode="interface_avg",
        agg_mode="average",
    ):
        results = {}
        rows = []

        for side in self.VALID_SIDES:
            out = self.effective_trans_of_face(
                side=side,
                mobility_mode=mobility_mode,
                ref_mu_mode=ref_mu_mode,
                agg_mode=agg_mode,
            )
            results[side] = out
            rows.append({
                "side": side,
                "agg_mode": out["agg_mode"],
                "n_layers": out["n_layers"],
                "n_links": out["n_links"],
                "p_nb_avg": out["p_nb_avg"],
                "p_if_avg": out["p_if_avg"],
                "dp_macro": out["dp_macro"],
                "mu_ref": out["mu_ref"],
                "lambda_ref": out["lambda_ref"],
                "total_flux": out["total_flux"],
                "T_face_total": out["T_face_total"],
                "T_face_avglink": out["T_face_avglink"],
            })

        results["summary_df"] = pd.DataFrame(rows)
        return results