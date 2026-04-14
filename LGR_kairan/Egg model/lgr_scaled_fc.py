import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tran_fc import compute_eff_tran_map_for_lgrs

def side_neighbor_globals(model, ic, jc, k):
    """
    Return 0-based global structured coarse neighbor indices around parent coarse cell (ic,jc,k),
    where ic,jc,k are 1-based.
    """
    return {
        "left":  model.convert_ijk_to_gindex_1based(ic - 1, jc,     k, model.level0.nx, model.level0.ny),
        "right": model.convert_ijk_to_gindex_1based(ic + 1, jc,     k, model.level0.nx, model.level0.ny),
        "up":    model.convert_ijk_to_gindex_1based(ic,     jc - 1, k, model.level0.nx, model.level0.ny),
        "down":  model.convert_ijk_to_gindex_1based(ic,     jc + 1, k, model.level0.nx, model.level0.ny),
    }

def rebuild_raw_lateral_fc_from_imag(model, lgr_orders, lgr_offsets):
    model.level0.discretize()
    disc0 = model.level0.discretizer
    g2l0 = np.asarray( disc0.global_to_local, dtype=int)

    rows = []
    for name in lgr_orders:
        cfg = model.lgrs[name]
        ic = cfg['lgr_coords_in_parent_grid']['i_range'][0]
        jc = cfg['lgr_coords_in_parent_grid']['j_range'][0]
        k1, k2 = cfg['lgr_coords_in_parent_grid']['k_range']
        rx,ry,rz = cfg['lgr_coords_in_parent_grid']['refine']
        

        model.level1_imag[name].discretize()
        disc_im = model.level1_imag[name].discretizer
        cmi, cpi, Ti, Ti_therm = disc_im.calc_structured_discr()
        
        cmi = np.asarray(cmi, dtype=int)
        cpi = np.asarray(cpi, dtype=int)
        Ti = np.asarray(Ti, dtype=float)
        Ti_therm = np.asarray(Ti_therm, dtype=float)

        nxim = rx +2
        nyim = ry +2
        nxy_im = nxim * nyim
        plane_size = rx * ry
        fine_global_offset = lgr_offsets[name]

        # build per layer fine/ halo index map 
        fine_im_set = set()
        coarse_im_set = set()
        imag_fine_to_global = {} # this global means consider actnum and concatenate lgr to the end of level0
        image_ring_to_level0 = {}
        imag_side_info = {}

        for kk, k_1b in enumerate(range(k1, k2 +1)): # k_1b means 1-based layer index in level0
            nbr_g = side_neighbor_globals(model, ic, jc, k_1b)
            nbr_l = {k: int(g2l0[v]) for k, v in nbr_g.items()} # local indices in level0 after actnum squeeze

            base_im = kk * nxy_im
            base_fine = kk * plane_size
            
            # central fine cells in imaginary grid
            
            for j in range(1, ry+1):
                for i in range(1, rx+1):
                    imag_idx = base_im + j*nxim + i # index of 5*5 grid
                    fine_local = base_fine + (j-1)*rx + (i-1) # local fine index in level1
                    fine_global = fine_global_offset + fine_local # global index of fine cell

                    imag_fine_to_global[imag_idx] = fine_global
                    fine_im_set.add(imag_idx)
                    
            # left and right halo cells
            for jj in range(1,ry+1):
                left_idx = base_im + jj*nxim + 0
                right_idx = base_im + jj*nxim + (rx+1)
                image_ring_to_level0[left_idx] = nbr_l['left']
                image_ring_to_level0[right_idx] = nbr_l['right']
                imag_side_info[left_idx] = ("left", jj-1, k_1b, kk)
                imag_side_info[right_idx] = ("right", jj-1, k_1b, kk)
                coarse_im_set.add(left_idx)
                coarse_im_set.add(right_idx)
            # up and down halo cells
            for ii in range(1,rx+1):
                up_idx = base_im + 0*nxim + ii
                down_idx = base_im + (ry+1)*nxim + ii
                image_ring_to_level0[up_idx] = nbr_l['up']
                image_ring_to_level0[down_idx] = nbr_l['down']
                imag_side_info[up_idx] = ("up", ii-1, k_1b, kk)
                imag_side_info[down_idx] = ("down", ii-1, k_1b, kk)
                coarse_im_set.add(up_idx)
                coarse_im_set.add(down_idx)

        for cm, cp, t, tt in zip(cmi, cpi, Ti, Ti_therm):
            if (cm // nxy_im) != (cp // nxy_im):
                continue
            cm_is_fine = cm in fine_im_set
            cp_is_fine = cp in fine_im_set
            cm_is_coarse = cm in coarse_im_set
            cp_is_coarse = cp in coarse_im_set

            # fine-coarse connection
            if cm_is_fine and cp_is_coarse:
                fine_global = imag_fine_to_global[cm]
                coarse_local = image_ring_to_level0[cp]
                side, face_pos, k_1b, kk = imag_side_info[cp]

            elif cp_is_fine and cm_is_coarse:
                fine_global = imag_fine_to_global[cp]
                coarse_local = image_ring_to_level0[cm]
                side, face_pos, k_1b, kk = imag_side_info[cm]
            else:
                continue
            rows.append({
                'lgr_name': name,
                'side': side,
                'face_pos': face_pos,
                'layer_local': kk, # the sequence of the layer in the reservoir, 0-based
                'layer_global': k_1b, # the layer index in level0, 1-based
                'coarse_local': coarse_local, # local index in level0 after actnum squeeze
                'fine_global': fine_global, # global index of fine cell considering actnum and concatenate lgr to the end of level0
                "raw_T": float(t), # the raw transmissibility calculated from the imaginary grid discretization
                "raw_T_therm": float(tt),
            })
            df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["lgr_name", "layer_global", "side", "face_pos", "fine_global"]
        ).reset_index(drop=True)
    return df

def build_flow_based_scaled_fc(
        model,
        lgr_orders,
        lgr_offsets,
        verbose=True
):
    eff_tran_map, eff_detail = compute_eff_tran_map_for_lgrs(
    model=model,
    lgr_orders=lgr_orders,
    coarse_patch_size=5,
    run_days=365.0,
    n_steps=20,
    mobility_mode="interface_avg",
    tag_filter=None,   
    )
    raw_df = rebuild_raw_lateral_fc_from_imag(model, lgr_orders, lgr_offsets)
    out_rows = []
    grouped = raw_df.groupby(['lgr_name', 'layer_global', 'side'], sort=True)
    
    for (lgr_name, layer_global, side), g in grouped:
        g = g.copy().reset_index(drop=True)

        try:
            T_target_avg_link = float(eff_tran_map[lgr_name][layer_global][side])
        except KeyError as e:
            raise KeyError(
            f'missing effective transmissibility for lgr {lgr_name} layer {layer_global} side {side} in eff_tran_map'
            ) from e
        g['scaled_tran'] = T_target_avg_link
        g['scaled_tran_thermal'] = g['raw_T_therm']
        out_rows.append(g)

    out_df = pd.concat(out_rows, axis=0, ignore_index=True)
    out_df = out_df.sort_values(
        ["lgr_name", "layer_global", "side", "face_pos", "fine_global"]
    ).reset_index(drop=True)

    if verbose:
        summary = (
            out_df.groupby(["lgr_name", "layer_global", "side"], as_index=False)
            .agg(
                n_links=("scaled_tran", "size"),
                raw_total=("raw_T", "sum"),
                scaled_total=("scaled_tran", "sum"),
            )
        )
        print("\n=== Scaled lateral FC summary ===")
        print(summary)

    fc_cm = out_df["coarse_local"].to_numpy(dtype=int)
    fc_cp = out_df["fine_global"].to_numpy(dtype=int)
    fc_T = out_df["scaled_tran"].to_numpy(dtype=float)
    fc_Tt = out_df["scaled_tran_thermal"].to_numpy(dtype=float)

    return fc_cm, fc_cp, fc_T, fc_Tt, out_df