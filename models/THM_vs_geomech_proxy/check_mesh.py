"""
Consistency checks for the tetrahedral meshes used by this model.

The discretizer builds a per-cell least-squares gradient from the connections of
a cell: one per neighbouring cell and one per tagged boundary face. It needs
those directions to span 3D, so a mesh can be perfectly valid geometrically and
still be unusable. An untagged domain face gives the cell behind it no boundary
connection; if the remaining connections are coplanar, A^T*A is singular and its
inverse comes out non-finite -- the "gradient least-squares failed" report in
discretizer/src/discretizer.cpp.

Running these checks takes seconds; discovering the same problem through a run
costs the mesh read plus the full discretization first.

Usage:
    python check_mesh.py meshes/case_5/mesh.msh
    python check_mesh.py meshes/case_5/mesh.msh --case case_5
"""

import argparse
import collections
import itertools
import sys

import numpy as np

FLAT_TOL = 1e-6


def tet_faces(tet):
    """
    Every face of every tet, as sorted node triples.

    Row ``i`` belongs to cell ``i % len(tet)``, which is what lets the caller map
    faces back to cells without a dict.

    :param tet: tetrahedra connectivity, shape (n_tets, 4).
    :return: sorted node triples, shape (4 * n_tets, 3).
    """
    f = np.concatenate([tet[:, [0, 1, 2]], tet[:, [0, 1, 3]],
                        tet[:, [0, 2, 3]], tet[:, [1, 2, 3]]])
    return np.sort(f, axis=1)


def _flat_axes(coords):
    """Axes along which a set of points has (numerically) zero extent."""
    return [ax for ax, i in (('x', 0), ('y', 1), ('z', 2))
            if coords[:, i].max() - coords[:, i].min() < FLAT_TOL]


def check_mesh(path, expected_bnd_tags=None, expected_matrix_tags=None, verbose=True):
    """
    Run the consistency checks on one .msh file.

    :param path: mesh file to check.
    :param expected_bnd_tags: dict name -> physical tag the model expects on the boundary.
    :param expected_matrix_tags: iterable of volume physical tags the model expects.
    :param verbose: print a report.
    :return: number of failed checks.
    """
    import meshio

    mesh = meshio.read(path)
    pts = mesh.points
    tets = [cb.data for cb in mesh.cells if cb.type == 'tetra']
    tris = [cb.data for cb in mesh.cells if cb.type == 'triangle']
    if not tets:
        # The face/stencil checks below enumerate the four faces of a tet. Meshes
        # of other element types (extruded wedge meshes such as no_damage_zone,
        # hexahedral box meshes such as 17_17_15) are reported as skipped rather
        # than failed: nothing was found wrong with them, they are simply not
        # covered yet.
        kinds = ', '.join(f'{cb.type} x{len(cb.data)}' for cb in mesh.cells) or 'none'
        print(f'mesh: {path}')
        print(f'  nodes {len(pts)}   cells: {kinds}')
        print()
        print('==> SKIPPED: these checks only cover tetrahedral meshes')
        return 0
    tet = tets[0]
    tri = tris[0] if tris else np.zeros((0, 3), dtype=int)

    tet_tags = tri_tags = None
    phys = mesh.cell_data.get('gmsh:physical')
    if phys is not None:
        for cb, tg in zip(mesh.cells, phys, strict=True):
            if cb.type == 'tetra':
                tet_tags = np.asarray(tg)
            elif cb.type == 'triangle':
                tri_tags = np.asarray(tg)

    failed = 0
    say = print if verbose else (lambda *a, **k: None)
    n_tet = len(tet)

    say(f'mesh: {path}')
    say(f'  nodes {len(pts)}   tets {n_tet}   triangles {len(tri)}')
    say(f'  bbox  x {pts[:, 0].min():.2f}..{pts[:, 0].max():.2f}'
        f'   y {pts[:, 1].min():.2f}..{pts[:, 1].max():.2f}'
        f'   z {pts[:, 2].min():.2f}..{pts[:, 2].max():.2f}')

    faces = tet_faces(tet)
    uniq, inverse, counts = np.unique(faces, axis=0, return_inverse=True,
                                      return_counts=True)
    bnd = uniq[counts == 1]
    tri_set = set(map(tuple, np.sort(tri, axis=1).tolist()))
    bnd_set = set(map(tuple, bnd.tolist()))

    # ---- 1. boundary coverage ------------------------------------------------
    missing = sorted(bnd_set - tri_set)
    internal = tri_set - bnd_set
    say(f'\n[1] boundary coverage: {len(bnd)} boundary faces, '
        f'{len(tri)} triangles ({len(internal)} of them internal)')
    if missing:
        failed += 1
        c = pts[np.array(missing)].mean(axis=1)
        say(f'  FAIL: {len(missing)} boundary faces carry no triangle element')
        for ax, i in (('x', 0), ('y', 1), ('z', 2)):
            lo, hi = c[:, i].min(), c[:, i].max()
            note = '   <-- an entire plane is untagged' if hi - lo < FLAT_TOL else ''
            say(f'        {ax}: {lo:10.2f} .. {hi:10.2f}{note}')
    else:
        say('  OK: every boundary face carries a triangle element')

    # ---- 2. manifoldness -----------------------------------------------------
    n_bad = int((counts > 2).sum())
    say(f'\n[2] manifoldness: faces shared by more than two tets: {n_bad}')
    if n_bad:
        failed += 1
        say('  FAIL: mesh is non-manifold')
    else:
        say('  OK')

    # ---- 3. degenerate cells -------------------------------------------------
    p = pts[tet]
    vol = np.abs(np.einsum('ij,ij->i', p[:, 1] - p[:, 0],
                           np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))) / 6.0
    edges = np.stack([np.linalg.norm(p[:, i] - p[:, j], axis=1)
                      for i, j in itertools.combinations(range(4), 2)], axis=1)
    quality = vol / (edges.max(axis=1) ** 3 + 1e-300)   # 0.118 for a regular tet
    say(f'\n[3] cell quality: volume {vol.min():.4g} .. {vol.max():.4g}; '
        f'min v/hmax^3 {quality.min():.3e} (regular tet 0.118)')
    n_deg = int((quality < 1e-6).sum())
    if n_deg:
        failed += 1
        say(f'  FAIL: {n_deg} degenerate / sliver tets')
    else:
        say('  OK')

    # ---- 4. gradient stencil rank --------------------------------------------
    # What the discretizer actually needs. A boundary face without a triangle
    # element contributes no connection, so drop those before testing the rank.
    # An interior connection carries the vector between the two CELL centroids
    # (not the face centroid): that is what the discretizer puts in a row of A,
    # and it is why a cell whose neighbours all sit at the same height ends up
    # with an identically zero column even though its faces are not coplanar.
    centroid = p.mean(axis=1)
    face_centroid = pts[faces].mean(axis=1)
    cell_of_row = np.tile(np.arange(n_tet), 4)
    direction = np.zeros((len(faces), 3))

    order = np.argsort(inverse, kind='stable')
    group_start = np.concatenate([[0], np.cumsum(counts)])[:-1]

    pair = counts == 2
    r0 = order[group_start[pair]]
    r1 = order[group_start[pair] + 1]
    c0, c1 = cell_of_row[r0], cell_of_row[r1]
    direction[r0] = centroid[c1] - centroid[c0]
    direction[r1] = centroid[c0] - centroid[c1]

    contributes = counts[inverse] == 2
    lone = order[group_start[counts == 1]]
    tagged = np.array([tuple(f) in tri_set for f in faces[lone]], dtype=bool)
    kept = lone[tagged]
    direction[kept] = face_centroid[kept] - centroid[cell_of_row[kept]]
    contributes[kept] = True

    # (n_tets, 4, 3): the four connection directions of each cell
    stacked = direction.reshape(4, n_tet, 3).transpose(1, 0, 2)
    sv = np.linalg.svd(stacked, compute_uv=False)
    scale = np.maximum(sv[:, 0], 1e-300)
    rank_ok = sv[:, 2] > 1e-8 * scale
    n_conn = contributes.reshape(4, n_tet).sum(axis=0)
    bad = np.flatnonzero(~rank_ok)
    say('\n[4] gradient stencil rank (the check that actually bites)')
    if bad.size:
        failed += 1
        say(f'  FAIL: {bad.size} cells whose connections do not span 3D')
        for c in bad[:5]:
            say(f'        cell {c} at ({centroid[c][0]:10.2f},{centroid[c][1]:10.2f},'
                f'{centroid[c][2]:10.2f}) with {n_conn[c]} connections')
        if bad.size > 5:
            say(f'        ... {bad.size - 5} more')
    else:
        say(f'  OK: all {n_tet} cells span 3 independent directions '
            f'(min connections per cell: {n_conn.min()})')

    # ---- 5. physical tags ----------------------------------------------------
    say('\n[5] physical tags')
    if tet_tags is not None:
        vol_counts = collections.Counter(tet_tags.tolist())
        say(f'  volume tags : {dict(sorted(vol_counts.items()))}')
        if expected_matrix_tags is not None:
            want = set(expected_matrix_tags)
            missing_t = sorted(want - set(vol_counts))
            extra_t = sorted(set(vol_counts) - want)
            if missing_t or extra_t:
                failed += 1
                say(f'  FAIL: case expects {sorted(want)}; '
                    f'missing {missing_t}, unexpected {extra_t}')
            else:
                say('  OK: volume tags match the case')

    if tri_tags is not None:
        say('  surface tags:')
        present = set(tri_tags.tolist())
        for t in sorted(present):
            sel = tri[tri_tags == t]
            c = pts[sel].mean(axis=1)
            on_bnd = sum(1 for f in map(tuple, np.sort(sel, axis=1).tolist())
                         if f in bnd_set)
            if on_bnd == len(sel):
                kind = 'boundary'
            elif on_bnd == 0:
                kind = 'INTERNAL'
            else:
                kind = f'mixed ({on_bnd}/{len(sel)} on the boundary)'
            flat = _flat_axes(c)
            where = ''
            if flat:
                axis = flat[0]
                where = f', flat in {axis} at {c[0, "xyz".index(axis)]:.2f}'
            say(f'    {t:>6}: {len(sel):6d} tris  {kind}{where}')

        if expected_bnd_tags:
            for name, t in sorted(expected_bnd_tags.items()):
                sel = tri[tri_tags == t] if t in present else np.zeros((0, 3), int)
                if len(sel) == 0:
                    failed += 1
                    say(f'  FAIL: {name} (tag {t}) has no triangles')
                    continue
                on_bnd = sum(1 for f in map(tuple, np.sort(sel, axis=1).tolist())
                             if f in bnd_set)
                if on_bnd != len(sel):
                    failed += 1
                    say(f'  FAIL: {name} (tag {t}) carries {len(sel) - on_bnd} '
                        f'triangles that are not on the domain boundary')

    say(f'\n==> {"PASSED" if failed == 0 else f"{failed} CHECK(S) FAILED"}')
    return failed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mesh')
    ap.add_argument('--case', default=None,
                    help='case name; pulls the expected tags from set_case.set_input_data')
    args = ap.parse_args()

    bnd_tags = matrix_tags = None
    if args.case:
        from set_case import set_input_data
        idata = set_input_data(args.case, physics_type='single_phase_thermal',
                               wells_type='doublet')
        bnd_tags = idata.mesh.bnd_tags
        matrix_tags = idata.mesh.matrix_tags

    return 1 if check_mesh(args.mesh, bnd_tags, matrix_tags) else 0


if __name__ == '__main__':
    sys.exit(main())
