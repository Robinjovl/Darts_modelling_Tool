"""
Helpers for the model test suite (models/run_test_suite2.py).

A model test typically:

  * writes its reference once, with UPLOAD_PKL=1 using `save_vtk_ref()`,
  * compares the solution of one reported timestep with that reference on every later pipeline run using `compare_vtk_with_ref()`.
"""

import os

import numpy as np

# Default tolerances of the reference comparison.
REL_TOLERANCE = 1e-6
ABS_TOLERANCE = 1e-8


def read_vtk(filename: str, props: list):
    """
    Read cell centers, cell data, points and point data of the listed properties.

    :param filename: vtk file to read
    :param props: property names to read
    :return: (centers, cell_data, points, point_data)
    """
    import meshio

    mesh = meshio.read(filename=filename)

    # cell data
    centers = np.empty([0, 3])
    cell_data = {}
    for geom_name, geom in mesh.cells_dict.items():
        centers = np.append(centers, np.average(mesh.points[geom], axis=1), axis=0)
        for prop in props:
            if prop in mesh.cell_data_dict:
                if prop not in cell_data:
                    cell_data[prop] = []
                cell_data[prop].append(mesh.cell_data_dict[prop][geom_name])

    # point data
    points = mesh.points
    point_data = {}
    for prop_name, prop in mesh.point_data.items():
        if prop_name in props:
            point_data[prop_name] = prop

    return centers, cell_data, points, point_data


def save_vtk_ref(
    vtk_cur_fname: str,
    vtk_ref_fname: str,
    props: list = None,
    compression: str = 'zlib',
):
    """
    Store a computed vtk solution as the reference one.

    Keeping only the compared properties and writing the file compressed to make reference files smaller.

    :param vtk_cur_fname: computed solution to store.
    :param vtk_ref_fname: reference file to write; its directory is created if missing.
    :param props: cell properties to keep, all of them if None.
    :param compression: meshio compression of the reference file, None to switch it off.
    :return: 0
    """
    import meshio

    mesh = meshio.read(vtk_cur_fname)
    if props is None:
        cell_data = mesh.cell_data
    else:
        cell_data = {
            prop: mesh.cell_data[prop] for prop in props if prop in mesh.cell_data
        }
    ref_dir = os.path.dirname(vtk_ref_fname)
    if ref_dir:
        os.makedirs(ref_dir, exist_ok=True)
    meshio.write(
        vtk_ref_fname,
        meshio.Mesh(mesh.points, mesh.cells, cell_data=cell_data),
        binary=True,
        compression=compression,
    )
    print('SAVED REFERENCE VTK FILE', vtk_ref_fname)
    return 0


def compare_vtk_with_ref(
    vtk_cur_fname: str,
    vtk_ref_fname: str,
    props: list,
    rel_tolerance: float = REL_TOLERANCE,
    abs_tolerance: float = ABS_TOLERANCE,
    verbose: bool = True,
):
    """
    Compare a computed vtk solution with the reference one, property by property.

    Cell centers and points are compared as well, so a changed mesh is reported too.

    :param vtk_cur_fname: computed solution.
    :param vtk_ref_fname: reference solution.
    :param props: cell properties to compare; a property missing from the reference file
                  is not compared, one missing only from the solution is an error.
    :param rel_tolerance: relative tolerance of the comparison.
    :param abs_tolerance: absolute tolerance of the comparison.
    :param verbose: print the maximum relative difference of every compared item.
    :return: 0 if the solutions match, 1 otherwise.
    """
    if not os.path.exists(vtk_ref_fname):
        print('REFERENCE VTK FILE', os.path.abspath(vtk_ref_fname), 'does not exist.')
        print('Run with UPLOAD_PKL=1 to create or update this reference file.')
        return 1
    if not os.path.exists(vtk_cur_fname):
        print(
            'SOLUTION VTK FILE',
            os.path.abspath(vtk_cur_fname),
            'was not written by the run.',
        )
        return 1

    ref = read_vtk(vtk_ref_fname, props)  # the reference solution
    cur = read_vtk(vtk_cur_fname, props)  # the current solution
    names = [
        'centers',
        'cell_data',
        'points',
        'point_data',
    ]  # object names to be compared

    eps_div = 1e-15  # to avoid division by zero
    ret_flag = 0
    # the pass criterion is np.isclose: abs.diff <= abs_tolerance + rel_tolerance * |reference|
    print(
        'compare: tolerances: abs',
        abs_tolerance,
        'rel',
        rel_tolerance,
    )
    for n, r, c in zip(names, ref, cur, strict=True):
        if isinstance(r, dict):  # cell_data is a dict, so check each item there
            if len(r) == 0:  # point_data is empty, skip it
                continue
            missing = [prop for prop in r.keys() if prop not in c]
            if missing:
                print('There are no properties', missing, 'in', vtk_cur_fname)
                ret_flag = 1
            ns = [prop for prop in r.keys() if prop in c]
            rs, cs = [r[prop] for prop in ns], [c[prop] for prop in ns]  # dict to list
        else:
            ns, rs, cs = (
                [n],
                [r],
                [c],
            )  # create a list just to have a loop below for both cases
        for ni, ri, ci in zip(ns, rs, cs, strict=True):
            r1 = np.array(ri)
            c1 = np.array(ci)
            if r1.shape != c1.shape:
                print(
                    'There is a shape difference', r1.shape, 'vs', c1.shape, 'for', ni
                )
                ret_flag = 1
                continue
            abs_diff = np.fabs(r1 - c1)  # absolute difference
            rel_diff = abs_diff / (np.fabs(r1) + eps_div)  # relative difference
            ok = np.isclose(r1, c1, rtol=rel_tolerance, atol=abs_tolerance)
            # how many values exceed each tolerance on its own
            n_abs = int((abs_diff > abs_tolerance).sum())
            n_rel = int((rel_diff > rel_tolerance).sum())
            counts = (
                f'{n_abs} of {abs_diff.size} values over abs.tol, {n_rel} over rel.tol'
            )
            if ok.all():
                if verbose:
                    abs_max = abs_diff.max() if abs_diff.size else 0.0
                    rel_max = rel_diff.max() if rel_diff.size else 0.0
                    print(
                        'Comparing',
                        ni,
                        'abs.diff',
                        abs_max,
                        'rel.diff',
                        rel_max,
                        '(' + counts + ')',
                    )
            else:
                ret_flag = 1
                # report the difference over the failed values only,
                # the maximum over all values can be dominated by near-zero references
                bad = ~ok
                print(
                    'There is a difference for',
                    ni,
                    'abs.diff',
                    abs_diff[bad].max(),
                    'rel.diff',
                    rel_diff[bad].max(),
                    'in',
                    int(bad.sum()),
                    'of',
                    bad.size,
                    'values;',
                    counts,
                )
    print('compare:', 'OK' if ret_flag == 0 else 'FAILED')
    return ret_flag
