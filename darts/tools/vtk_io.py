import numpy as np


def write_lines_vtp(path: str, nodes_xyz: np.ndarray, output_properties: dict = None):
    """
    Write a polyline as a VTK XML PolyData file (.vtp) - convenient for well visualization.

    Nodes define N segments via consecutive pairs (node i → node i+1).
    ParaView's Tube filter requires PolyData input - this function produces it.

    :param path: Output file path (should end in .vtp).
    :param nodes_xyz: Node coordinates, shape (n_nodes, 3).
    :param output_properties: Optional dict of per-segment arrays, each length n_nodes-1.
    """
    coords = np.asarray(nodes_xyz, dtype=float)
    npts = coords.shape[0]
    nseg = npts - 1

    if output_properties is not None:
        for name, vals in output_properties.items():
            arr = np.asarray(vals).ravel()
            if arr.shape[0] != nseg:
                raise ValueError(
                    f"output_properties['{name}'] length {arr.shape[0]} != n_segments {nseg}"
                )

    connectivity = " ".join(f"{i} {i + 1}" for i in range(nseg))
    offsets = " ".join(str(2 * (i + 1)) for i in range(nseg))

    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">')
    xml.append('  <PolyData>')
    xml.append(
        f'    <Piece NumberOfPoints="{npts}" NumberOfVerts="0" '
        f'NumberOfLines="{nseg}" NumberOfStrips="0" NumberOfPolys="0">'
    )
    xml.append('      <Points>')
    xml.append(
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">'
    )
    for pt in coords:
        xml.append(f'          {pt[0]:.10g} {pt[1]:.10g} {pt[2]:.10g}')
    xml.append('        </DataArray>')
    xml.append('      </Points>')
    xml.append('      <Lines>')
    xml.append('        <DataArray type="Int32" Name="connectivity" format="ascii">')
    xml.append(f'          {connectivity}')
    xml.append('        </DataArray>')
    xml.append('        <DataArray type="Int32" Name="offsets" format="ascii">')
    xml.append(f'          {offsets}')
    xml.append('        </DataArray>')
    xml.append('      </Lines>')
    if output_properties:
        xml.append('      <CellData>')
        for name, vals in output_properties.items():
            arr = np.asarray(vals).ravel().astype(float)
            xml.append(
                f'        <DataArray type="Float64" Name="{name}" format="ascii">'
            )
            xml.append('          ' + ' '.join(f'{v:.10g}' for v in arr))
            xml.append('        </DataArray>')
        xml.append('      </CellData>')
    xml.append('    </Piece>')
    xml.append('  </PolyData>')
    xml.append('</VTKFile>')

    with open(path, "w") as fh:
        fh.write("\n".join(xml))


def _normalize_pvd_entry(entry):
    # Reservoir solution PVDs use the compact (time, file) form because they
    # contain one dataset per timestep. Well PVDs can contain several wells at
    # the same timestep, so they pass (time, file, group, part). The group is a
    # readable label such as "well_P1"; part is the numeric dataset id ParaView
    # uses to keep those same-timestep datasets separate.
    if len(entry) == 2:
        timestep, filename = entry
        group = ""
        part = 0
    elif len(entry) == 4:
        timestep, filename, group, part = entry
    else:
        raise ValueError(
            "PVD entries must be (timestep, filename) or "
            "(timestep, filename, group, part)."
        )

    return timestep, filename, group, part


def write_pvd(path: str, entries: list[tuple]):
    """
    Write a ParaView Data (.pvd) collection file referencing a time series of VTK files.

    :param path: Output file path (should end in .pvd).
    :param entries: List of (timestep, filename) or
                    (timestep, filename, group, part) entries. Filenames are written
                    without path modification and should be relative to the directory
                    containing the .pvd file.
    """
    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">')
    xml.append('  <Collection>')
    for entry in entries:
        timestep, filename, group, part = _normalize_pvd_entry(entry)
        xml.append(
            f'    <DataSet timestep="{timestep:.10g}" '
            f'group="{group}" part="{part}" file="{filename}"/>'
        )
    xml.append('  </Collection>')
    xml.append('</VTKFile>')

    with open(path, "w") as fh:
        fh.write("\n".join(xml))


def _block_shapes_match(cur_blocks, ref_blocks):
    if len(cur_blocks) != len(ref_blocks):
        return False
    return all(c.shape == r.shape for c, r in zip(cur_blocks, ref_blocks, strict=True))


def write_vtk_difference(
    vtk_cur_fname: str,
    vtk_ref_fname: str,
    vtk_diff_fname: str,
    props: list = None,
    include_values: bool = True,
    relative: bool = True,
    eps_div: float = 1e-15,
    compression: str = 'zlib',
    verbose: bool = True,
):
    """
    Write the difference between two vtk solutions into a separate vtk file.

    The output carries the mesh of the current solution and, for every compared
    property, a ``<prop>_diff`` array (current - reference). This makes a visual
    comparison of two runs a matter of opening one file in ParaView instead of
    switching back and forth between two.

    :param vtk_cur_fname: current solution.
    :param vtk_ref_fname: reference solution to subtract.
    :param vtk_diff_fname: file to write; its directory is created if missing.
    :param props: property names to compare, all properties present in both files if None.
    :param include_values: also store the compared values themselves, as ``<prop>``
                           (current) and ``<prop>_ref`` (reference).
    :param relative: also store ``<prop>_reldiff`` = diff / (|reference| + eps_div).
    :param eps_div: regularization of the relative difference denominator.
    :param compression: meshio compression of the output file, None to switch it off.
    :param verbose: print the maximum absolute and relative difference of every property.
    :return: dict prop -> (max absolute difference, max relative difference).
    """
    import os

    import meshio

    cur = meshio.read(vtk_cur_fname)
    ref = meshio.read(vtk_ref_fname)

    if cur.points.shape != ref.points.shape:
        raise ValueError(
            f'Meshes differ: {cur.points.shape[0]} points in {vtk_cur_fname} vs '
            f'{ref.points.shape[0]} in {vtk_ref_fname}; the two solutions cannot be subtracted.'
        )
    max_point_shift = (
        float(np.abs(cur.points - ref.points).max()) if cur.points.size else 0.0
    )
    if verbose and max_point_shift > 0.0:
        print(
            f'write_vtk_difference: warning: node coordinates differ, max shift {max_point_shift:.3e}'
        )

    if props is None:
        names = [name for name in cur.cell_data if name in ref.cell_data]
    else:
        names = list(props)

    stats = {}
    cell_data = {}
    for name in names:
        if name not in cur.cell_data:
            if verbose:
                print(
                    f'write_vtk_difference: no property "{name}" in {vtk_cur_fname}, skipped'
                )
            continue
        if name not in ref.cell_data:
            if verbose:
                print(
                    f'write_vtk_difference: no property "{name}" in {vtk_ref_fname}, skipped'
                )
            continue
        cur_blocks = [np.asarray(b, dtype=float) for b in cur.cell_data[name]]
        ref_blocks = [np.asarray(b, dtype=float) for b in ref.cell_data[name]]
        if not _block_shapes_match(cur_blocks, ref_blocks):
            if verbose:
                print(
                    f'write_vtk_difference: property "{name}" has a different shape in the two files, skipped'
                )
            continue

        diff_blocks = [c - r for c, r in zip(cur_blocks, ref_blocks, strict=True)]
        rel_blocks = [
            d / (np.abs(r) + eps_div)
            for d, r in zip(diff_blocks, ref_blocks, strict=True)
        ]

        if include_values:
            cell_data[name] = cur_blocks
            cell_data[name + '_ref'] = ref_blocks
        cell_data[name + '_diff'] = diff_blocks
        if relative:
            cell_data[name + '_reldiff'] = rel_blocks

        max_abs = max(
            (float(np.abs(d).max()) for d in diff_blocks if d.size), default=0.0
        )
        max_rel = max(
            (float(np.abs(d).max()) for d in rel_blocks if d.size), default=0.0
        )
        stats[name] = (max_abs, max_rel)
        if verbose:
            print(f'{name}: max abs diff {max_abs:.6e}, max rel diff {max_rel:.6e}')

    if not cell_data:
        raise ValueError(
            f'No comparable cell property found in {vtk_cur_fname} and {vtk_ref_fname}.'
        )

    diff_dir = os.path.dirname(vtk_diff_fname)
    if diff_dir:
        os.makedirs(diff_dir, exist_ok=True)
    meshio.write(
        vtk_diff_fname,
        meshio.Mesh(cur.points, cur.cells, cell_data=cell_data),
        binary=True,
        compression=compression,
    )
    if verbose:
        print('SAVED DIFFERENCE VTK FILE', vtk_diff_fname)
    return stats
