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
