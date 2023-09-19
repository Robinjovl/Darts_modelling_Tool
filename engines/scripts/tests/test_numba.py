import numpy as np
from darts.reservoirs.mesh.geometrymodule import Hexahedron, Wedge, Pyramid, Tetrahedron
from darts.reservoirs.mesh.nb_geometrymodule import nbHexahedron, nbWedge, nbPyramid, nbTetrahedron
import meshio
from numba import jit, types
from numba.typed import Dict, List
import time

def run_test():
    start_time = time.time()
    mesh_data = meshio.read('transfinite.msh')
    print("mesh read --- %s seconds ---" % (time.time() - start_time))
    print("mesh of " + str(len(mesh_data.cells['hexahedron'])) + " hexahedrons loaded")

    geometries = Dict.empty(types.unicode_type, types.int32[:,:])
    for geometry, entries in mesh_data.cells.items():
        geometries[geometry] = entries

    start_time = time.time()
    cells = try_to_read(mesh_data.cells, mesh_data.points)
    print("mesh process --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    cell_nb = try_to_read_numba(geometries, mesh_data.points)
    print("mesh process numba --- %s seconds ---" % (time.time() - start_time))

    for i in range(len(cells)):
        assert(np.abs(cells[i].volume - cell_nb[i].volume) < 1.E-5 * cells[i].volume)
    print("volumes are the same")

def try_to_read(geometries, points):
    cells = []
    perm = np.array([10.0,10.0,10.0], dtype=np.float64)
    # Main loop over different existing geometries
    for geometry, data in geometries.items():
        if geometry == 'hexahedron':
            for ith_cell, nodes_to_cell in enumerate(data):
                cells.append(Hexahedron(nodes_to_cell, points[nodes_to_cell, :], geometry, perm))
        # if geometry == 'wedge':
        #     for ith_cell, nodes_to_cell in enumerate(data):
        #         cells.append(Wedge(nodes_to_cell, points[nodes_to_cell, :], geometry, perm))
        # if geometry == 'tetra':
        #     for ith_cell, nodes_to_cell in enumerate(data):
        #         cells.append(Tetrahedron(nodes_to_cell, points[nodes_to_cell, :], geometry, perm))
        # if geometry == 'pyramid':
        #     for ith_cell, nodes_to_cell in enumerate(data):
        #         cells.append(Pyramid(nodes_to_cell, points[nodes_to_cell, :], geometry, perm))
    return cells

@jit(nopython=True)
def try_to_read_numba(geometries, points):
    cells_nb = List()#.empty_list(nbHexahedron)
    perm = np.array([10.0,10.0,10.0], dtype=np.float64)
    # Main loop over different existing geometries
    for geometry, data in geometries.items():
        if geometry == 'hexahedron':
            for ith_cell, nodes_to_cell in enumerate(data):
                cells_nb.append(nbHexahedron(nodes_to_cell, points[nodes_to_cell, :], geometry, perm, -1))
        # if geometry == 'wedge':
        #     for ith_cell, nodes_to_cell in enumerate(data):
        #         cells_nb.append(nbWedge(nodes_to_cell, points[nodes_to_cell, :], perm, -1))
        # if geometry == 'tetra':
        #     for ith_cell, nodes_to_cell in enumerate(data):
        #         cells_nb.append(nbTetrahedron(nodes_to_cell, points[nodes_to_cell, :], perm, -1))
        # if geometry == 'pyramid':
        #     for ith_cell, nodes_to_cell in enumerate(data):
        #         cells_nb.append(nbPyramid(nodes_to_cell, points[nodes_to_cell, :], perm, -1))
    return cells_nb

run_test()