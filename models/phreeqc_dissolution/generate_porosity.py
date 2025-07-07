import gstools as gs
import numpy as np
import meshio

# structured field with a size 100x100 and a grid-size of 1x1

def generate_random_field(nx=100, len_scale=8, var=1):
    x = y = range(nx)
    model = gs.Spherical(dim=2, var=1, len_scale=len_scale)
    srf = gs.SRF(model)
    field = srf((x, y), mesh_type='structured')

    np.savetxt(f'spherical_{nx}_{len_scale}.txt', field)
    srf.plot()

def generate_random_field_unstructured_mesh(len_scale, mesh_file):
    # read MSH file
    mesh = meshio.read(mesh_file)
    pts = mesh.points[mesh.cells[1].data]
    c = np.average(pts, axis=1)
    edge_vectors = np.stack((pts[:, 1, :] - pts[:, 0, :], pts[:, 2, :] - pts[:, 0, :],
                                    pts[:, 3, :] - pts[:, 0, :]), axis=-1)
    volumes = np.abs(np.linalg.det(edge_vectors)) / 6.0

    model = gs.Spherical(dim=3, var=1, len_scale=len_scale)
    srf = gs.SRF(model)
    field = srf((c[:, 0], c[:, 1], c[:, 2]), mesh_type='unstructured', point_volumes=volumes)

    np.savetxt(f'core_{len_scale}.txt', field)
    # srf.plot()

# generate_random_field(nx=50, len_scale=5, var=1)
mesh_file = './input/core_60k.msh'
generate_random_field_unstructured_mesh(len_scale=0.01, mesh_file=mesh_file)

