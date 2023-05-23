import numpy as np
import meshio



def calc_equivalent_WI(mesh_file, well_coord, centroid_info, kx_list, ky_list, dz, skin=0):
    distance = []
    for j, centroid in enumerate(centroid_info):
        distance.append(np.linalg.norm(well_coord - centroid))
    min_dis = np.min(distance)
    idx_well = distance.index(min_dis)


    # perm = np.ones(np.size(kx_list))
    # perm[idx_well] = 100
    # mesh = meshio.read(mesh_file)
    # points = mesh.points
    # cells = mesh.cells
    # cell_data = permeability = {'wedge': {"perm": np.array(perm)}}
    #
    # meshio.write_points_cells(
    #     "_Brugge.vtk",
    #     points,
    #     cells,
    #     # Optionally provide extra data on points, cells, etc.
    #     # point_data=point_data,
    #     cell_data=cell_data,
    #     # field_data=field_data
    #     )


    kx = kx_list[idx_well]
    ky = ky_list[idx_well]
    mesh = meshio.read(mesh_file)
    points = mesh.points
    cells = mesh.cells
    node_cell = cells['wedge'][idx_well]
    coord_top_triangle = points[node_cell[0:3]]
    # in counter-clockwise direction
    x1 = coord_top_triangle[0][0]
    y1 = coord_top_triangle[0][1]
    x2 = coord_top_triangle[2][0]
    y2 = coord_top_triangle[2][1]
    x3 = coord_top_triangle[1][0]
    y3 = coord_top_triangle[1][1]

    area_top_triangle = abs((x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2) / 2)
    dx = dy = np.sqrt(area_top_triangle)
    well_radius = 0.1524

    peaceman_rad = 0.28 * np.sqrt(np.sqrt(ky / kx) * dx ** 2 + np.sqrt(kx / ky) * dy ** 2) / \
                   ((ky / kx) ** (1 / 4) + (kx / ky) ** (1 / 4))
    well_index = 2 * np.pi * dz * np.sqrt(kx * ky) / (np.log(peaceman_rad / well_radius) + skin)
    return well_index * 0.0085267146719160104986876640419948  # multiplied by Darcy constant