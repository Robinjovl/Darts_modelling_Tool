from darts.tools.keyword_file_tools import load_single_keyword
import math
import numpy as np
import pandas as pd
import gmsh
import meshio
import random




class mesh_creator():
    def __init__(self, rand_seed, nx, ny, nz, lc_bound, thickness, struct_mesh_path, ACTNUM_path, depth_path,
                 file_name, well_coord_path):
        # nx = 139
        # ny = 48
        # nz = 9
        #
        # lc = 2100  # characteristic length
        # thickness = 72  # meter
        # struct_mesh_path = 'C:\DARTS\darts-opt\models\Brugge\dxdydz.in'
        # ACTNUM_path = 'C:\DARTS\darts-opt\models\Brugge\ACTNUM.in'
        # depth_path = 'C:\DARTS\darts-opt\models\Brugge\depth.in'
        # file_name = "Brugge.msh"

        random.seed(rand_seed)

        dx = load_single_keyword(struct_mesh_path, 'DX')
        dy = load_single_keyword(struct_mesh_path, 'DY')
        dz = load_single_keyword(struct_mesh_path, 'DZ')

        actnum = load_single_keyword(ACTNUM_path, 'ACTNUM')
        depth = load_single_keyword(depth_path, 'DEPTH')

        actnum[dx == 0] = 0
        actnum[dy == 0] = 0
        actnum[dz == 0] = 0

        mean_dx = np.mean(actnum[0:(nx*ny)]*dx[0:nx*ny])
        mean_dy = np.mean(actnum[0:nx*ny]*dy[0:nx*ny])
        mean_dz = np.mean(actnum[0:nx*ny]*dz[0:nx*ny])
        mean_depth = np.mean(actnum[0:(nx*ny)]*depth[0:nx*ny])

        lc1 = 2200  # parameter defining the mesh size at the boundary
        lc2 = 330  # parameter defining the mesh size at the wells


        gmsh.initialize()

        gmsh.option.setNumber("General.Terminal", 1)

        gmsh.model.add("My_Brugge_mesh")
        point_tag = 1
        index = 1

        # creating the top coordinaates of the reserovir from Actnum and Dx
        top_act = actnum[0:nx * ny]
        top_actnum = top_act.reshape(ny, nx)

        ## creating boundaries:
        # each row contains two points defining the edge of the reservoir
        for j in range(0, ny, 9):
            first_nod = np.array([0, 0])
            last_nod = np.array([0, 0])
            exam = np.zeros(nx)
            for i in range(nx):
                if top_actnum[j, i] == 1:
                    exam[i] = 1

            activee = np.where(exam == 1)

            first_nod = np.array([j, activee[0][0]])

            last_nod = np.array([j, activee[0][len(activee[0]) - 1]])

            print('first and last Node:', first_nod, last_nod)

            if first_nod[0] < 30:
                lc = lc1
            else:
                lc = lc2



            # we need to have a conversion factor to adjust the coordinate with the reference
            # coordinates from the data (here I assumed Denis's coordinates for the corners of the reservoir is accurate)
            if all(last_nod == first_nod):
                x1 = (first_nod[1] * mean_dx + 110) / 0.78
                y1 = first_nod[0] * mean_dy + 110
                z1 = 0
                lc = lc + random.uniform(-10, 10)
                gmsh.model.geo.addPoint(x1, y1, z1, lc, point_tag)
                print('Point tag generated: ', point_tag)
                point_tag += 1

            else:
                x1 = (first_nod[1] * mean_dx + 110) / 0.78
                y1 = (first_nod[0] * mean_dy + 110) / 0.73
                z1 = 0

                x2 = (last_nod[1] * mean_dx + 110) / 0.78
                y2 = (last_nod[0] * mean_dy + 110) / 0.73
                z2 = 0
                lc = lc + random.uniform(-10, 10)
                gmsh.model.geo.addPoint(x1, y1, z1, lc, point_tag)
                print('Point tag generated:', point_tag)
                point_tag += 1
                gmsh.model.geo.addPoint(x2, y2, z2, lc, point_tag)
                print('Point tag generated:', point_tag)
                point_tag += 1

        n_point = point_tag - 1

        # set point tags
        points = np.arange(n_point) + 1

        # finding the odd and the even point tags
        odd_points = points[np.where(points % 2 == 0)]
        even_points = points[np.where(points % 2 != 0)]

        # making the lines
        number_of_lines = 0
        # first line from point 1 to point 2
        gmsh.model.geo.addLine(1, 2, 1)
        print('line 1 to 2 line tag 1:')
        number_of_lines += 1

        # other points from 2 to odd points
        for i in range(0, len(odd_points) - 1, 1):
            gmsh.model.geo.addLine(odd_points[i], odd_points[i + 1], i + 2)
            print('line ' + str(odd_points[i]) + 'to' + str(odd_points[i + 1]) + ',line tag:' + str(i + 2))
            number_of_lines += 1

        # from the last odd point to last even point
        gmsh.model.geo.addLine(odd_points[len(odd_points) - 1], even_points[len(even_points) - 1], len(odd_points) + 1)
        number_of_lines += 1
        print(
            'line ' + str(odd_points[len(odd_points) - 1]) + 'to' + str(even_points[len(even_points) - 1]) + ',line tag:' + str(
                len(odd_points) + 1))

        # other even points
        for i in range(len(even_points) - 1, 0, -1):
            gmsh.model.geo.addLine(even_points[i], even_points[i - 1], n_point - i + 1)
            print('line ' + str(even_points[i]) + 'to' + str(even_points[i - 1]) + ',line tag:' + str(n_point - i + 1))
            number_of_lines += 1

        print('--------------------------------------')
        print('number of lines:', number_of_lines)
        print('--------------------------------------')

        # building the curve loop
        n_lines = np.arange(number_of_lines) + 1

        gmsh.model.geo.addCurveLoop(n_lines, 211)

        # building the surface for the top curve loop
        ps2 = gmsh.model.geo.addPlaneSurface([211])

        ########## making 3D volume ##########
        # here (2, ps2) means the plane ps2 is a 2D entity
        # numElements=[1] is similar to Layers{1} in .geo file
        gmsh.model.geo.extrude([(2, ps2)], 0, 0, thickness, numElements=[1], recombine=True)


        ########### Well coordinates ############
        well_coord = np.genfromtxt(well_coord_path)
        for i, wc in enumerate(well_coord):
            gmsh.model.geo.addPoint(wc[0], wc[1], wc[2])


        ########### adding physical group ##############
        # setting physical group to the volume
        gmsh.model.addPhysicalGroup(3, [1], 999)
        gmsh.model.setPhysicalName(3, 1, "The volume")


        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(file_name)

        gmsh.finalize()

