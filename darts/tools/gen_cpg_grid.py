import numpy as np

def gen_cpg_grid(nx : int, ny : int, nz : int, 
                 dx : float, dy : float, dz : float,
                 permx : float, permy : float, permz : float, poro : float,
                 filename=None, burden_dz=None,
                 start_x : float=0, start_y : float =0, start_z : float =2300):
    '''
    Generate a regular grid in corner point geometry format (COORD, ZCORN) and optionally output it to the file
    :param nx: number of reservoir blocks in the x-direction
    :param ny: number of reservoir blocks in the y-direction
    :param nz: number of reservoir blocks in the z-direction
    :param dx: size of the reservoir blocks in the x-direction (scalar) [m]
    :param dy: size of the reservoir blocks in the y-direction (scalar) [m]
    :param dz: size of the reservoir blocks in the z-direction (scalar) [m]
    :param permx: permeability of the reservoir blocks in the x-direction (scalar) [mD]
    :param permy: permeability of the reservoir blocks in the y-direction (scalar) [mD]
    :param permz: permeability of the reservoir blocks in the z-direction (scalar) [mD]
    :param poro: porosity of the reservoir blocks (scalar)
    :param filename: filename to write text output in GRDECL format 
    :param burden_dz: array of thickness for over and under burden layers. If None - do not add burden layers
    :param start_x: mesh lower bound coordinate by X [m]
    :param start_y: mesh lower bound coordinate by Y [m]
    :param start_z: mesh lower bound coordinate by Z (the depth of the top layer) [m]  
    '''
    # generate corner point geometry
    coord = np.empty(6 * (nx + 1) * (ny + 1), dtype=float)
    zcorn = np.empty(8 * nx * ny * nz, dtype=float)
    idx = 0
    for j in range(ny + 1):
        y = start_y + j * dy
        for i in range(nx + 1):
            x = start_x + i * dx
            coord[idx + 0] = x
            coord[idx + 1] = y
            coord[idx + 2] = start_z
            coord[idx + 3] = x
            coord[idx + 4] = y
            coord[idx + 5] = start_z + nz * dz
            idx += 6

    zcorn[:4 * nx * ny] = start_z
    for i in range(1, nz + 1):
        zcorn[(2 * i - 1) * 4 * nx * ny:(2 * i + 1) * 4 * nx * ny] = start_z + i * dz

    if burden_dz is not None:
        for i in range(0, len(burden_dz)):
            # for each burden layer, zcorn has 4 * nx * ny number of values
            zcorn = np.concatenate(
                [zcorn[:4 * nx * ny] - burden_dz[i],
                 zcorn[:4 * nx * ny],
                 zcorn,
                 zcorn[-4 * nx * ny:],
                 zcorn[-4 * nx * ny:] + burden_dz[i],
                 ]
            )

    # write to file
    if filename is not None:
        with open(filename, 'w') as grdecl:
            grdecl.write("SPECGRID\n")
            if burden_dz is not None:
                grdecl.write(f"{nx} {ny} {nz+2*len(burden_dz)} 1 F\n")
                grdecl.write('/\n\nACTNUM\n')
                np.savetxt(grdecl, np.ones(nx * ny * (nz+2*len(burden_dz))).reshape(-1, 12), fmt='%d', delimiter=' ', newline='\n')
            else:
                grdecl.write(f"{nx} {ny} {nz} 1 F\n")
                grdecl.write('/\n\nACTNUM\n')
                np.savetxt(grdecl, np.ones(nx * ny * nz), fmt='%d',
                           delimiter=' ', newline='\n')

            grdecl.write("/\n\nCOORD\n")
            np.savetxt(grdecl, coord.reshape(-1, 6), delimiter=' ', newline='\n')

            grdecl.write('/\n\nZCORN\n')
            np.savetxt(grdecl, zcorn.reshape(-1, 8), delimiter=' ', newline='\n')

            # write uniform properties
            n = nx * ny * nz
            arr = np.zeros(n)
            for value, arr_name in [(permx, 'PERMX'), (permy, 'PERMY'), (permz, 'PERMZ'), (poro, 'PORO')]:
                arr.fill(value)
                grdecl.write('/\n\n' + arr_name + '\n')
                np.savetxt(grdecl, arr, delimiter=' ', newline='\n')

            grdecl.write('/')

    return coord, zcorn


if __name__ =='__main__':
    nx = 200
    ny = 200
    nz = 100
    gen_cpg_grid(nx=nx, ny=ny, nz=nz, dx=50, dy=50, dz=10,
                 permx=10, permy=10, permz=10, poro=0.2,
                 filename=str(nx)+'x'+str(ny)+'x'+str(nz)+'.grdecl')