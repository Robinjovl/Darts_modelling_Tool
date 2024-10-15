import numpy as np
from darts.tools.keyword_file_tools import save_few_keywords

def gen_cpg_grid(nx : int, ny : int, nz : int, 
                 dx, dy, dz,
                 permx : float, permy : float, permz : float, poro : float,
                 gridname=None, propname=None, burden_dz=None,
                 start_x : float=0, start_y : float =0, start_z : float =2000):
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
    :param gridname: filename to write text output in GRDECL format containing coordinate information
    :param propname: filename to write text output in GRDECL format containing static reservoir properties
    :param burden_dz: array of thickness for over and under burden layers. If None - do not add burden layers
    :param start_x: mesh lower bound coordinate by X [m]
    :param start_y: mesh lower bound coordinate by Y [m]
    :param start_z: mesh lower bound coordinate by Z (the depth of the top layer) [m]  
    '''
    if np.isscalar(dx):
        dx_array = np.zeros(nx) + dx
    else:
        dx_array = dx

    if np.isscalar(dy):
        dy_array = np.zeros(ny) + dy
    else:
        dy_array = dy

    if np.isscalar(dz):
        dz_array = np.zeros(nz) + dz
    else:
        dz_array = dz
    # generate corner point geometry
    coord = np.empty(6 * (nx + 1) * (ny + 1), dtype=float)
    zcorn = np.empty(8 * nx * ny * nz, dtype=float)
    end_z = start_z + dz_array.sum()
    idx = 0
    y = start_y
    for j in range(ny + 1):
        x = start_x
        for i in range(nx + 1):
            coord[idx + 0] = x
            coord[idx + 1] = y
            coord[idx + 2] = start_z
            coord[idx + 3] = x
            coord[idx + 4] = y
            coord[idx + 5] = end_z
            idx += 6
            if i < nx:
                x += dx_array[i]
        if j < ny:
            y += dy_array[j]
    zcorn[:4 * nx * ny] = start_z
    z = start_z
    for k in range(1, nz + 1):
        z += dz_array[k - 1]
        zcorn[(2 * k - 1) * 4 * nx * ny:(2 * k + 1) * 4 * nx * ny] = z

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

    if burden_dz is not None:
        nz2 = nz + 2 * len(burden_dz)
    else:
        nz2 = nz

    specgrid = np.array([nx, ny, nz2], dtype=np.int32)
    n = nx * ny * nz2
    actnum = np.ones(n, dtype=np.int32)
    poro_arr = np.zeros(n) + poro
    permx_arr = np.zeros(n) + permx
    permy_arr = np.zeros(n) + permy
    permz_arr = np.zeros(n) + permz

    # write to file
    if gridname is not None:
        specgrid_ = [nx, ny, nz2, '1', 'F']
        keys = ['SPECGRID', 'COORD', 'ZCORN', 'ACTNUM']
        data = [specgrid_, coord, zcorn, actnum]
        save_few_keywords(gridname, keys, data)

        data = []
        keys = []
        for arr, arr_name in [(poro_arr, 'PORO'), (permx_arr, 'PERMX'), (permy_arr, 'PERMY'), (permz_arr, 'PERMZ')]:
            data.append(arr)
            keys.append(arr_name)
        save_few_keywords(propname, keys, data)

    arrays = {}
    arrays['SPECGRID'] = specgrid
    arrays['COORD'] = coord
    arrays['ZCORN'] = zcorn
    arrays['ACTNUM'] = actnum
    arrays['PORO'] = poro_arr
    arrays['PERMX'] = permx_arr
    arrays['PERMY'] = permy_arr
    arrays['PERMZ'] = permz_arr
    return arrays


if __name__ =='__main__':
    nx = 200
    ny = 200
    nz = 100
    gen_cpg_grid(nx=nx, ny=ny, nz=nz, dx=50, dy=50, dz=10,
                 permx=10, permy=10, permz=10, poro=0.2,
                 gridname='grid.grdecl', propname='reservoir.in')