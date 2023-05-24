import numpy as np


def save_array(arr: np.array, fname: str, keyword: str, actnum: np.array, mode='w'):
    '''
    writes numpy array of n_active_cell size to text file in GRDECL format with n_cells_total
    :param arr: numpy array to write
    :param fname: filename
    :param keyword: keyword for array
    :param actnum: actnum array
    :param mode: 'w' to rewrite the file or 'a' to append
    :return: None
    '''
    arr_full = make_full_cube(arr, actnum)
    with open(fname, mode) as f:
        f.write(keyword + '\n')
        s = ''
        for i in range(arr_full.size):
            s += str(arr_full[i]) + ' '
            if (i+1) % 6 == 0:  # write only 6 values per row
                f.write(s + '\n')
                s = ''
        f.write(s + '\n')
        f.write('/\n')
        print('Array saved to file', fname, ' (keyword ' + keyword + ')')


def make_full_cube(cube: np.array, actnum: np.array):
    '''
    returns 1d-array of size nx*ny*nz, filled with zeros where actnum is zero
    :param cube: 1d-array of size n_active_cells
    :param actnum: 1d-array of size nx*ny*nz
    :return:
    '''
    if actnum.size == cube.size:
        return cube
    cube_full = np.zeros(actnum.size)
    #j = 0
    #for i in range(actnum.size):
    #    if actnum[i] > 0:
    #        cube_full[i] = cube[j]
    #        j += 1
    cube_full[actnum > 0] = cube
    return cube_full
    
   