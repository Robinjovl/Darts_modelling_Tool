import numpy as np
import sys

def fmt(x):
    return '{:.3}'.format(x)

def fmt2(x):
    return '{:.2}'.format(x)

def print_range(model, time):
    P = model.get_pressure()
    T = model.get_temperature()

    print('Time', fmt(time/365), ' years; ',
          'P_range:', fmt(P.min()), '-', fmt(P.max()), 'bars; ',
          'T_range: ' + fmt(T.min()) + ' - ' + fmt(T.max()) + ' degrees' if T is not None else '')

def print_range_array(array, name, units=''):
    if not array[name] is None:
        print(name, ':', fmt(array[name].min()), '-', fmt(array[name].max()), units)


def rotate_grid(arrays, fname_out='grid.grdecl', rot_angle=25):
    '''
    rotate the grid counter clock wise in XY plane (COORD array)
    rot_angle - angle, in degrees
    arrays - dictionary with grid data to be rotated (which read_arrays returns)
    write result to fname_out
    '''
    coord_1d = arrays['COORD']
    coord = coord_1d.reshape(coord_1d.size//6, 6)
    x1 = coord[:, 0]
    x2 = coord[:, 3]
    y1 = coord[:, 1]
    y2 = coord[:, 4]
    xy1 = np.vstack([x1, y1])
    xy2 = np.vstack([x2, y2])

    theta = np.radians(rot_angle)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])

    xy1_new = rot_matrix @ xy1
    xy2_new = rot_matrix @ xy2

    coord_new = coord.copy()
    coord_new[:, 0] = xy1_new[0, :]
    coord_new[:, 1] = xy1_new[1, :]
    coord_new[:, 3] = xy2_new[0, :]
    coord_new[:, 4] = xy2_new[1, :]
    coord_new = coord_new.flatten()

    keys = ['SPECGRID', 'COORD', 'ZCORN', 'ACTNUM']
    data = [arrays['SPECGRID'], coord_new, arrays['ZCORN'], arrays['ACTNUM']]
    from darts.tools.keyword_file_tools import save_few_keywords
    save_few_keywords(fname_out, keys, data)

# duplicate the screen output to a log-file
class logger(object):
    def __init__(self, log_fname):
        self.terminal = sys.stdout
        self.log = open(log_fname, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()