import numpy as np
import sys
import os

def darts_version_ge(target=(1, 5, 1)):
    '''
    Return True if the installed open-darts version is >= target (tuple of ints).
    Used to switch between API variants (e.g. the OBL grid setup and the
    PropertyContainer keyword changed after 1.5.0). Falls back to False (older
    API) if the version cannot be determined.
    '''
    try:
        from importlib.metadata import version
        parts = version('open-darts').split('.')
        v = tuple(int(''.join(c for c in p if c.isdigit()) or 0) for p in parts[:3])
        v = v + (0,) * (3 - len(v))
        return v >= tuple(target)
    except Exception:
        return False

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
