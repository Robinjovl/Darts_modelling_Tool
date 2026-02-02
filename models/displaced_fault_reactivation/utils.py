import numpy as np
from typing import Dict, Any
import hashlib
import json
from hashlib import sha1

from darts.engines import index_vector, value_vector

def dict_hash(dict: Dict[str, Any]) -> str:
    # returns a MD5 hash of a dictionary
    dhash = hashlib.md5()
    # sort arguments since {'a': 1, 'b': 2} is the same as {'b': 2, 'a': 1}

    dict_for_hash = {}
    for k in dict.keys():
        if k in ['self.contacts']:
            continue
        elif k in ['self.pm']:
            for kk in ['cell_m', 'cell_p', 'stencil', 'offset', 'tran', 'rhs', 'tran_biot', 'rhs_biot']:
                dict_for_hash[k + '.' + kk] = hash_array(getattr(dict[k], kk))
        elif k in ['self.unstr_discr']:
            for kk in ['mat_cells_tot', 'frac_cells_tot']:
                dict_for_hash[k + '.' + kk] = str(getattr(dict[k], kk))
        else:
            dict_for_hash[k] = hash_array(dict[k])

    print('dict_for_hash', dict_for_hash)
    encoded = json.dumps(dict_for_hash, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def hash_array(a):
    iv = type(index_vector())
    vv = type(value_vector())
    if type(a) in [iv, vv]:  # value/index_vector to numpy array
        tmp = np.array(a)
    else:
        tmp = a
    if isinstance(tmp, np.ndarray):
        b = sha1(tmp).hexdigest()  # str with a hash of an array
    else:
        b = a  # or the same object
    return b