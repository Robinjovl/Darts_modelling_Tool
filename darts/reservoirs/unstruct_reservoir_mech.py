import numpy as np

from darts.discretizer import elem_loc

class bound_cond:
    '''
    General representation of boundary condition: a*p + b*f = r (a=1,b=0 - Dirichlet, a=0,b=1 - Neumann)
    '''
    def __init__(self):
        # flow
        self.NO_FLOW = {'a': 0.0, 'b': 1.0, 'r': 0.0}
        self.AQUIFER = lambda p: {'a': 1.0, 'b': 0.0, 'r': p}
        # mechanics
        self.ROLLER = {'an': 1.0, 'bn': 0.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        self.FREE = {'an': 0.0, 'bn': 1.0, 'rn': 0.0, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0, 0, 0])}
        self.STUCK = lambda un, ut: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 1.0, 'bt': 0.0, 'rt': np.array(ut)}
        self.LOAD = lambda Fn, Ft: {'an': 0.0, 'bn': 1.0, 'rn': Fn, 'at': 0.0, 'bt': 1.0, 'rt': np.array(Ft)}
        self.STUCK_ROLLER = lambda un: {'an': 1.0, 'bn': 0.0, 'rn': un, 'at': 0.0, 'bt': 1.0, 'rt': np.array([0.0, 0.0, 0.0])}

def set_domain_tags(matrix_tags,
                    bnd_xm_tag, bnd_xp_tag,
                    bnd_ym_tag, bnd_yp_tag,
                    bnd_zm_tag, bnd_zp_tag,
                    fracture_tags=[], frac_bnd_tags=[]):
    '''
    :param matrix_tag: list of integers
    :param bnd_tags: list of integers
    :param fracture_tag: list of integers
    :param frac_bnd_tag: list of integers
    :return: dictionary of sets containing integer tags for each element type; dictionary of tags for 6 boundaries
    '''
    boundary_tags = [bnd_xm_tag, bnd_xp_tag, bnd_ym_tag, bnd_yp_tag, bnd_zm_tag, bnd_zp_tag]
    domain_tags = dict()
    domain_tags[elem_loc.MATRIX] = set(matrix_tags)
    domain_tags[elem_loc.FRACTURE] = set(fracture_tags)
    domain_tags[elem_loc.BOUNDARY] = set(boundary_tags)
    domain_tags[elem_loc.FRACTURE_BOUNDARY] = set(frac_bnd_tags)

    bnd_tags = dict()
    bnd_tags['BND_X-'] = bnd_xm_tag
    bnd_tags['BND_X+'] = bnd_xp_tag
    bnd_tags['BND_Y-'] = bnd_ym_tag
    bnd_tags['BND_Y+'] = bnd_yp_tag
    bnd_tags['BND_Z-'] = bnd_zm_tag
    bnd_tags['BND_Z+'] = bnd_zp_tag

    return domain_tags, bnd_tags


class reservoir_mech:
    '''
    '''
    def __init__(self, discretizer='mech_discretizer'):
        self.discretizer = discretizer