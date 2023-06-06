from darts.engines import *
from time import sleep
import sys


def test_itors(n_points=64, grav_pc=0, np=2, nc=2, nb=100, adaptive=True, static=False, single=False, long=True):
    # n_threads = 2
    # set_num_threads(n_threads)
    redirect_darts_output('')

    print("OBL resolution: ", n_points, end='; ')
    print("Graity & Capillarity: ", grav_pc, end='; ')
    print("Phases: ", np, end='; ')
    print("Components: ", nc, end='; ')
    print("Model size: %d " % nb, flush=True)

    if grav_pc:
        n_ops = nc + nc * np + 2 * np
    else:
        n_ops = 2 * nc

    names = []
    if adaptive:
        names.append(eval("operator_set_interpolator_i_d_%d_%d" % (nc, n_ops)))

        # if single:
        #     names.append(eval("operator_set_interpolator_l_f_%d_%d" % (nc, n_ops)))
        if long:
            names.append(eval("operator_set_interpolator_l_d_%d_%d" % (nc, n_ops)))

    if static:
        # CPU
        names.append(eval("operator_set_interpolator_static_i_d_%d_%d" % (nc, n_ops)))

        if single:
            names.append(eval("operator_set_interpolator_static_i_f_%d_%d" % (nc, n_ops)))
        #GPU
        names.append(eval("operator_set_interpolator_gpu_i_d_%d_%d" % (nc, n_ops)))
        if single:
            names.append(eval("operator_set_interpolator_gpu_i_f_%d_%d" % (nc, n_ops)))

    # names.append (eval("operator_set_interpolator_static_i_d_%d_%d" % (nc, n_ops)))
    # names.append ( eval("operator_set_interpolator_gpu_%d_%d" % (nc, n_ops)))

    fake_os = operator_set_from_files(nc, n_ops)
    fake_os.axis_points = index_vector([n_points] * nc)

    tables = []

    for n in names:
        try:
            tables.append(n(None, fake_os.axis_points, fake_os.axis_min, fake_os.axis_max))
        except MemoryError:
            tables.append(None)

    for i, t in enumerate(tables):
        print('{:80s}'.format(str(names[i])), end='', flush=True)
        if t is not None:
            t.benchmark(n_points, nb, 0.1 ** nc, 1)
        else:
            print('--------------')

    for t in tables:
        del t


# adaptive first

for nc in range(2, 6):
    for nb in [100, 10000, 1000000]:
        test_itors(n_points=64,grav_pc=0, np=2, nc=nc, nb=nb, adaptive=True, static=False, single=False)

#now with static
for nc in range(2, 5):
    for nb in [100, 10000, 1000000]:
        test_itors(n_points=40, grav_pc=0, np=2, nc=nc, nb=nb, adaptive=True, static=True, single=True)
