import argparse

import matplotlib.pyplot as plt
import numpy as np
from darts.engines import redirect_darts_output
from model import Model

from darts.physics.base.operators_base import PropertyOperators as props


def parse_args():
    parser = argparse.ArgumentParser(description="Two-phase compositional model")
    parser.add_argument(
        "platform",
        nargs="?",
        default="cpu",
        help="Compute platform positional (cpu/gpu); accepted for run_test_suite2 compatibility",
    )
    parser.add_argument(
        "--transport-scheme",
        choices=("spu", "weno2"),
        default="spu",
        help="Advective transport reconstruction (default: spu)",
    )
    parser.add_argument("--runtime", type=float, default=1000.0)
    parser.add_argument("--nx", type=int, default=1000)
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip interactive and file plots"
    )
    return parser.parse_args()


def plot_sol(n):
    Xn = np.array(n.physics.engine.X, copy=False)
    nc = n.property_container.nc + n.thermal
    P = Xn[0 : n.reservoir.nb * nc : nc]
    z = np.ones((nc, n.reservoir.nb))
    phi = np.ones(n.reservoir.nb)
    sat_ev = props(n.property_container)
    prop = np.zeros(2 * n.property_container.nph)

    plt.figure(num=1, figsize=(12, 8), dpi=100)
    for i in range(nc - 1):
        z[i][:] = Xn[i + 1 : n.reservoir.nb * nc : nc]
        z[-1][:] -= z[i][:]

    for i in range(n.reservoir.nb):
        state = Xn[i * nc : (i + 1) * nc]
        sat_ev.evaluate(state, prop)
        phi[i] -= prop[2]  # (z[-1, i] * density_tot / prop[-1])

    for i in range(3):
        plt.subplot(330 + (i + 1))
        plt.plot(z[i] / (1 - z[3]))
        plt.title('Composition' + str(i + 1), y=1)

    i = 3
    plt.subplot(330 + (i + 1))
    plt.plot(phi)
    plt.title('Porosity', y=1)

    i = 4
    plt.subplot(330 + (i + 1))
    plt.plot(P)
    plt.title('Pressure', y=1)

    plt.show()


if __name__ == '__main__':
    args = parse_args()
    redirect_darts_output(f'run_{args.transport_scheme}.log')
    n = Model(transport_scheme=args.transport_scheme, nx=args.nx, runtime=args.runtime)
    # n.params.linear_type = n.params.linear_solver_t.cpu_superlu
    n.init()
    n.set_output()

    n.run(args.runtime)
    n.print_timers()
    n.print_stat()

    n.output.store_well_time_data(save_output_files=True)
    if not args.no_plots:
        n.output.plot_well_time_data(phase_volumetric_rates=True)

    if not args.no_plots:
        Xn = np.array(n.physics.engine.X, copy=False)
        nc = n.physics.nc + n.physics.thermal
        nb = n.reservoir.mesh.n_res_blocks

        plt.figure(num=1, figsize=(12, 8), dpi=100)
        for i in range(nc if nc < 3 else 3):
            plt.subplot(330 + (i + 1))
            plt.plot(Xn[i : nb * nc : nc])
        plt.savefig('out.png')

# z_c10 = Xn[nc-1:n.reservoir.nb*nc:nc]

# rho_aq = n.property_container.density_ev['wat'].evaluate(P, z_co2)
# Sg = np.zeros(n.reservoir.nb)
#
# for i in range (n.reservoir.nb):
#     x_list = Xn[i*nc:(i+1)*nc]
#     state = value_vector(x_list)
#     Sg[i] = n.properties(state)

# """ start plots """
# plt.figure(num=1, figsize=(12, 8), dpi=100)
# """ sg and x """
# plt.subplot(211)
# plt.plot(z1)
# #plt.imshow(np.reshape(T, (220, 60)).T)
#
# plt.title('First composition', y=1)
#
# plt.subplot(212)
# plt.plot(P)
# #plt.imshow(np.reshape(P, (220, 60)).T)
# plt.title('Pressure', y=1)

# """ sg and x """
# plt.subplot(223)
# plt.plot(P)
# plt.title('Pressure', y=1)
#
# plt.subplot(224)
# plt.plot(T)
# plt.title('Gas saturation', y=1)
