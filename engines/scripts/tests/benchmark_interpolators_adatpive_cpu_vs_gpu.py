import numpy as np
import pandas as pd
from darts.engines import *
import matplotlib.pyplot as plt
import seaborn as sns
from benchmark_interpolators_tools import *

df_filename = 'test.csv'
load_df = 0
save_df = 1
plot = 1

def make_itors(n_dims=2, n_ops=0, n_points=64, min=0, max=10):
    axes_n_points = index_vector([n_points] * n_dims)
    axes_min = value_vector([min] * n_dims)
    axes_max = value_vector([max] * n_dims)

    # set_num_threads(n_threads)
    redirect_darts_output('')

    if n_ops == 0:
        n_ops = 2 * n_dims

    itors = []

    itors.append(
        test_itor(n_dims, n_ops, axes_n_points, axes_min, axes_max, 'multilinear', 'adaptive', '2', 'cpu', 'd', 'i'))
    itors.append(
        test_itor(n_dims, n_ops, axes_n_points, axes_min, axes_max, 'multilinear', 'adaptive', '3', 'gpu', 'd', 'i'))

    return itors

if load_df:
    df = pd.read_csv(df_filename)
else:
    df = pd.DataFrame()

if save_df:

    for n_dims in range(7, 8):
        for n_points in [2]:
            itors = make_itors(n_dims, n_points=n_points)
            for nb in [100, 1000, 10000, 100000, 1000000]:
            #for nb in [100, 1000, 10000]:
                X = value_vector(np.random.random(nb * n_dims))
                ref = 0
                for i in itors:
                    i.prepare_to_interpolate(nb)
                    if type(ref) == int:
                        ref = i.interpolate_array(X)
                    else:
                        i.interpolate_array(X)
                        if not i.validate_last_result(ref):
                            print('Validation fail for %s' % i)
                    i.reset_timing()

                # interleave timings several times
                for k in range(5):
                    for i in itors:
                        for r in range(10):
                            i.interpolate_array(X)

                print (n_dims, n_points, nb, itors[0].min_interpolation_time / itors[1].min_interpolation_time)
                # for i in itors:
                #     print(i.name, i.min_interpolation_time, i.max_interpolation_time, i.avg_interpolation_time)
                #     df = df.append({'name': i.name,
                #                 'n_dims': i.n_dims,
                #                 'n_ops': i.n_ops,
                #                 'n_points':n_points,
                #                 'type': i.type,
                #                 'mode': i.mode,
                #                 'version': i.version,
                #                 'platform': i.platform,
                #                 'precision': i.precision,
                #                 'index': i.index,
                #                 'nb': nb,
                #                 'min': i.min_interpolation_time,
                #                 'max': i.max_interpolation_time,
                #                 'avg': i.avg_interpolation_time,
                #                 }, ignore_index=True)

    #df.to_csv(df_filename)


if plot:
    df['avg(OMIPS)'] = df['nb'] * df['n_ops'] / df['min'] / 1000000
    #df['std(OMIPS)'] = (df['nb'] * df['n_ops'] / df['max'] - df['nb'] * df['n_ops'] / df['min']) / 2 / 1000000
    #print(df['avg(OMIPS)'])
    #print(df['std(OMIPS)'])
    #df['ver_prec']
    g = sns.relplot(
        data=df,
        x="nb", y="avg(OMIPS)",
        hue="n_dims", style="platform", size="n_points",
    )
    g.set(xscale="log")

    #df.plot(kind='bar', x='nb', y='avg(OMIPS)', yerr='std(OMIPS)')
    plt.show()


#
#
# # now with static
# for nc in range(2, 5):
#     for nb in [100, 10000, 1000000]:
#         test_itors(n_points=40, grav_pc=0, np=2, nc=nc, nb=nb, adaptive=True, static=True, single=True)
