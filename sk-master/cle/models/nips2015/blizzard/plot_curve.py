import ipdb
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from cle.cle.utils import unpickle

dir_path = '/raid/chungjun/repos/sk/cle/models/nips2015/blizzard/'
models = ['m0_1', 'm1_1', 'm2_1', 'm3_1']
#models = ['m0_2', 'm1_2', 'm2_2']
#models = ['m0_3', 'm1_3', 'm2_3']
#models = ['m0_4', 'm1_4', 'm2_4']
labels = ['RNN-Gaussian', 'RNN-GMM', 'RNN-VAEGauss', 'RNN-VAEGMM']
colors = ['r', 'g', 'b', 'c']
save_name = 'valid_curves_1.png'

fig = plt.figure()
for i, model in enumerate(models):
    #exp = unpickle(dir_path + 'pkl/' + model + '_best.pkl')
    exp = unpickle(dir_path + 'pkl/' + model + '.pkl')
    mon = np.asarray(exp.trainlog._ddmonitors)

    nll_lower_bound = mon[:, 0]
    legend_size = 10

    plt.plot(nll_lower_bound, color=colors[i], label=labels[i])
    plt.xscale('log')
plt.legend(loc='upper right', prop={'size': legend_size})
plt.grid()
plt.savefig(dir_path + save_name, bbox_inches='tight', format='png')
