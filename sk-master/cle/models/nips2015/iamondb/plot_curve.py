import ipdb
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from cle.cle.utils import unpickle

dir_path = '/raid/chungjun/repos/sk/cle/models/nips2015/blizzard/'
models = ['m0_1', 'm1_1', 'm2_1']
labels = ['Baseline NLL', 'RNN-VAE w/o prior RNN NLL upper bound ', 'RNN-VAE w/ prior RNN NLL upper bound']
colors = ['r', 'g', 'b']
save_name = 'valid'
max_epoch = -1

fig = plt.figure()
for i, model in enumerate(models):
    exp = unpickle(dir_path + 'pkl/' + model + '_best.pkl')
    mon = np.asarray(exp.trainlog._ddmonitors)

    nll_lower_bound = mon[:max_epoch, 0]
    legend_size = 10

    nll_lower_bound = np.clip(nll_lower_bound, a_min=-2000, a_max=50)

    plt.plot(nll_lower_bound, color=colors[i], label=labels[i])
    plt.legend(loc='upper right', prop={'size': legend_size})
plt.savefig(dir_path + save_name + '_curves.png', bbox_inches='tight', format='png')
