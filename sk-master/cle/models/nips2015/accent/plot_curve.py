import ipdb
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from cle.cle.utils import unpickle

dir_path = '/home/junyoung/repos/sk/cle/models/nips2015/accent/'
#dir_path = '/raid/chungjun/repos/sk/cle/models/nips2015/accent/'
#models = ['m1_1', 'm2_1', 'm2_prior', 'm3_1', 'storn0_1']
models = ['m1_1', 'm2_1', 'm3_1', 'storn0_1']
colors = ['g', 'b', 'c', 'm']
labels = ['RNN-GMM', 'RNN-VAEGauss', 'RNN-VAEGMM', 'STORN-Gauss']
save_name = 'valid_curves_1.png'

fig = plt.figure()
for i, model in enumerate(models):
    #exp = unpickle(dir_path + 'pkl/' + model + '_best.pkl')
    exp = unpickle(dir_path + 'pkl/' + model + '.pkl')
    mon = np.asarray(exp.trainlog._ddmonitors)

    valid_nll_lower_bound = mon[:, 0]
    legend_size = 10
    print valid_nll_lower_bound.min()

    plt.plot(valid_nll_lower_bound, linestyle='-', color=colors[i], label=labels[i])
    #plt.xscale('log')

#valid_nll_lower_bound = np.load(dir_path + 'pkl/valid_m2_1.npy')
#plt.plot(valid_nll_lower_bound, linestyle='-', color='b', label='RNN-VAE')
plt.legend(loc='upper right', prop={'size': legend_size})
plt.grid()
plt.savefig(dir_path + save_name, bbox_inches='tight', format='png')
