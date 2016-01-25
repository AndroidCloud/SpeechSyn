import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import theano

from cle.cle.layers.cost import GaussianLayer
from cle.cle.utils import unpickle

from scipy.io import wavfile
from sk.datasets.blizzard_h5 import Blizzard_h5_tbptt


data_path = '/raid/chungjun/data/blizzard_unseg/'
save_path = '/raid/chungjun/repos/sk/cle/models/nips2015/blizzard/sample/'
exp_path = '/raid/chungjun/repos/sk/cle/models/nips2015/blizzard/pkl/'

frame_size = 200
# How many samples to generate
batch_size = 10
# How many timesteps to generate
n_steps = 480
debug = 0

exp_name = 'm2_3'
save_name = 'm2_3_sample_'

file_name = 'blizzard_unseg_tbptt'
normal_params = np.load(data_path + file_name + '_normal.npz')
X_mean = normal_params['X_mean']
X_std = normal_params['X_std']

exp = unpickle(exp_path + exp_name + '_best.pkl')
nodes = exp.model.nodes
names = [node.name for node in nodes]

output = GaussianLayer(name='output',
                       parent=['theta_mu',
                               'theta_sig'],
                       use_sample=1,
                       nout=frame_size)

[main_lstm, prior, kl,
 x_1, x_2, x_3, x_4,
 z_1, z_2, z_3, z_4,
 phi_1, phi_2, phi_3, phi_4, phi_mu, phi_sig,
 prior_1, prior_2, prior_3, prior_4, prior_mu, prior_sig,
 theta_1, theta_2, theta_3, theta_4, theta_mu, theta_sig] = nodes


def inner_fn(s_tm1):

    prior_1_t = prior_1.fprop([s_tm1])
    prior_2_t = prior_2.fprop([prior_1_t])
    prior_3_t = prior_3.fprop([prior_2_t])
    prior_4_t = prior_4.fprop([prior_3_t])
    prior_mu_t = prior_mu.fprop([prior_4_t])
    prior_sig_t = prior_sig.fprop([prior_4_t])

    z_t = prior.fprop([prior_mu_t, prior_sig_t])

    z_1_t = z_1.fprop([z_t])
    z_2_t = z_2.fprop([z_1_t])
    z_3_t = z_3.fprop([z_2_t])
    z_4_t = z_4.fprop([z_3_t])

    theta_1_t = theta_1.fprop([z_4_t, s_tm1])
    theta_2_t = theta_2.fprop([theta_1_t])
    theta_3_t = theta_3.fprop([theta_2_t])
    theta_4_t = theta_4.fprop([theta_3_t])
    theta_mu_t = theta_mu.fprop([theta_4_t])
    theta_sig_t = theta_sig.fprop([theta_4_t])

    x_t = output.fprop([theta_mu_t, theta_sig_t])

    x_1_t = x_1.fprop([x_t])
    x_2_t = x_2.fprop([x_1_t])
    x_3_t = x_3.fprop([x_2_t])
    x_4_t = x_4.fprop([x_3_t])
 
    s_t = main_lstm.fprop([[x_4_t, z_4_t], [s_tm1]])

    return s_t, x_t, theta_mu_t

((s_t, y, m), updates) =\
    theano.scan(fn=inner_fn,
                outputs_info=[main_lstm.get_init_state(batch_size),
                              None, None],
                n_steps=n_steps)

for k, v in updates.iteritems():
    k.default_update = v

test_fn = theano.function(inputs=[],
                          outputs=[y, m],
                          updates=updates,
                          allow_input_downcast=True,
                          on_unused_input='ignore')

samples = test_fn()[-1]
samples = np.transpose(samples, (1, 0, 2))
samples = samples.reshape(batch_size, -1)
samples = samples * X_std + X_mean

fig_size = (100, 30)
yet_another_save_name = 'm3_trunc_bptt_deeper_sample_zoom_in_'
if not os.path.exists(save_path):
    os.mkdir(save_path)
for i, sample in enumerate(samples):
    wave_path = save_path + save_name + str(i) + '.wav'
    wavfile.write(wave_path, 16000, np.int16(sample))
    sample_path = save_path + save_name + str(i) + '.png'
    fig = plt.figure()
    plt.plot(sample)
    plt.savefig(sample_path, bbox_inches='tight', format='png')
    sample_path = save_path + yet_another_save_name + str(i) + '.png'
    fig = plt.figure(figsize=fig_size)
    sample = sample[12000:18000]
    plt.plot(sample)
    #plt.plot([200, 200], [-1000, 1000], 'r-', lw=2)
    #plt.plot([400, 400], [-1000, 1000], 'r-', lw=2)
    #plt.plot([600, 600], [-1000, 1000], 'r-', lw=2)
    #plt.plot([800, 800], [-1000, 1000], 'r-', lw=2)
    #plt.plot([1000, 1000], [-1000, 1000], 'r-', lw=2)
    plt.savefig(sample_path, bbox_inches='tight', format='png')


data_path = '/raid/chungjun/data/blizzard/'
train_data = Blizzard_h5_tbptt(name='train',
                               path=data_path,
                               frame_size=frame_size,
                               X_mean=39.049,
                               X_std=3203.767)

save_name = 'ground_truth_zoom_in_'
sub_X = train_data.data[:10]
for i, x in enumerate(sub_X):
    sample = x.flatten()
    sample_path = save_path + save_name + str(i) + '.png'
    fig = plt.figure(figsize=fig_size)
    sample = sample[1000:7000]
    plt.plot(sample)
    #plt.plot([200, 200], [-1000, 1000], 'r-', lw=2)
    #plt.plot([400, 400], [-1000, 1000], 'r-', lw=2)
    #plt.plot([600, 600], [-1000, 1000], 'r-', lw=2)
    #plt.plot([800, 800], [-1000, 1000], 'r-', lw=2)
    #plt.plot([1000, 1000], [-1000, 1000], 'r-', lw=2)
    plt.savefig(sample_path, bbox_inches='tight', format='png')
