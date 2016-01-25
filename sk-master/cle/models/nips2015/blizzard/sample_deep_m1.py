import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import theano

from cle.cle.layers.cost import GMMLayer
from cle.cle.utils import unpickle, tolist, OrderedDict

from scipy.io import wavfile


data_path = '/raid/chungjun/data/blizzard_unseg/'
save_path = '/raid/chungjun/repos/sk/cle/models/nips2015/blizzard/sample/'
pkl_path = '/raid/chungjun/repos/sk/cle/models/nips2015/blizzard/pkl/'

frame_size = 200
# How many samples to generate
batch_size = 10
# How many timesteps to generate
n_steps = 480
#n_steps = 160
debug = 0

pkl_name = 'deep_m1_2_30000updates.pkl'
save_name = 'deep_m1_2_30000updates_sample_'

file_name = 'blizzard_unseg_tbptt'
normal_params = np.load(data_path + file_name + '_normal.npz')
X_mean = normal_params['X_mean']
X_std = normal_params['X_std']

exp = unpickle(pkl_path + pkl_name)
nodes = exp.model.nodes
names = [node.name for node in nodes]

y = GMMLayer(name='y',
             parent=['theta_mu',
                     'theta_sig',
                     'coeff'],
             use_sample=1,
             nout=frame_size)

[lstm_1, lstm_2, lstm_3,
 x_1, x_2, x_3, x_4,
 theta_1, theta_2, theta_3, theta_4, theta_mu, theta_sig, coeff] = nodes
s_1_0 = lstm_1.get_init_state(batch_size)
s_2_0 = lstm_2.get_init_state(batch_size)
s_3_0 = lstm_3.get_init_state(batch_size)


def inner_fn(s_1_tm1, s_2_tm1, s_3_tm1):

    theta_1_t = theta_1.fprop([s_3_tm1])
    theta_2_t = theta_2.fprop([theta_1_t])
    theta_3_t = theta_3.fprop([theta_2_t])
    theta_4_t = theta_4.fprop([theta_3_t])
    theta_mu_t = theta_mu.fprop([theta_4_t])
    theta_sig_t = theta_sig.fprop([theta_4_t])
    coeff_t = coeff.fprop([theta_4_t])

    x_t, mu_t = y.sample_mean([theta_mu_t, theta_sig_t, coeff_t])

    x_1_t = x_1.fprop([x_t])
    x_2_t = x_2.fprop([x_1_t])
    x_3_t = x_3.fprop([x_2_t])
    x_4_t = x_4.fprop([x_3_t])

    s_1_t = lstm_1.fprop([[x_4_t], [s_1_tm1]])
    s_2_t = lstm_2.fprop([[s_1_t], [s_2_tm1]])
    s_3_t = lstm_3.fprop([[s_2_t], [s_3_tm1]])

    return s_1_t, s_2_t, s_3_t, x_t, mu_t

((s_1_t, s_2_t, s_3_t, y, m), updates) =\
    theano.scan(fn=inner_fn,
                outputs_info=[s_1_0, s_2_0, s_3_0, None, None],
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
#np.save('deep_m1_5_samples.npy', samples)

if not os.path.exists(save_path):
    os.mkdir(save_path)
for i, sample in enumerate(samples):
    wave_path = save_path + save_name + str(i) + '.wav'
    wavfile.write(wave_path, 16000, np.int16(sample))
    sample_path = save_path + save_name + str(i) + '.png'
    fig = plt.figure()
    plt.plot(sample)
    plt.savefig(sample_path, bbox_inches='tight', format='png')
