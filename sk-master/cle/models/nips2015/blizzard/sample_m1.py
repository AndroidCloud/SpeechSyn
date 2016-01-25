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


save_path = '/raid/chungjun/repos/sk/cle/models/nips2015/blizzard/sample/'
exp_path = '/raid/chungjun/repos/sk/cle/models/nips2015/blizzard/pkl/'

frame_size = 200
# How many samples to generate
batch_size = 10
# How many timesteps to generate
n_steps = 160
debug = 0

exp_name = 'baseline'
save_name = 'baseline_sample_'

X_mean = 39.034
X_std = 3155.069

exp = unpickle(exp_path + exp_name + '_best.pkl')
nodes = exp.model.nodes
names = [node.name for node in nodes]

y = GMMLayer(name='y',
             parent=['theta_mu',
                     'theta_sig',
                     'coeff'],
             use_sample=1,
             nout=frame_size)

for node in nodes:
    if hasattr(node, 'batch_size'):
        node.batch_size = batch_size

[main_lstm,
 x_1, x_2,
 theta_1, theta_2, theta_mu, theta_sig, coeff] = nodes


def inner_fn(s_tm1):

    theta_1_t = theta_1.fprop([s_tm1])
    theta_2_t = theta_2.fprop([theta_1_t])
    theta_mu_t = theta_mu.fprop([theta_2_t])
    theta_sig_t = theta_sig.fprop([theta_2_t])
    coeff_t = coeff.fprop([theta_2_t])

    x_t = y.fprop([theta_mu_t, theta_sig_t, coeff_t])
    mu_t = y.argmax_mean([theta_mu_t, coeff_t])

    x_1_t = x_1.fprop([x_t])
    x_2_t = x_2.fprop([x_1_t])

    s_t = main_lstm.fprop([[x_2_t], [s_tm1]])

    return s_t, x_t, mu_t

((s_t, y, m), updates) =\
    theano.scan(fn=inner_fn,
                outputs_info=[main_lstm.get_init_state(), None, None],
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

if not os.path.exists(save_path):
    os.mkdir(save_path)
for i, sample in enumerate(samples):
    wave_path = save_path + save_name + str(i) + '.wav'
    wavfile.write(wave_path, 16000, np.int16(sample))
    sample_path = save_path + save_name + str(i) + '.png'
    fig = plt.figure()
    plt.plot(sample)
    plt.savefig(sample_path, bbox_inches='tight', format='png')
