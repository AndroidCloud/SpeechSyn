import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import theano
import theano.tensor as T

from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers.cost import GaussianLayer
from cle.cle.utils import unpickle, tolist, OrderedDict

from scipy.io import wavfile
from sk.datasets.onomatopoeia import Onomatopoeia


data_path = '/raid/chungjun/data/ubisoft/onomatopoeia/'
save_path = '/raid/chungjun/repos/sk/cle/models/nips2015/onomatopoeia/sample/'
exp_path = '/raid/chungjun/repos/sk/cle/models/nips2015/onomatopoeia/pkl/'
#data_path = '/home/junyoung/data/ubisoft/onomatopoeia/'
#save_path = '/home/junyoung/repos/sk/cle/models/nips2015/onomatopoeia/sample/'
#exp_path = '/home/junyoung/repos/sk/cle/models/nips2015/onomatopoeia/pkl/'

frame_size = 100
# How many samples to generate
batch_size = 10
# How many timesteps to generate
n_steps = 150
debug = 0

exp_name = 'm3'
save_name = 'm3_sample_'

#data = Onomatopoeia(name='EKENWAY_test',
#                    path=datapath,
#                    frame_size=frame_size)
data = Onomatopoeia(name='valid',
                    path=data_path,
                    frame_size=frame_size)

X_mean = data.X_mean
X_std = data.X_std

exp = unpickle(exp_path + exp_name + '_best.pkl')
nodes = exp.model.nodes
names = [node.name for node in nodes]

output = GaussianLayer(name='output',
                       parent=['theta_mu',
                               'theta_sig'],
                       use_sample=1,
                       nout=frame_size)

[main_lstm, prior, kl,
 x_emb_1, x_emb_2,
 z_emb_1, z_emb_2,
 phi_emb_1, phi_emb_2, phi_mu, phi_sig,
 prior_emb_1, prior_emb_2, prior_mu, prior_sig,
 theta_emb_1, theta_emb_2, theta_mu, theta_sig] = nodes
s_0 = main_lstm.get_init_state(batch_size)


def inner_fn(s_tm1):

    prior_emb_1_t = prior_emb_1.fprop([s_tm1])
    prior_emb_2_t = prior_emb_2.fprop([prior_emb_1_t])
    prior_mu_t = prior_mu.fprop([prior_emb_2_t])
    prior_sig_t = prior_sig.fprop([prior_emb_2_t])

    z_t = prior.fprop([prior_mu_t, prior_sig_t])

    z_emb_1_t = z_emb_1.fprop([z_t])
    z_emb_2_t = z_emb_2.fprop([z_emb_1_t])

    theta_emb_1_t = theta_emb_1.fprop([z_emb_2_t, s_tm1])
    theta_emb_2_t = theta_emb_2.fprop([theta_emb_1_t])
    theta_mu_t = theta_mu.fprop([theta_emb_2_t])
    theta_sig_t = theta_sig.fprop([theta_emb_2_t])

    x_t = output.fprop([theta_mu_t, theta_sig_t])
    x_emb_1_t = x_emb_1.fprop([x_t])
    x_emb_2_t = x_emb_2.fprop([x_emb_1_t])

    s_t = main_lstm.fprop([[x_emb_2_t, z_emb_2_t], [s_tm1]])

    return s_t, x_t, theta_mu_t

((s_t, y, m), updates) =\
    theano.scan(fn=inner_fn,
                outputs_info=[s_0, None, None],
                n_steps=n_steps)

for k, v in updates.iteritems():
    k.default_update = v

test_fn = theano.function(inputs=[],
                          outputs=[y, m],
                          updates=updates,
                          allow_input_downcast=True,
                          on_unused_input='ignore')

samples = test_fn()[0]
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
