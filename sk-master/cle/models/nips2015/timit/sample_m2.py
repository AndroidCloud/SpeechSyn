import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import theano

from cle.cle.data import Iterator
from cle.cle.layers.cost import GaussianLayer
from cle.cle.utils import unpickle

from scipy.io import wavfile
from sk.datasets.timit import TIMIT


data_path = '/home/junyoung/data/timit/readable/'
save_path = '/home/junyoung/repos/sk/cle/models/nips2015/timit/sample/'
exp_path = '/home/junyoung/repos/sk/cle/models/nips2015/timit/pkl/'

frame_size = 200
label_size = 200
# How many samples to generate
batch_size = 1
num_sample = 10
debug = 1

exp_name = 'm7_cond_v2'
save_name = 'm7_cond_v2_sample_'

train_data = TIMIT(name='train',
                   path=data_path,
                   frame_size=frame_size,
                   shuffle=0,
                   use_n_gram=1)

X_mean = train_data.X_mean
X_std = train_data.X_std

test_data = TIMIT(name='test',
                  path=data_path,
                  frame_size=frame_size,
                  shuffle=0,
                  use_n_gram=1,
                  X_mean=X_mean,
                  X_std=X_std)

exp = unpickle(exp_path + exp_name + '_best.pkl')
nodes = exp.model.nodes
names = [node.name for node in nodes]

output = GaussianLayer(name='output',
                       parent=['theta_mu',
                               'theta_sig'],
                       use_sample=1,
                       nout=frame_size)

x, y, spk_info, mask = train_data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((15, batch_size, frame_size), dtype=np.float32)
    y.tag.test_value = np.zeros((15, batch_size, label_size), dtype=np.float32)
    temp = np.ones((15, batch_size), dtype=np.float32)
    temp[:, -2:] = 0.
    mask.tag.test_value = temp
    spk_info.tag.test_value = np.zeros((batch_size, 630), dtype=np.float32)

[main_lstm, prior, kl,
 x_1, x_2, x_3, x_4,
 z_1, z_2, z_3, z_4,
 y_1, y_2, y_3, y_4,
 phi_1, phi_2, phi_3, phi_4, phi_mu, phi_sig,
 prior_1, prior_2, prior_3, prior_4, prior_mu, prior_sig,
 theta_1, theta_2, theta_3, theta_4, theta_mu, theta_sig] = nodes


def inner_fn(y_t, s_tm1, spk_info):

    y_1_t = y_1.fprop([y_t, spk_info])
    y_2_t = y_2.fprop([y_1_t])
    y_3_t = y_3.fprop([y_2_t])
    y_4_t = y_4.fprop([y_3_t])

    prior_1_t = prior_1.fprop([s_tm1, y_4_t, spk_info])
    prior_2_t = prior_2.fprop([prior_1_t])
    prior_3_t = prior_3.fprop([prior_2_t])
    prior_4_t = prior_4.fprop([prior_3_t])
    prior_mu_t = prior_mu.fprop([prior_4_t, y_4_t, spk_info])
    prior_sig_t = prior_sig.fprop([prior_4_t, y_4_t, spk_info])

    z_t = prior.fprop([prior_mu_t, prior_sig_t])

    z_1_t = z_1.fprop([z_t, y_4_t, spk_info])
    z_2_t = z_2.fprop([z_1_t])
    z_3_t = z_3.fprop([z_2_t])
    z_4_t = z_4.fprop([z_3_t])

    theta_1_t = theta_1.fprop([z_4_t, s_tm1, y_4_t, spk_info])
    theta_2_t = theta_2.fprop([theta_1_t])
    theta_3_t = theta_3.fprop([theta_2_t])
    theta_4_t = theta_4.fprop([theta_3_t])
    theta_mu_t = theta_mu.fprop([theta_4_t, y_4_t, spk_info])
    theta_sig_t = theta_sig.fprop([theta_4_t, y_4_t, spk_info])

    x_t = output.fprop([theta_mu_t, theta_sig_t])
    x_1_t = x_1.fprop([x_t, y_4_t, spk_info])
    x_2_t = x_2.fprop([x_1_t])
    x_3_t = x_3.fprop([x_2_t])
    x_4_t = x_4.fprop([x_3_t])

    s_t = main_lstm.fprop([[x_4_t, z_4_t, y_4_t, spk_info], [s_tm1]])

    return s_t, theta_mu_t

((s_t, m_t), updates) =\
    theano.scan(fn=inner_fn,
                sequences=[y],
                outputs_info=[main_lstm.get_init_state(batch_size),
                              None],
                non_sequences=[spk_info])

for k, v in updates.iteritems():
    k.default_update = v

test_fn = theano.function(inputs=[y, spk_info],
                          outputs=[m_t],
                          updates=updates,
                          allow_input_downcast=True,
                          on_unused_input='ignore')

if not os.path.exists(save_path):
    os.mkdir(save_path)

DataProvider = [Iterator(train_data, batch_size, start=0, end=num_sample),
                Iterator(test_data, batch_size, start=0, end=num_sample)]

for data in DataProvider:
    cnt = 0
    for batch in data:
        sample = test_fn(batch[1], batch[2])[-1]
        sample = np.transpose(sample, (1, 0, 2))
        sample = sample.reshape(batch_size, -1).flatten()
        sample = sample * X_std + X_mean
        wave_path = save_path + save_name + data.name + '_' + str(cnt) + '.wav'
        wavfile.write(wave_path, 16000, np.int16(sample))
        sample_path = save_path + save_name + data.name + '_' + str(cnt) + '.png'
        fig = plt.figure()
        plt.plot(sample)
        plt.savefig(sample_path, bbox_inches='tight', format='png')
        print "Generating %s [%d / %d] sample" % (data.name, cnt+1, num_sample)
        cnt += 1
