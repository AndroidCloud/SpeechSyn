import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.data import Iterator
from cle.cle.cost import BiGMM
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.layer import PriorLayer
from cle.cle.layers.recurrent import LSTM
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize,
    EarlyStopping,
    WeightNorm
)
from cle.cle.train.opt import Adam
from cle.cle.utils import flatten
from sk.datasets.iamondb import IAMOnDB
import os


data_path = '/data/lisatmp3/iamondb/'
save_path = os.path.join('/Tmp/', os.getenv('USER'))

batch_size = 100
frame_size = 3
main_lstm_dim = 1500
p_x_dim = 200
x2s_dim = 200
main_lstm_input_dim = 1000
k = 20
target_size = frame_size * k
lr = 0.001
debug = 0

model = Model()
train_data = IAMOnDB(name='train',
                     path=data_path)

valid_data = IAMOnDB(name='valid',
                     path=data_path)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')
init_b_sig = InitCell('const', mean=0.6)

x, mask = train_data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((15, batch_size, frame_size), dtype=np.float32)
    temp = np.ones((15, batch_size), dtype=np.float32)
    temp[:, -2:] = 0.
    mask.tag.test_value = temp

x_1 = FullyConnectedLayer(name='x_1',
                          parent=['x_t'],
                          parent_dim=[frame_size],
                          nout=x2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

x_2 = FullyConnectedLayer(name='x_2',
                          parent=['x_1'],
                          parent_dim=[x2s_dim],
                          nout=x2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

x_3 = FullyConnectedLayer(name='x_3',
                          parent=['x_2'],
                          parent_dim=[x2s_dim],
                          nout=x2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

x_4 = FullyConnectedLayer(name='x_4',
                          parent=['x_3'],
                          parent_dim=[x2s_dim],
                          nout=x2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

main_lstm = LSTM(name='main_lstm',
             parent=['x_4'],
             parent_dim=[x2s_dim],
             batch_size=batch_size,
             nout=main_lstm_dim,
             unit='tanh',
             init_W=init_W,
             init_U=init_U,
             init_b=init_b)

theta_1 = FullyConnectedLayer(name='theta_1',
                              parent=['s_tm1'],
                              parent_dim=[main_lstm_dim],
                              nout=p_x_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

theta_2 = FullyConnectedLayer(name='theta_2',
                              parent=['theta_1'],
                              parent_dim=[p_x_dim],
                              nout=p_x_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

theta_3 = FullyConnectedLayer(name='theta_3',
                              parent=['theta_2'],
                              parent_dim=[p_x_dim],
                              nout=p_x_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

theta_4 = FullyConnectedLayer(name='theta_4',
                              parent=['theta_3'],
                              parent_dim=[p_x_dim],
                              nout=p_x_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

theta_mu = FullyConnectedLayer(name='theta_mu',
                               parent=['theta_4'],
                               parent_dim=[p_x_dim],
                               nout=target_size,
                               unit='linear',
                               init_W=init_W,
                               init_b=init_b)

theta_sig = FullyConnectedLayer(name='theta_sig',
                                parent=['theta_4'],
                                parent_dim=[p_x_dim],
                                nout=target_size,
                                unit='softplus',
                                cons=1e-4,
                                init_W=init_W,
                                init_b=init_b_sig)

coeff = FullyConnectedLayer(name='coeff',
                            parent=['theta_4'],
                            parent_dim=[p_x_dim],
                            nout=k,
                            unit='softmax',
                            init_W=init_W,
                            init_b=init_b)

nodes = [main_lstm,
         x_1, x_2, x_3, x_4,
         theta_1, theta_2, theta_3, theta_4, theta_mu, theta_sig, coeff]

for node in nodes:
    node.initialize()

params = flatten([node.get_params().values() for node in nodes])


def inner_fn(x_t, s_tm1):

    theta_1_t = theta_1.fprop([s_tm1])
    theta_2_t = theta_2.fprop([theta_1_t])
    theta_3_t = theta_3.fprop([theta_2_t])
    theta_4_t = theta_4.fprop([theta_3_t])
    theta_mu_t = theta_mu.fprop([theta_4_t])
    theta_sig_t = theta_sig.fprop([theta_4_t])
    coeff_t = coeff.fprop([theta_4_t])

    x_1_t = x_1.fprop([x_t])
    x_2_t = x_2.fprop([x_1_t])
    x_3_t = x_3.fprop([x_2_t])
    x_4_t = x_4.fprop([x_3_t])
 
    s_t = main_lstm.fprop([[x_4_t], [s_tm1]])

    return s_t, theta_mu_t, theta_sig_t, coeff_t

((s_t, theta_mu_t, theta_sig_t, coeff_t), updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x],
                outputs_info=[main_lstm.get_init_state(batch_size),
                              None, None, None])

for k, v in updates.iteritems():
    k.default_update = v

reshaped_x = x.reshape((x.shape[0]*x.shape[1], -1))
reshaped_theta_mu = theta_mu_t.reshape((theta_mu_t.shape[0]*theta_mu_t.shape[1], -1))
reshaped_theta_sig = theta_sig_t.reshape((theta_sig_t.shape[0]*theta_sig_t.shape[1], -1))
reshaped_coeff = coeff_t.reshape((coeff_t.shape[0]*coeff_t.shape[1], -1))

recon = BiGMM(reshaped_x, reshaped_theta_mu, reshaped_theta_sig, reshaped_coeff)
recon = recon.reshape((theta_mu_t.shape[0], theta_mu_t.shape[1]))
recon = recon * mask
recon_term = recon.sum()
recon_term.name = 'nll'

max_x = x.max()
mean_x = x.mean()
min_x = x.min()
max_x.name = 'max_x'
mean_x.name = 'mean_x'
min_x.name = 'min_x'

max_theta_mu = theta_mu_t.max()
mean_theta_mu = theta_mu_t.mean()
min_theta_mu = theta_mu_t.min()
max_theta_mu.name = 'max_theta_mu'
mean_theta_mu.name = 'mean_theta_mu'
min_theta_mu.name = 'min_theta_mu'

max_theta_sig = theta_sig_t.max()
mean_theta_sig = theta_sig_t.mean()
min_theta_sig = theta_sig_t.min()
max_theta_sig.name = 'max_theta_sig'
mean_theta_sig.name = 'mean_theta_sig'
min_theta_sig.name = 'min_theta_sig'

coeff_max = reshaped_coeff.max()
coeff_min = reshaped_coeff.min()
coeff_mean_max = reshaped_coeff.mean(axis=0).max()
coeff_mean_min = reshaped_coeff.mean(axis=0).min()
coeff_max.name = 'coeff_max'
coeff_min.name = 'coeff_min'
coeff_mean_max.name = 'coeff_mean_max'
coeff_mean_min.name = 'coeff_mean_min'

model.inputs = [x, mask]
model._params = params
model.nodes = nodes

optimizer = Adam(
    lr=lr
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(1000000),
    Monitoring(freq=300,
               ddout=[recon_term,
                      max_theta_sig, mean_theta_sig, min_theta_sig,
                      max_x, mean_x, min_x,
                      max_theta_mu, mean_theta_mu, min_theta_mu,
                      coeff_max, coeff_min, coeff_mean_max, coeff_mean_min],
               data=[Iterator(valid_data, batch_size)]),
    Picklize(freq=300,
             path=save_path),
    EarlyStopping(freq=300, path=save_path),
    WeightNorm()
]

mainloop = Training(
    name='m0',
    data=Iterator(train_data, batch_size),
    model=model,
    optimizer=optimizer,
    cost=recon_term,
    outputs=[recon_term],
    extension=extension
)
mainloop.run()
