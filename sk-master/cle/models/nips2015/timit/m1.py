import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.data import Iterator
from cle.cle.cost import GMM
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
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
from sk.datasets.timit import TIMIT


#data_path = '/home/junyoung/data/timit/readable/'
#save_path = '/home/junyoung/repos/sk/cle/models/nips2015/timit/pkl/'
data_path = '/data/lisa/data/timit/readable/'
save_path = '/data/lisatmp/chungjun/nips2015/timit/pkl/'

batch_size = 64
frame_size = 200
main_lstm_dim = 2000
p_x_dim = 450
x2s_dim = 450
k = 20
target_size = frame_size * k
lr = 1e-3
debug = 0

model = Model()
train_data = TIMIT(name='train',
                   path=data_path,
                   frame_size=frame_size,
                   shuffle=0,
                   use_n_gram=1)

X_mean = train_data.X_mean
X_std = train_data.X_std

valid_data = TIMIT(name='valid',
                   path=data_path,
                   frame_size=frame_size,
                   shuffle=0,
                   use_n_gram=1,
                   X_mean=X_mean,
                   X_std=X_std)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')
init_b_sig = InitCell('const', mean=0.6)

x, x_mask = train_data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((15, batch_size, frame_size), dtype=np.float32)
    temp = np.ones((15, batch_size), dtype=np.float32)
    temp[:, -2:] = 0.
    x_mask.tag.test_value = temp

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

x_shape = x.shape
x_in = x.reshape((x_shape[0]*x_shape[1], -1))
x_1_in = x_1.fprop([x_in])
x_2_in = x_2.fprop([x_1_in])
x_3_in = x_3.fprop([x_2_in])
x_4_in = x_4.fprop([x_3_in])
x_4_in = x_4_in.reshape((x_shape[0], x_shape[1], -1))
s_0 = main_lstm.get_init_state(batch_size)


def inner_fn(x_t, s_tm1):

    s_t = main_lstm.fprop([[x_t], [s_tm1]])

    return s_t

(s_t, updates) = theano.scan(fn=inner_fn,
                             sequences=[x_4_in],
                             outputs_info=[s_0])

for k, v in updates.iteritems():
    k.default_update = v

s_t = s_t[:-1]
s_shape = s_t.shape
s_in = T.concatenate([s_0, s_t.reshape((s_shape[0]*s_shape[1], -1))], axis=0)
theta_1_in = theta_1.fprop([s_in])
theta_2_in = theta_2.fprop([theta_1_in])
theta_3_in = theta_3.fprop([theta_2_in])
theta_4_in = theta_4.fprop([theta_3_in])
theta_mu_in = theta_mu.fprop([theta_4_in])
theta_sig_in = theta_sig.fprop([theta_4_in])
coeff_in = coeff.fprop([theta_4_in])

recon = GMM(x_in, theta_mu_in, theta_sig_in, coeff_in)
recon = recon.reshape((x_shape[0], x_shape[1]))
recon = recon * x_mask
recon_term = recon.sum(axis=0).mean()
recon_term.name = 'nll'

max_x = x.max()
mean_x = x.mean()
min_x = x.min()
max_x.name = 'max_x'
mean_x.name = 'mean_x'
min_x.name = 'min_x'

max_theta_mu = theta_mu_in.max()
mean_theta_mu = theta_mu_in.mean()
min_theta_mu = theta_mu_in.min()
max_theta_mu.name = 'max_theta_mu'
mean_theta_mu.name = 'mean_theta_mu'
min_theta_mu.name = 'min_theta_mu'

max_theta_sig = theta_sig_in.max()
mean_theta_sig = theta_sig_in.mean()
min_theta_sig = theta_sig_in.min()
max_theta_sig.name = 'max_theta_sig'
mean_theta_sig.name = 'mean_theta_sig'
min_theta_sig.name = 'min_theta_sig'

coeff_max = coeff_in.max()
coeff_min = coeff_in.min()
coeff_mean_max = coeff_in.mean(axis=0).max()
coeff_mean_min = coeff_in.mean(axis=0).min()
coeff_max.name = 'coeff_max'
coeff_min.name = 'coeff_min'
coeff_mean_max.name = 'coeff_mean_max'
coeff_mean_min.name = 'coeff_mean_min'

model.inputs = [x, x_mask]
model._params = params
model.nodes = nodes

optimizer = Adam(
    lr=lr
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(200),
    Monitoring(freq=64,
               ddout=[recon_term,
                      max_theta_sig, mean_theta_sig, min_theta_sig,
                      max_x, mean_x, min_x,
                      max_theta_mu, mean_theta_mu, min_theta_mu],
               data=[Iterator(valid_data, batch_size)]),
    Picklize(freq=64, path=save_path),
    EarlyStopping(freq=64, path=save_path),
    WeightNorm()
]

mainloop = Training(
    name='m1_2',
    data=Iterator(train_data, batch_size),
    model=model,
    optimizer=optimizer,
    cost=recon_term,
    outputs=[recon_term],
    extension=extension
)
mainloop.run()
