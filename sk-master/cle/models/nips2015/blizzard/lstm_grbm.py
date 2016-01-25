import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.data import Iterator
from cle.cle.cost import grbm_free_energy
from cle.cle.models import Model
from cle.cle.layers import InitCell, RealVectorLayer
from cle.cle.layers.feedforward import FullyConnectedLayer, GRBM
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
from cle.cle.utils import flatten, sharedX
from cle.cle.utils.compat import OrderedDict
from sk.datasets.blizzard_h5 import Blizzard_h5_tbptt


#data_path = '/raid/chungjun/data/blizzard_unseg/'
#save_path = '/raid/chungjun/repos/sk/cle/models/nips2015/blizzard/pkl/'
data_path = '/home/junyoung/data/blizzard_unseg/'
save_path = '/home/junyoung/repos/sk/cle/models/nips2015/blizzard/pkl/'

reset_freq = 4
batch_size = 128
frame_size = 200
latent_size = 200
lstm_1_dim = 200
lstm_2_dim = 200
lstm_3_dim = 200
p_x_dim = 100
x2s_dim = 100
z2s_dim = 100
grbm_dim = 200
lr = 1e-3
debug = 1

file_name = 'blizzard_unseg_tbptt'
normal_params = np.load(data_path + file_name + '_normal.npz')
X_mean = normal_params['X_mean']
X_std = normal_params['X_std']
 
model = Model()
train_data = Blizzard_h5_tbptt(name='train',
                               path=data_path,
                               frame_size=frame_size,
                               file_name=file_name,
                               X_mean=X_mean,
                               X_std=X_std)

valid_data = Blizzard_h5_tbptt(name='valid',
                               path=data_path,
                               frame_size=frame_size,
                               file_name=file_name,
                               X_mean=X_mean,
                               X_std=X_std)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')
init_b_sig = InitCell('const', mean=0.6)

x = train_data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((15, batch_size, frame_size), dtype=theano.config.floatX)

x_1 = FullyConnectedLayer(name='x_1',
                          parent=['grbm'],
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

z_1 = FullyConnectedLayer(name='z_1',
                          parent=['z_t'],
                          parent_dim=[latent_size],
                          nout=z2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

z_2 = FullyConnectedLayer(name='z_2',
                          parent=['z_1'],
                          parent_dim=[z2s_dim],
                          nout=z2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

z_3 = FullyConnectedLayer(name='z_3',
                          parent=['z_2'],
                          parent_dim=[z2s_dim],
                          nout=z2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

z_4 = FullyConnectedLayer(name='z_4',
                          parent=['z_3'],
                          parent_dim=[z2s_dim],
                          nout=z2s_dim,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

lstm_1 = LSTM(name='lstm_1',
              parent=['x_4', 'z_4'],
              parent_dim=[x2s_dim, z2s_dim],
              batch_size=batch_size,
              nout=lstm_1_dim,
              unit='tanh',
              init_W=init_W,
              init_U=init_U,
              init_b=init_b)

lstm_2 = LSTM(name='lstm_2',
              parent=['lstm_1'],
              parent_dim=[lstm_1_dim],
              batch_size=batch_size,
              nout=lstm_2_dim,
              unit='tanh',
              init_W=init_W,
              init_U=init_U,
              init_b=init_b)

lstm_3 = LSTM(name='lstm_3',
              parent=['lstm_2'],
              parent_dim=[lstm_2_dim],
              batch_size=batch_size,
              nout=lstm_3_dim,
              unit='tanh',
              init_W=init_W,
              init_U=init_U,
              init_b=init_b)

theta_1 = FullyConnectedLayer(name='theta_1',
                              parent=['s_3_tm1'],
                              parent_dim=[lstm_3_dim],
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

theta_b = FullyConnectedLayer(name='theta_b',
                              parent=['theta_4'],
                              parent_dim=[p_x_dim],
                              nout=grbm_dim,
                              unit='linear',
                              init_W=init_W,
                              init_b=init_b)

theta_c = FullyConnectedLayer(name='theta_c',
                              parent=['theta_4'],
                              parent_dim=[p_x_dim],
                              nout=frame_size,
                              unit='linear',
                              init_W=init_W,
                              init_b=init_b)

theta_sig = RealVectorLayer(name='theta_sig',
                            #nout=frame_size,
                            nout=1,
                            unit='softplus',
                            cons=1e-4,
                            lr_scaler=0.01,
                            init_b=init_b_sig)

grbm = GRBM(name='grbm',
            parent=['x', 'theta_b', 'theta_c', 'theta_sig'],
            parent_dim=[frame_size, grbm_dim, frame_size, frame_size],
            nout=grbm_dim,
            k_step=15,
            init_W=init_W)

nodes = [lstm_1, lstm_2, lstm_3,
         x_1, x_2, x_3, x_4,
         z_1, z_2, z_3, z_4,
         theta_1, theta_2, theta_3, theta_4, theta_b, theta_c, theta_sig, grbm]

for node in nodes:
    node.initialize()

params = flatten([node.get_params().values() for node in nodes])

step_count = sharedX(0, name='step_count')
last_lstm_1 = np.zeros((batch_size, lstm_1_dim*2), dtype=theano.config.floatX)
last_lstm_2 = np.zeros((batch_size, lstm_2_dim*2), dtype=theano.config.floatX)
last_lstm_3 = np.zeros((batch_size, lstm_3_dim*2), dtype=theano.config.floatX)
lstm_1_tm1 = sharedX(last_lstm_1, name='lstm_1_tm1')
lstm_2_tm1 = sharedX(last_lstm_2, name='lstm_2_tm1')
lstm_3_tm1 = sharedX(last_lstm_3, name='lstm_3_tm1')
update_list = [step_count, lstm_1_tm1, lstm_2_tm1, lstm_3_tm1]

step_count = T.switch(T.le(step_count, reset_freq), step_count + 1, 0)
s_1_0 = T.switch(T.or_(T.cast(T.eq(step_count, 0), 'int32'),
                       T.cast(T.eq(T.sum(lstm_1_tm1), 0.), 'int32')),
                 lstm_1.get_init_state(batch_size), lstm_1_tm1)
s_2_0 = T.switch(T.or_(T.cast(T.eq(step_count, 0), 'int32'),
                       T.cast(T.eq(T.sum(lstm_2_tm1), 0.), 'int32')),
                 lstm_2.get_init_state(batch_size), lstm_2_tm1)
s_3_0 = T.switch(T.or_(T.cast(T.eq(step_count, 0), 'int32'),
                       T.cast(T.eq(T.sum(lstm_3_tm1), 0.), 'int32')),
                 lstm_3.get_init_state(batch_size), lstm_3_tm1)

x_shape = x.shape
x_in = x.reshape((x_shape[0]*x_shape[1], -1))


def inner_fn(x_t, s_1_tm1, s_2_tm1, s_3_tm1):

    theta_1_t = theta_1.fprop([s_3_tm1])
    theta_2_t = theta_2.fprop([theta_1_t])
    theta_3_t = theta_3.fprop([theta_2_t])
    theta_4_t = theta_4.fprop([theta_3_t])
    theta_b_t = theta_b.fprop([theta_4_t])
    theta_c_t = theta_c.fprop([theta_4_t])
    theta_sig_t = theta_sig.fprop([theta_4_t])

    v_t, h_t = grbm.fprop([x_t, theta_b_t, theta_c_t, theta_sig_t]) 

    x_1_t = x_1.fprop([v_t])
    x_2_t = x_2.fprop([x_1_t])
    x_3_t = x_3.fprop([x_2_t])
    x_4_t = x_4.fprop([x_3_t])
 
    z_1_t = z_1.fprop([h_t])
    z_2_t = z_2.fprop([z_1_t])
    z_3_t = z_3.fprop([z_2_t])
    z_4_t = z_4.fprop([z_3_t])

    s_1_t = lstm_1.fprop([[x_4_t, z_4_t], [s_1_tm1]])
    s_2_t = lstm_2.fprop([[s_1_t], [s_2_tm1]])
    s_3_t = lstm_3.fprop([[s_2_t], [s_3_tm1]])

    return s_1_t, s_2_t, s_3_t, theta_b_t, theta_c_t, theta_sig_t, v_t

((s_1_t, s_2_t, s_3_t, theta_b_t, theta_c_t, theta_sig_t, v_t), updates) = theano.scan(fn=inner_fn,
                             sequences=[x],
                             outputs_info=[s_1_0, s_2_0, s_3_0, None, None, None, None])

for k, v in updates.iteritems():
    k.default_update = v

lstm_1_tm1 = s_1_t[-1]
lstm_2_tm1 = s_2_t[-1]
lstm_3_tm1 = s_3_t[-1]
theta_b_in = theta_b_t.reshape((x_shape[0]*x_shape[1], -1))
theta_c_in = theta_c_t.reshape((x_shape[0]*x_shape[1], -1))
theta_sig_in = theta_sig.fprop()
v_in = v_t.reshape((x_shape[0]*x_shape[1], -1))

W = grbm.params['W_x__grbm']
free_energy = grbm_free_energy(x_in, W, [theta_b_in, theta_c_in, theta_sig_in]) -\
              grbm_free_energy(v_in, W, [theta_b_in, theta_c_in, theta_sig_in])
free_energy = free_energy.mean()
free_energy.name = 'grbm_free_energy'
recon_err = T.sqr(x_in - v_in).mean()
recon_err.name = 'reconstruction_error'

max_x = x.max()
mean_x = x.mean()
min_x = x.min()
max_x.name = 'max_x'
mean_x.name = 'mean_x'
min_x.name = 'min_x'

max_theta_mu = v_in.max()
mean_theta_mu = v_in.mean()
min_theta_mu = v_in.min()
max_theta_mu.name = 'max_theta_mu'
mean_theta_mu.name = 'mean_theta_mu'
min_theta_mu.name = 'min_theta_mu'

max_theta_sig = theta_sig_in.max()
mean_theta_sig = theta_sig_in.mean()
min_theta_sig = theta_sig_in.min()
max_theta_sig.name = 'max_theta_sig'
mean_theta_sig.name = 'mean_theta_sig'
min_theta_sig.name = 'min_theta_sig'

model.inputs = [x]
model._params = params
model.nodes = nodes
model.set_updates(update_list)

lr_scalers = OrderedDict()
for node in nodes:
    if node.lr_scaler is not None:
        for key in node.params.keys():
            lr_scalers[key] = node.lr_scaler

optimizer = Adam(
    lr=lr,
    lr_scalers=lr_scalers
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(4),
    Monitoring(freq=100,
               ddout=[free_energy, recon_err,
                      max_theta_sig, mean_theta_sig, min_theta_sig,
                      max_x, mean_x, min_x,
                      max_theta_mu, mean_theta_mu, min_theta_mu],
               data=[Iterator(valid_data, batch_size, start=2040064, end=2070064)]), #112640 is 5%
    Picklize(freq=1000, force_save_freq=10000, path=save_path),
    EarlyStopping(freq=1000, force_save_freq=10000, path=save_path),
    WeightNorm()
]

mainloop = Training(
    name='lstm_grbm',
    data=Iterator(train_data, batch_size, start=0, end=2040064),
    model=model,
    optimizer=optimizer,
    cost=free_energy,
    outputs=[free_energy],
    extension=extension
)
mainloop.run()
