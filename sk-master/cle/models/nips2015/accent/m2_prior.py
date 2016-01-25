import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.data import Iterator
from cle.cle.cost import Gaussian
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
from cle.cle.utils import flatten, sharedX

from sk.datasets.accent import Accent_h5


data_path = '/raid/chungjun/data/accent/accent_speech/'
save_path = '/raid/chungjun/repos/sk/cle/models/nips2015/accent/pkl/'
#data_path = '/home/junyoung/data/accent/accent_speech/'
#save_path = '/home/junyoung/repos/sk/cle/models/nips2015/accent/pkl/'

reset_freq = 4
batch_size = 128
frame_size = 200
latent_size = 200
main_lstm_dim = 2000
q_z_dim = 400
p_z_dim = 400
p_x_dim = 400
x2s_dim = 400
z2s_dim = 400
target_size = frame_size
lr = 1e-3
debug = 0

file_name = 'accent_tbptt'
normal_params = np.load(data_path + file_name + '_normal.npz')
X_mean = normal_params['X_mean']
X_std = normal_params['X_std']

model = Model()
train_data = Accent_h5(name='train',
                       path=data_path,
                       frame_size=frame_size,
                       X_mean=X_mean,
                       X_std=X_std)

valid_data = Accent_h5(name='valid',
                       path=data_path,
                       frame_size=frame_size,
                       X_mean=X_mean,
                       X_std=X_std)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')
init_b_sig = InitCell('const', mean=0.6)

x = train_data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((15, batch_size, frame_size), dtype=np.float32)

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

main_lstm = LSTM(name='main_lstm',
                 parent=['x_4', 'z_4'],
                 parent_dim=[x2s_dim, z2s_dim],
                 batch_size=batch_size,
                 nout=main_lstm_dim,
                 unit='tanh',
                 init_W=init_W,
                 init_U=init_U,
                 init_b=init_b)

phi_1 = FullyConnectedLayer(name='phi_1',
                            parent=['x_4', 's_tm1'],
                            parent_dim=[x2s_dim, main_lstm_dim],
                            nout=q_z_dim,
                            unit='relu',
                            init_W=init_W,
                            init_b=init_b)

phi_2 = FullyConnectedLayer(name='phi_2',
                            parent=['phi_1'],
                            parent_dim=[q_z_dim],
                            nout=q_z_dim,
                            unit='relu',
                            init_W=init_W,
                            init_b=init_b)

phi_3 = FullyConnectedLayer(name='phi_3',
                            parent=['phi_2'],
                            parent_dim=[q_z_dim],
                            nout=q_z_dim,
                            unit='relu',
                            init_W=init_W,
                            init_b=init_b)

phi_4 = FullyConnectedLayer(name='phi_4',
                            parent=['phi_3'],
                            parent_dim=[q_z_dim],
                            nout=q_z_dim,
                            unit='relu',
                            init_W=init_W,
                            init_b=init_b)

phi_mu = FullyConnectedLayer(name='phi_mu',
                             parent=['phi_4'],
                             parent_dim=[q_z_dim],
                             nout=latent_size,
                             unit='linear',
                             init_W=init_W,
                             init_b=init_b)

phi_sig = FullyConnectedLayer(name='phi_sig',
                              parent=['phi_4'],
                              parent_dim=[q_z_dim],
                              nout=latent_size,
                              unit='softplus',
                              cons=1e-4,
                              init_W=init_W,
                              init_b=init_b_sig)

prior = PriorLayer(name='prior',
                   parent=['phi_mu', 'phi_sig'],
                   parent_dim=[latent_size, latent_size],
                   use_sample=1,
                   num_sample=1,
                   nout=latent_size)

kl = PriorLayer(name='kl',
                parent=['phi_mu', 'phi_sig'],
                parent_dim=[latent_size, latent_size],
                use_sample=0,
                nout=latent_size)

theta_1 = FullyConnectedLayer(name='theta_1',
                              parent=['z_4', 's_tm1'],
                              parent_dim=[z2s_dim, main_lstm_dim],
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

nodes = [main_lstm, prior, kl,
         x_1, x_2, x_3, x_4,
         z_1, z_2, z_3, z_4,
         phi_1, phi_2, phi_3, phi_4, phi_mu, phi_sig,
         theta_1, theta_2, theta_3, theta_4, theta_mu, theta_sig]

for node in nodes:
    node.initialize()

params = flatten([node.get_params().values() for node in nodes])

step_count = sharedX(0, name='step_count')
last_main_lstm = np.zeros((batch_size, main_lstm_dim*2), dtype=theano.config.floatX)
main_lstm_tm1 = sharedX(last_main_lstm, name='main_lstm_tm1')
update_list = [step_count, main_lstm_tm1]

step_count = T.switch(T.le(step_count, reset_freq), step_count + 1, 0)
s_0 = T.switch(T.or_(T.cast(T.eq(step_count, 0), 'int32'),
                     T.cast(T.eq(T.sum(main_lstm_tm1), 0.), 'int32')),
               main_lstm.get_init_state(batch_size), main_lstm_tm1)

x_shape = x.shape
x_in = x.reshape((x_shape[0]*x_shape[1], -1))
x_1_in = x_1.fprop([x_in])
x_2_in = x_2.fprop([x_1_in])
x_3_in = x_3.fprop([x_2_in])
x_4_in = x_4.fprop([x_3_in])
x_4_in = x_4_in.reshape((x_shape[0], x_shape[1], -1))


def inner_fn(x_t, s_tm1):

    phi_1_t = phi_1.fprop([x_t, s_tm1])
    phi_2_t = phi_2.fprop([phi_1_t])
    phi_3_t = phi_3.fprop([phi_2_t])
    phi_4_t = phi_4.fprop([phi_3_t])
    phi_mu_t = phi_mu.fprop([phi_4_t])
    phi_sig_t = phi_sig.fprop([phi_4_t])

    z_t = prior.fprop([phi_mu_t, phi_sig_t])

    z_1_t = z_1.fprop([z_t])
    z_2_t = z_2.fprop([z_1_t])
    z_3_t = z_3.fprop([z_2_t])
    z_4_t = z_4.fprop([z_3_t])

    s_t = main_lstm.fprop([[x_t, z_4_t], [s_tm1]])

    return s_t, phi_mu_t, phi_sig_t, z_4_t

((s_t, phi_mu_t, phi_sig_t, z_4_t), updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x_4_in],
                outputs_info=[s_0, None, None, None])

for k, v in updates.iteritems():
    k.default_update = v

s_t, main_lstm_tm1 = s_t[:-1], s_t[-1]
s_shape = s_t.shape
s_in = T.concatenate([s_0, s_t.reshape((s_shape[0]*s_shape[1], -1))], axis=0)
z_4_shape = z_4_t.shape
z_4_in = z_4_t.reshape((z_4_shape[0]*z_4_shape[1], -1))
theta_1_in = theta_1.fprop([z_4_in, s_in])
theta_2_in = theta_2.fprop([theta_1_in])
theta_3_in = theta_3.fprop([theta_2_in])
theta_4_in = theta_4.fprop([theta_3_in])
theta_mu_in = theta_mu.fprop([theta_4_in])
theta_sig_in = theta_sig.fprop([theta_4_in])

z_shape = phi_mu_t.shape
phi_mu_in = phi_mu_t.reshape((z_shape[0]*z_shape[1], -1))
phi_sig_in = phi_sig_t.reshape((z_shape[0]*z_shape[1], -1))
kl_in = kl.fprop([phi_mu_in, phi_sig_in])
kl_t = kl_in.reshape((z_shape[0], z_shape[1]))

recon = Gaussian(x_in, theta_mu_in, theta_sig_in)
recon = recon.reshape((x_shape[0], x_shape[1]))
recon_term = recon.mean()
kl_term = kl_t.mean()
nll_lower_bound = recon_term + kl_term
nll_lower_bound.name = 'nll_lower_bound'
recon_term.name = 'recon_term'
kl_term.name = 'kl_term'

kl_ratio = kl_term / T.abs_(recon_term)
kl_ratio.name = 'kl_term proportion'

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

max_phi_sig = phi_sig_t.max()
mean_phi_sig = phi_sig_t.mean()
min_phi_sig = phi_sig_t.min()
max_phi_sig.name = 'max_phi_sig'
mean_phi_sig.name = 'mean_phi_sig'
min_phi_sig.name = 'min_phi_sig'

model.inputs = [x]
model._params = params
model.nodes = nodes
model.set_updates(update_list)

optimizer = Adam(
    lr=lr
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(40),
    Monitoring(freq=200,
               ddout=[nll_lower_bound, recon_term, kl_term, kl_ratio,
                      max_phi_sig, mean_phi_sig, min_phi_sig,
                      max_theta_sig, mean_theta_sig, min_theta_sig,
                      max_x, mean_x, min_x,
                      max_theta_mu, mean_theta_mu, min_theta_mu],
               data=[Iterator(valid_data, batch_size, start=99968, end=113536)]),
    Picklize(freq=200, path=save_path),
    EarlyStopping(freq=200, path=save_path),
    WeightNorm()
]

mainloop = Training(
    name='m2_prior_2',
    data=Iterator(train_data, batch_size, start=0, end=99968),
    model=model,
    optimizer=optimizer,
    cost=nll_lower_bound,
    outputs=[nll_lower_bound],
    extension=extension
)
mainloop.run()
