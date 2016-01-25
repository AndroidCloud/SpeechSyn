import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.data import Iterator
from cle.cle.cost import GMM
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

from sk.datasets.onomatopoeia import Onomatopoeia


#data_path = '/raid/chungjun/data/ubisoft/onomatopoeia/'
#save_path = '/raid/chungjun/repos/sk/cle/models/nips2015/onomatopoeia/pkl/'
data_path = '/home/junyoung/data/ubisoft/onomatopoeia/'
save_path = '/home/junyoung/repos/sk/cle/models/nips2015/onomatopoeia/pkl/'

batch_size = 64
frame_size = 200
latent_size = 200
main_lstm_dim = 2000
q_z_dim = 300
p_z_dim = 300
p_x_dim = 400
x2s_dim = 400
z2s_dim = 300
k = 20
target_size = frame_size * k
lr = 3e-4
debug = 0

model = Model()
train_data = Onomatopoeia(name='train',
                          path=data_path,
                          frame_size=frame_size)

X_mean = train_data.X_mean
X_std = train_data.X_std

valid_data = Onomatopoeia(name='valid',
                          path=data_path,
                          frame_size=frame_size,
                          X_mean=X_mean,
                          X_std=X_std)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')
#init_b_sig = InitCell('const', mean=0.6)
init_b_sig = InitCell('const')

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

prior_1 = FullyConnectedLayer(name='prior_1',
                              parent=['s_tm1'],
                              parent_dim=[main_lstm_dim],
                              nout=p_z_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

prior_2 = FullyConnectedLayer(name='prior_2',
                              parent=['prior_1'],
                              parent_dim=[p_z_dim],
                              nout=p_z_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

prior_3 = FullyConnectedLayer(name='prior_3',
                              parent=['prior_2'],
                              parent_dim=[p_z_dim],
                              nout=p_z_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

prior_4 = FullyConnectedLayer(name='prior_4',
                              parent=['prior_3'],
                              parent_dim=[p_z_dim],
                              nout=p_z_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

prior_mu = FullyConnectedLayer(name='prior_mu',
                               parent=['prior_4'],
                               parent_dim=[p_z_dim],
                               nout=latent_size,
                               unit='linear',
                               init_W=init_W,
                               init_b=init_b)

prior_sig = FullyConnectedLayer(name='prior_sig',
                                parent=['prior_4'],
                                parent_dim=[p_z_dim],
                                nout=latent_size,
                                unit='softplus',
                                cons=1e-4,
                                init_W=init_W,
                                init_b=init_b_sig)

kl = PriorLayer(name='kl',
                parent=['phi_mu', 'phi_sig', 'prior_mu', 'prior_sig'],
                parent_dim=[latent_size, latent_size, latent_size, latent_size],
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

coeff = FullyConnectedLayer(name='coeff',
                            parent=['theta_4'],
                            parent_dim=[p_x_dim],
                            nout=k,
                            unit='softmax',
                            init_W=init_W,
                            init_b=init_b)

nodes = [main_lstm, prior, kl,
         x_1, x_2, x_3, x_4,
         z_1, z_2, z_3, z_4,
         phi_1, phi_2, phi_3, phi_4, phi_mu, phi_sig,
         prior_1, prior_2, prior_3, prior_4, prior_mu, prior_sig,
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

    phi_1_t = phi_1.fprop([x_t, s_tm1])
    phi_2_t = phi_2.fprop([phi_1_t])
    phi_3_t = phi_3.fprop([phi_2_t])
    phi_4_t = phi_4.fprop([phi_3_t])
    phi_mu_t = phi_mu.fprop([phi_4_t])
    phi_sig_t = phi_sig.fprop([phi_4_t])

    prior_1_t = prior_1.fprop([s_tm1])
    prior_2_t = prior_2.fprop([prior_1_t])
    prior_3_t = prior_3.fprop([prior_2_t])
    prior_4_t = prior_4.fprop([prior_3_t])
    prior_mu_t = prior_mu.fprop([prior_4_t])
    prior_sig_t = prior_sig.fprop([prior_4_t])

    z_t = prior.fprop([phi_mu_t, phi_sig_t])

    z_1_t = z_1.fprop([z_t])
    z_2_t = z_2.fprop([z_1_t])
    z_3_t = z_3.fprop([z_2_t])
    z_4_t = z_4.fprop([z_3_t])

    s_t = main_lstm.fprop([[x_t, z_4_t], [s_tm1]])

    return s_t, phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, z_4_t

((s_t, phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, z_4_t), updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x_4_in],
                outputs_info=[s_0, None, None, None, None, None])

for k, v in updates.iteritems():
    k.default_update = v

s_t = s_t[:-1]
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
coeff_in = coeff.fprop([theta_4_in])

z_shape = phi_mu_t.shape
phi_mu_in = phi_mu_t.reshape((z_shape[0]*z_shape[1], -1))
phi_sig_in = phi_sig_t.reshape((z_shape[0]*z_shape[1], -1))
prior_mu_in = prior_mu_t.reshape((z_shape[0]*z_shape[1], -1))
prior_sig_in = prior_sig_t.reshape((z_shape[0]*z_shape[1], -1))
kl_in = kl.fprop([phi_mu_in, phi_sig_in, prior_mu_in, prior_sig_in])
kl_t = kl_in.reshape((z_shape[0], z_shape[1]))

recon = GMM(x_in, theta_mu_in, theta_sig_in, coeff_in)
recon = recon.reshape((x_shape[0], x_shape[1]))
recon = recon * x_mask
kl_t = kl_t * x_mask
recon_term = recon.sum(axis=0).mean()
kl_term = kl_t.sum(axis=0).mean()
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

coeff_max = coeff_in.max()
coeff_min = coeff_in.min()
coeff_mean_max = coeff_in.mean(axis=0).max()
coeff_mean_min = coeff_in.mean(axis=0).min()
coeff_max.name = 'coeff_max'
coeff_min.name = 'coeff_min'
coeff_mean_max.name = 'coeff_mean_max'
coeff_mean_min.name = 'coeff_mean_min'

max_phi_sig = phi_sig_t.max()
mean_phi_sig = phi_sig_t.mean()
min_phi_sig = phi_sig_t.min()
max_phi_sig.name = 'max_phi_sig'
mean_phi_sig.name = 'mean_phi_sig'
min_phi_sig.name = 'min_phi_sig'

max_prior_sig = prior_sig_t.max()
mean_prior_sig = prior_sig_t.mean()
min_prior_sig = prior_sig_t.min()
max_prior_sig.name = 'max_prior_sig'
mean_prior_sig.name = 'mean_prior_sig'
min_prior_sig.name = 'min_prior_sig'

model.inputs = [x, x_mask]
model._params = params
model.nodes = nodes

optimizer = Adam(
    lr=lr
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(200),
    Monitoring(freq=94,
               ddout=[nll_lower_bound, recon_term, kl_term, kl_ratio,
                      max_phi_sig, mean_phi_sig, min_phi_sig,
                      max_prior_sig, mean_prior_sig, min_prior_sig,
                      max_theta_sig, mean_theta_sig, min_theta_sig,
                      max_x, mean_x, min_x,
                      max_theta_mu, mean_theta_mu, min_theta_mu,
                      coeff_max, coeff_min, coeff_mean_max, coeff_mean_min],
               data=[Iterator(valid_data, batch_size)]),
    Picklize(freq=94, path=save_path),
    EarlyStopping(freq=94, path=save_path),
    WeightNorm()
]

mainloop = Training(
    name='m3_3',
    data=Iterator(train_data, batch_size),
    model=model,
    optimizer=optimizer,
    cost=nll_lower_bound,
    outputs=[nll_lower_bound],
    extension=extension
)
mainloop.run()
