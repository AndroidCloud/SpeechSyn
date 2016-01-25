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
from cle.cle.utils.compat import OrderedDict

from sk.datasets.onomatopoeia import Onomatopoeia


#data_path = '/raid/chungjun/data/ubisoft/onomatopoeia/'
#save_path = '/raid/chungjun/repos/sk/cle/models/nips2015/onomatopoeia/pkl/'
data_path = '/home/junyoung/data/ubisoft/onomatopoeia/'
save_path = '/home/junyoung/repos/sk/cle/models/nips2015/onomatopoeia/pkl/'

batch_size = 64
frame_size = 200
latent_size = 200
encoder_dim = 1400
decoder_dim = 1400
q_z_dim = 400
p_x_dim = 400
x2s_dim = 400
z2s_dim = 400
target_size = frame_size
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
init_b_sig = InitCell('const', mean=0.6)

x, x_mask = train_data.theano_vars()
if debug:
    x.tag.test_value = np.zeros((15, batch_size, frame_size), dtype=np.float32)
    temp = np.ones((15, batch_size), dtype=np.float32)
    temp[:, -2:] = 0.
    x_mask.tag.test_value = temp
x_tm1 = T.concatenate([T.zeros((1, x.shape[1], x.shape[2])), x[:-1]], axis=0)
x_tm1.name = 'x_tm1'

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

encoder = LSTM(name='encoder',
               parent=['x_4'],
               parent_dim=[x2s_dim],
               batch_size=batch_size,
               nout=encoder_dim,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

decoder = LSTM(name='decoder',
               parent=['x_4_tm1', 'z_4'],
               parent_dim=[x2s_dim, z2s_dim],
               batch_size=batch_size,
               nout=decoder_dim,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

phi_1 = FullyConnectedLayer(name='phi_1',
                            parent=['enc_t'],
                            parent_dim=[encoder_dim],
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
                              parent=['dec_t'],
                              parent_dim=[decoder_dim],
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

nodes = [encoder, decoder, prior, kl,
         x_1, x_2, x_3, x_4,
         z_1, z_2, z_3, z_4,
         phi_1, phi_2, phi_3, phi_4, phi_mu, phi_sig,
         theta_1, theta_2, theta_3, theta_4, theta_mu, theta_sig]

for node in nodes:
    node.initialize()

params = flatten([node.get_params().values() for node in nodes])

enc_0 = encoder.get_init_state(batch_size)
dec_0 = decoder.get_init_state(batch_size)

x_shape = x.shape
x_in = x.reshape((x_shape[0]*x_shape[1], -1))
x_1_in = x_1.fprop([x_in])
x_2_in = x_2.fprop([x_1_in])
x_3_in = x_3.fprop([x_2_in])
x_4_in = x_4.fprop([x_3_in])
x_4_in = x_4_in.reshape((x_shape[0], x_shape[1], -1))

x_tm1_shape = x_tm1.shape
x_in_tm1 = x_tm1.reshape((x_tm1_shape[0]*x_tm1_shape[1], -1))
x_1_in_tm1 = x_1.fprop([x_in_tm1])
x_2_in_tm1 = x_2.fprop([x_1_in_tm1])
x_3_in_tm1 = x_3.fprop([x_2_in_tm1])
x_4_in_tm1 = x_4.fprop([x_3_in_tm1])
x_4_in_tm1 = x_4_in_tm1.reshape((x_shape[0], x_shape[1], -1))


def inner_fn(x_t, x_tm1, enc_tm1, dec_tm1):

    enc_t = encoder.fprop([[x_t], [enc_tm1]])

    phi_1_t = phi_1.fprop([enc_t])
    phi_2_t = phi_2.fprop([phi_1_t])
    phi_3_t = phi_3.fprop([phi_2_t])
    phi_4_t = phi_4.fprop([phi_3_t])
    phi_mu_t = phi_mu.fprop([phi_4_t])
    phi_sig_t = phi_sig.fprop([phi_4_t])

    z_t = prior.fprop([phi_mu_t, phi_sig_t])
    kl_t = kl.fprop([phi_mu_t, phi_sig_t])

    z_1_t = z_1.fprop([z_t])
    z_2_t = z_2.fprop([z_1_t])
    z_3_t = z_3.fprop([z_2_t])
    z_4_t = z_4.fprop([z_3_t])

    dec_t = decoder.fprop([[x_tm1, z_4_t], [dec_tm1]])

    theta_1_t = theta_1.fprop([dec_t])
    theta_2_t = theta_2.fprop([theta_1_t])
    theta_3_t = theta_3.fprop([theta_2_t])
    theta_4_t = theta_4.fprop([theta_3_t])
    theta_mu_t = theta_mu.fprop([theta_4_t])
    theta_sig_t = theta_sig.fprop([theta_4_t])

    return enc_t, dec_t, phi_mu_t, phi_sig_t, kl_t, theta_mu_t, theta_sig_t

((enc_t, dec_t, phi_mu_t, phi_sig_t, kl_t, theta_mu_t, theta_sig_t), updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x_4_in, x_4_in_tm1],
                outputs_info=[enc_0, dec_0, None, None, None, None, None])

for k, v in updates.iteritems():
    k.default_update = v

reshaped_x = x.reshape((x.shape[0]*x.shape[1], -1))
reshaped_theta_mu = theta_mu_t.reshape((theta_mu_t.shape[0]*theta_mu_t.shape[1], -1))
reshaped_theta_sig = theta_sig_t.reshape((theta_sig_t.shape[0]*theta_sig_t.shape[1], -1))

recon = Gaussian(reshaped_x, reshaped_theta_mu, reshaped_theta_sig)
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

max_phi_sig = phi_sig_t.max()
mean_phi_sig = phi_sig_t.mean()
min_phi_sig = phi_sig_t.min()
max_phi_sig.name = 'max_phi_sig'
mean_phi_sig.name = 'mean_phi_sig'
min_phi_sig.name = 'min_phi_sig'

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
                      max_theta_sig, mean_theta_sig, min_theta_sig,
                      max_x, mean_x, min_x,
                      max_theta_mu, mean_theta_mu, min_theta_mu],
               data=[Iterator(valid_data, batch_size)]),
    Picklize(freq=94, path=save_path),
    EarlyStopping(freq=94, path=save_path),
    WeightNorm()
]

mainloop = Training(
    name='storn0_2',
    data=Iterator(train_data, batch_size),
    model=model,
    optimizer=optimizer,
    cost=nll_lower_bound,
    outputs=[nll_lower_bound],
    extension=extension
)
mainloop.run()
