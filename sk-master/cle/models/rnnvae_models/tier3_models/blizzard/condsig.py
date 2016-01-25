import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.data import Iterator
from cle.cle.cost import Gaussian, GMM
from cle.cle.models import Model
from cle.cle.layers import InitCell, RealVectorLayer
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
from cle.cle.utils.compat import OrderedDict
from sk.datasets.blizzard import Blizzard


data_path = '/home/chungjun/data/blizzard/segmented/'
save_path = '/raid/chungjun/repos/sk/saved/rnnvae/tier3_models/blizzard/'

batch_size = 100
inpsz = 200
latsz = 200
k = 10
outsz = inpsz * k
shared_nout = 1500
lr = 0.001
debug = 0

model = Model()
trdata = Blizzard(name='train',
                  path=data_path,
                  inpsz=inpsz)

init_W = InitCell('rand')
init_U = InitCell('ortho')
init_b = InitCell('zeros')
init_b_sig = InitCell('const', mean=0.6)

x, mask = trdata.theano_vars()
if debug:
    x.tag.test_value = np.zeros((15, batch_size, inpsz), dtype=np.float32)
    temp = np.ones((15, batch_size), dtype=np.float32)
    temp [:, -2:] = 0.
    mask.tag.test_value = temp

coder = LSTM(name='coder',
             parent=['x_t', 'z_t'],
             parent_dim=[inpsz, latsz],
             batch_size=batch_size,
             nout=shared_nout,
             unit='tanh',
             init_W=init_W,
             init_U=init_U,
             init_b=init_b)

phi_mu = FullyConnectedLayer(name='phi_mu',
                             parent=['x_t', 's_tm1'],
                             parent_dim=[inpsz, shared_nout],
                             nout=latsz,
                             unit='linear',
                             init_W=init_W,
                             init_b=init_b)

phi_sig = FullyConnectedLayer(name='phi_sig',
                              parent=['x_t', 's_tm1'],
                              parent_dim=[inpsz, shared_nout],
                              nout=latsz,
                              unit='softplus',
                              cons=1e-4,
                              init_W=init_W,
                              init_b=init_b_sig)

prior = PriorLayer(name='prior',
                   parent=['phi_mu', 'phi_sig'],
                   parent_dim=[latsz, latsz],
                   use_sample=1,
                   num_sample=1,
                   nout=latsz)

prior_mu = FullyConnectedLayer(name='prior_mu',
                               parent=['s_tm1'],
                               parent_dim=[shared_nout],
                               nout=latsz,
                               unit='linear',
                               init_W=init_W,
                               init_b=init_b)

prior_sig = FullyConnectedLayer(name='prior_sig',
                                parent=['s_tm1'],
                                parent_dim=[shared_nout],
                                nout=latsz,
                                unit='softplus',
                                cons=1e-4,
                                init_W=init_W,
                                init_b=init_b_sig)

kl = PriorLayer(name='kl',
                parent=['phi_mu', 'phi_sig', 'prior_mu', 'prior_sig'],
                parent_dim=[latsz, latsz, latsz, latsz],
                use_sample=0,
                nout=latsz)

theta_mu = FullyConnectedLayer(name='theta_mu',
                               parent=['s_tm1'],
                               parent_dim=[shared_nout],
                               nout=outsz,
                               unit='linear',
                               init_W=init_W,
                               init_b=init_b)

theta_sig = FullyConnectedLayer(name='theta_sig',
                                parent=['s_tm1'],
                                parent_dim=[shared_nout],
                                nout=outsz,
                                unit='softplus',
                                cons=1e-4,
                                init_W=init_W,
                                init_b=init_b_sig)

coeff = FullyConnectedLayer(name='coeff',
                            parent=['s_tm1'],
                            parent_dim=[shared_nout],
                            nout=k,
                            unit='softmax',
                            init_W=init_W,
                            init_b=init_b)

nodes = [coder, phi_mu, phi_sig, prior_mu, prior_sig, prior, kl,
         theta_mu, theta_sig, coeff]

for node in nodes:
    node.initialize()

params = flatten([node.get_params().values() for node in nodes])

def inner_fn(x_t, s_tm1):

    phi_mu_t = phi_mu.fprop([x_t, s_tm1])
    phi_sig_t = phi_sig.fprop([x_t, s_tm1])

    prior_mu_t = prior_mu.fprop([s_tm1])
    prior_sig_t = prior_sig.fprop([s_tm1])

    z_t = prior.fprop([phi_mu_t, phi_sig_t])
    kl_t = kl.fprop([phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t])

    theta_mu_t = theta_mu.fprop([s_tm1])
    theta_sig_t = theta_sig.fprop([s_tm1])
    coeff_t = coeff.fprop([s_tm1])

    s_t = coder.fprop([[x_t, z_t], [s_tm1]])

    return s_t, kl_t, prior_sig_t, phi_sig_t, theta_mu_t, theta_sig_t, coeff_t

((s_t, kl_t, prior_sig_t, phi_sig_t, theta_mu_t, theta_sig_t, coeff_t),
 updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x],
                outputs_info=[coder.get_init_state(),
                              None, None, None, None, None, None])

for k, v in updates.iteritems():
    k.default_update = v

reshaped_x = x.reshape((x.shape[0]*x.shape[1], -1))
reshaped_theta_mu = theta_mu_t.reshape((theta_mu_t.shape[0]*theta_mu_t.shape[1], -1))
reshaped_theta_sig = theta_sig_t.reshape((theta_sig_t.shape[0]*theta_sig_t.shape[1], -1))
reshaped_coeff = coeff_t.reshape((coeff_t.shape[0]*coeff_t.shape[1], -1))
reshaped_mask = mask.flatten()

kl_term = kl_t.reshape((kl_t.shape[0]*kl_t.shape[1], -1))
recon_term = GMM(reshaped_x, reshaped_theta_mu, reshaped_theta_sig, reshaped_coeff)
# Apply mask
kl_term = kl_term[reshaped_mask.nonzero()].mean()
recon_term = recon_term[reshaped_mask.nonzero()].mean()
cost = recon_term + kl_term
cost.name = 'cost'
recon_term.name = 'recon_term'
kl_term.name = 'kl_term'

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

max_prior_sig = prior_sig_t.max()
mean_prior_sig = prior_sig_t.mean()
min_prior_sig = prior_sig_t.min()
max_prior_sig.name = 'max_prior_sig'
mean_prior_sig.name = 'mean_prior_sig'
min_prior_sig.name = 'min_prior_sig'

W_x_t_s_t = coder.params['W_x_t__coder'][:, :shared_nout]
col_norm_W_x_t_s_t = T.sqrt((W_x_t_s_t**2).sum(axis=0))
max_col_norm_W_x_t_s_t = col_norm_W_x_t_s_t.max()
mean_col_norm_W_x_t_s_t = col_norm_W_x_t_s_t.mean()
min_col_norm_W_x_t_s_t = col_norm_W_x_t_s_t.min()
max_col_norm_W_x_t_s_t.name = 'max_col_norm_W_x_t_s_t'
mean_col_norm_W_x_t_s_t.name = 'mean_col_norm_W_x_t_s_t'
min_col_norm_W_x_t_s_t.name = 'min_col_norm_W_x_t_s_t'

W_z_t_s_t = coder.params['W_z_t__coder'][:, :shared_nout]
col_norm_W_z_t_s_t = T.sqrt((W_z_t_s_t**2).sum(axis=0))
max_col_norm_W_z_t_s_t = col_norm_W_z_t_s_t.max()
mean_col_norm_W_z_t_s_t = col_norm_W_z_t_s_t.mean()
min_col_norm_W_z_t_s_t = col_norm_W_z_t_s_t.min()
max_col_norm_W_z_t_s_t.name = 'max_col_norm_W_z_t_s_t'
mean_col_norm_W_z_t_s_t.name = 'mean_col_norm_W_z_t_s_t'
min_col_norm_W_z_t_s_t.name = 'min_col_norm_W_z_t_s_t'

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
    Monitoring(freq=100,
               ddout=[cost, recon_term, kl_term,
                      max_phi_sig, mean_phi_sig, min_phi_sig,
                      max_prior_sig, mean_prior_sig, min_prior_sig,
                      max_theta_sig, mean_theta_sig, min_theta_sig,
                      max_x, mean_x, min_x,
                      max_theta_mu, mean_theta_mu, min_theta_mu,
                      max_col_norm_W_x_t_s_t,
                      mean_col_norm_W_x_t_s_t,
                      min_col_norm_W_x_t_s_t,
                      max_col_norm_W_z_t_s_t,
                      mean_col_norm_W_z_t_s_t,
                      min_col_norm_W_z_t_s_t,
                      coeff_max, coeff_min, coeff_mean_max, coeff_mean_min],
               data=[Iterator(trdata, batch_size)]),
    Picklize(freq=1000,
             path=save_path),
    EarlyStopping(freq=500, path=save_path),
    WeightNorm()
]

mainloop = Training(
    name='condsig',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
