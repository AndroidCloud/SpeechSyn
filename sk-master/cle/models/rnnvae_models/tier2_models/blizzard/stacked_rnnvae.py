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
from sk.datasets.blizzard import Blizzard


data_path = '/home/chungjun/data/blizzard/segmented/'
save_path = '/raid/chungjun/repos/sk/saved/vae/blizzard/'

batch_size = 100
num_sample = 1
inpsz = 200
#latsz_1 = 100
#latsz_2 = 50
latsz_1 = 200
latsz_2 = 100
lat_emb = 1000
out_emb = 1000
k = 30
outsz = inpsz * k
shared_nout_1 = 3000
shared_nout_2 = 3000
#lr = 0.0005
#lr = 0.0003
lr = 0.001
debug = 0

model = Model()
trdata = Blizzard(name='train',
                  path=data_path,
                  use_derivative=0,
                  use_log_space=0,
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

coder_1 = LSTM(name='coder_1',
               parent=['x_t', 'z_1_t', 'z_2_t'],
               parent_dim=[inpsz, latsz_1, latsz_2],
               batch_size=batch_size,
               nout=shared_nout_1,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

phi_emb_1 = FullyConnectedLayer(name='phi_emb_1',
                                parent=['x_t', 's_1_tm1', 's_2_tm1'],
                                parent_dim=[inpsz, shared_nout_1, shared_nout_2],
                                nout=lat_emb,
                                #unit='tanh',
                                unit='relu',
                                init_W=init_W,
                                init_b=init_b)

phi_mu_1 = FullyConnectedLayer(name='phi_mu_1',
                               parent=['phi_emb_1'],
                               parent_dim=[lat_emb],
                               nout=latsz_1,
                               unit='linear',
                               init_W=init_W,
                               init_b=init_b)

phi_sig_1 = FullyConnectedLayer(name='phi_sig_1',
                                parent=['phi_emb_1'],
                                parent_dim=[lat_emb],
                                nout=latsz_1,
                                unit='softplus',
                                cons=1e-4,
                                init_W=init_W,
                                init_b=init_b_sig)

prior_1 = PriorLayer(name='prior_1',
                     parent=['phi_mu_1', 'phi_sig_1'],
                     parent_dim=[latsz_1, latsz_1],
                     use_sample=1,
                     num_sample=num_sample,
                     nout=latsz_1)

prior_emb_1 = FullyConnectedLayer(name='prior_emb_1',
                                  parent=['s_1_tm1', 's_2_tm1', 'prior_2'],
                                  parent_dim=[shared_nout_1, shared_nout_2, latsz_2],
                                  nout=lat_emb,
                                  #unit='tanh',
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

prior_mu_1 = FullyConnectedLayer(name='prior_mu_1',
                                 parent=['prior_emb_1'],
                                 parent_dim=[lat_emb],
                                 nout=latsz_1,
                                 unit='linear',
                                 init_W=init_W,
                                 init_b=init_b)

prior_sig_1 = FullyConnectedLayer(name='prior_sig_1',
                                  parent=['prior_emb_1'],
                                  parent_dim=[lat_emb],
                                  nout=latsz_1,
                                  unit='softplus',
                                  cons=1e-4,
                                  init_W=init_W,
                                  init_b=init_b_sig)

kl_1 = PriorLayer(name='kl_1',
                  parent=['phi_mu_1', 'phi_sig_1', 'prior_mu_1', 'prior_sig_1'],
                  parent_dim=[latsz_1, latsz_1, latsz_1, latsz_1],
                  use_sample=0,
                  nout=latsz_1)

theta_emb = FullyConnectedLayer(name='theta_emb',
                                parent=['s_1_tm1', 'z_1_t', 'z_2_t'],
                                parent_dim=[shared_nout_1, latsz_1, latsz_2],
                                nout=out_emb,
                                #unit='tanh',
                                unit='relu',
                                init_W=init_W,
                                init_b=init_b)

theta_mu = FullyConnectedLayer(name='theta_mu',
                               parent=['theta_emb'],
                               parent_dim=[out_emb],
                               nout=outsz,
                               unit='linear',
                               init_W=init_W,
                               init_b=init_b)

theta_sig = FullyConnectedLayer(name='theta_sig',
                                parent=['theta_emb'],
                                parent_dim=[out_emb],
                                nout=outsz,
                                unit='softplus',
                                cons=1e-4,
                                init_W=init_W,
                                init_b=init_b_sig)

coeff = FullyConnectedLayer(name='coeff',
                            parent=['theta_emb'],
                            parent_dim=[out_emb],
                            nout=k,
                            unit='softmax',
                            init_W=init_W,
                            init_b=init_b)

coder_2 = LSTM(name='coder_2',
               parent=['z_1_t', 'z_2_t'],
               parent_dim=[latsz_1, latsz_2],
               batch_size=batch_size,
               nout=shared_nout_2,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

phi_emb_2 = FullyConnectedLayer(name='phi_emb_2',
                                parent=['z_1_t', 's_2_tm1'],
                                parent_dim=[latsz_1, shared_nout_2],
                                nout=lat_emb,
                                #unit='tanh',
                                unit='relu',
                                init_W=init_W,
                                init_b=init_b)

phi_mu_2 = FullyConnectedLayer(name='phi_mu_2',
                               parent=['phi_emb_2'],
                               parent_dim=[lat_emb],
                               nout=latsz_2,
                               unit='linear',
                               init_W=init_W,
                               init_b=init_b)

phi_sig_2 = FullyConnectedLayer(name='phi_sig_2',
                                parent=['phi_emb_2'],
                                parent_dim=[lat_emb],
                                nout=latsz_2,
                                unit='softplus',
                                cons=1e-4,
                                init_W=init_W,
                                init_b=init_b_sig)

prior_2 = PriorLayer(name='prior_2',
                     parent=['phi_mu_2', 'phi_sig_2'],
                     parent_dim=[latsz_2, latsz_2],
                     use_sample=1,
                     num_sample=num_sample,
                     nout=latsz_2)

prior_emb_2 = FullyConnectedLayer(name='prior_emb_2',
                                  parent=['s_2_tm1'],
                                  parent_dim=[shared_nout_2],
                                  nout=lat_emb,
                                  #unit='tanh',
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

prior_mu_2 = FullyConnectedLayer(name='prior_mu_2',
                                 parent=['prior_emb_2'],
                                 parent_dim=[lat_emb],
                                 nout=latsz_2,
                                 unit='linear',
                                 init_W=init_W,
                                 init_b=init_b)

prior_sig_2 = FullyConnectedLayer(name='prior_sig_2',
                                  parent=['prior_emb_2'],
                                  parent_dim=[lat_emb],
                                  nout=latsz_2,
                                  unit='softplus',
                                  cons=1e-4,
                                  init_W=init_W,
                                  init_b=init_b_sig)

kl_2 = PriorLayer(name='kl_2',
                  parent=['phi_mu_2', 'phi_sig_2', 'prior_mu_2', 'prior_sig_2'],
                  parent_dim=[latsz_2, latsz_2, latsz_2, latsz_2],
                  use_sample=0,
                  nout=latsz_2)

nodes = [coder_1, phi_emb_1, phi_mu_1, phi_sig_1, prior_1, prior_emb_1,
         prior_mu_1, prior_sig_1, kl_1, theta_emb, theta_mu, theta_sig,
         coeff, coder_2, phi_emb_2, phi_mu_2, phi_sig_2, prior_2,
         prior_emb_2, prior_mu_2, prior_sig_2, kl_2]

for node in nodes:
    node.initialize()

params = flatten([node.get_params().values() for node in nodes])

def inner_fn(x_t, s_1_tm1, s_2_tm1):

    phi_emb_1_t = phi_emb_1.fprop([x_t, s_1_tm1, s_2_tm1])
    phi_mu_1_t = phi_mu_1.fprop([phi_emb_1_t])
    phi_sig_1_t = phi_sig_1.fprop([phi_emb_1_t])

    z_1_t = prior_1.fprop([phi_mu_1_t, phi_sig_1_t])

    phi_emb_2_t = phi_emb_2.fprop([z_1_t, s_2_tm1])
    phi_mu_2_t = phi_mu_2.fprop([phi_emb_2_t])
    phi_sig_2_t = phi_sig_2.fprop([phi_emb_2_t])

    z_2_t = prior_2.fprop([phi_mu_2_t, phi_sig_2_t])

    prior_emb_1_t = prior_emb_1.fprop([s_1_tm1, s_2_tm1, z_2_t])
    prior_mu_1_t = prior_mu_1.fprop([prior_emb_1_t])
    prior_sig_1_t = prior_sig_1.fprop([prior_emb_1_t])

    prior_emb_2_t = prior_emb_2.fprop([s_2_tm1])
    prior_mu_2_t = prior_mu_2.fprop([prior_emb_2_t])
    prior_sig_2_t = prior_sig_2.fprop([prior_emb_2_t])

    kl_1_t = kl_1.fprop([phi_mu_1_t, phi_sig_1_t, prior_mu_1_t, prior_sig_1_t])
    kl_2_t = kl_2.fprop([phi_mu_2_t, phi_sig_2_t, prior_mu_2_t, prior_sig_2_t])

    theta_emb_t = theta_emb.fprop([s_1_tm1, z_1_t, z_2_t])
    theta_mu_t = theta_mu.fprop([theta_emb_t])
    theta_sig_t = theta_sig.fprop([theta_emb_t])
    coeff_t = coeff.fprop([theta_emb_t])

    s_1_t = coder_1.fprop([[x_t, z_1_t, z_2_t], [s_1_tm1]])
    s_2_t = coder_2.fprop([[z_1_t, z_2_t], [s_2_tm1]])

    marginal_ll = []
    for i in xrange(num_sample):
        z_2_is = prior_2.fprop([phi_mu_2_t, phi_sig_2_t])
        z_1_is = prior_1.fprop([phi_mu_1_t, phi_sig_1_t])
        theta_emb_t = theta_emb.fprop([s_1_tm1, z_1_is, z_2_is])
        theta_mu_t = theta_mu.fprop([theta_emb_t])
        theta_sig_t = theta_sig.fprop([theta_emb_t])
        coeff_t = coeff.fprop([theta_emb_t])
        w_1 = Gaussian(z_1_is, prior_mu_1_t, prior_sig_1_t) -\
              Gaussian(z_1_is, phi_mu_1_t, phi_sig_1_t)
        w_2 = Gaussian(z_2_is, prior_mu_2_t, prior_sig_2_t) -\
              Gaussian(z_2_is, phi_mu_2_t, phi_sig_2_t)
        marginal_ll.append(GMM(x_t, theta_mu_t, theta_sig_t, coeff_t) + w_1 + w_2)
    marginal_ll = T.concatenate(marginal_ll, axis=0).mean()

    return s_1_t, s_2_t, kl_1_t, kl_2_t, phi_sig_1_t, phi_sig_2_t,\
           prior_sig_1_t, prior_sig_2_t, theta_mu_t, theta_sig_t, coeff_t,\
           marginal_ll

((s_1_t, s_2_t, kl_1_t, kl_2_t, phi_sig_1_t, phi_sig_2_t,
  prior_sig_1_t, prior_sig_2_t, theta_mu_t, theta_sig_t, coeff_t,
  marginal_ll),
 updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x],
                outputs_info=[coder_1.get_init_state(),
                              coder_2.get_init_state(),
                              None, None, None, None, None,
                              None, None, None, None, None])
for k, v in updates.iteritems():
    k.default_update = v

reshaped_x = x.reshape((x.shape[0]*x.shape[1], -1))
reshaped_theta_mu = theta_mu_t.reshape((theta_mu_t.shape[0]*theta_mu_t.shape[1], -1))
reshaped_theta_sig = theta_sig_t.reshape((theta_sig_t.shape[0]*theta_sig_t.shape[1], -1))
reshaped_coeff = coeff_t.reshape((coeff_t.shape[0]*coeff_t.shape[1], -1))
reshaped_mask = mask.flatten()
kl_1_term = kl_1_t.reshape((kl_1_t.shape[0]*kl_1_t.shape[1], -1))
kl_2_term = kl_2_t.reshape((kl_2_t.shape[0]*kl_2_t.shape[1], -1))
recon_term = GMM(reshaped_x, reshaped_theta_mu, reshaped_theta_sig, reshaped_coeff)
# Apply mask
kl_1_term = kl_1_term[reshaped_mask.nonzero()].mean()
kl_2_term = kl_2_term[reshaped_mask.nonzero()].mean()
recon_term = recon_term[reshaped_mask.nonzero()].mean()
cost = recon_term + kl_1_term + kl_2_term
ll_term = marginal_ll.mean()
cost.name = 'cost'
recon_term.name = 'recon_term'
kl_1_term.name = 'kl1_term'
kl_2_term.name = 'kl2_term'
ll_term.name = 'negative log-likelihood'

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

max_phi_sig_1 = phi_sig_1_t.max()
mean_phi_sig_1 = phi_sig_1_t.mean()
min_phi_sig_1 = phi_sig_1_t.min()
max_phi_sig_1.name = 'max_phi_sig_1'
mean_phi_sig_1.name = 'mean_phi_sig_1'
min_phi_sig_1.name = 'min_phi_sig_1'

max_prior_sig_1 = prior_sig_1_t.max()
mean_prior_sig_1 = prior_sig_1_t.mean()
min_prior_sig_1 = prior_sig_1_t.min()
max_prior_sig_1.name = 'max_prior_sig_1'
mean_prior_sig_1.name = 'mean_prior_sig_1'
min_prior_sig_1.name = 'min_prior_sig_1'

max_phi_sig_2 = phi_sig_2_t.max()
mean_phi_sig_2 = phi_sig_2_t.mean()
min_phi_sig_2 = phi_sig_2_t.min()
max_phi_sig_2.name = 'max_phi_sig_2'
mean_phi_sig_2.name = 'mean_phi_sig_2'
min_phi_sig_2.name = 'min_phi_sig_2'

max_prior_sig_2 = prior_sig_2_t.max()
mean_prior_sig_2 = prior_sig_2_t.mean()
min_prior_sig_2 = prior_sig_2_t.min()
max_prior_sig_2.name = 'max_prior_sig_2'
mean_prior_sig_2.name = 'mean_prior_sig_2'
min_prior_sig_2.name = 'min_prior_sig_2'

W_x_t_s1_t = coder_1.params['W_x_t__coder_1'][:, :shared_nout_1]
col_norm_W_x_t_s1_t = T.sqrt((W_x_t_s1_t**2).sum(axis=0))
max_col_norm_W_x_t_s1_t = col_norm_W_x_t_s1_t.max()
mean_col_norm_W_x_t_s1_t = col_norm_W_x_t_s1_t.mean()
min_col_norm_W_x_t_s1_t = col_norm_W_x_t_s1_t.min()
max_col_norm_W_x_t_s1_t.name = 'max_col_norm_W_x_t_s1_t'
mean_col_norm_W_x_t_s1_t.name = 'mean_col_norm_W_x_t_s1_t'
min_col_norm_W_x_t_s1_t.name = 'min_col_norm_W_x_t_s1_t'

W_z1_t_s1_t = coder_1.params['W_z_1_t__coder_1'][:, :shared_nout_1]
col_norm_W_z1_t_s1_t = T.sqrt((W_z1_t_s1_t**2).sum(axis=0))
max_col_norm_W_z1_t_s1_t = col_norm_W_z1_t_s1_t.max()
mean_col_norm_W_z1_t_s1_t = col_norm_W_z1_t_s1_t.mean()
min_col_norm_W_z1_t_s1_t = col_norm_W_z1_t_s1_t.min()
max_col_norm_W_z1_t_s1_t.name = 'max_col_norm_weight_z1_t_s1_t'
mean_col_norm_W_z1_t_s1_t.name = 'mean_col_norm_weight_z1_t_s1_t'
min_col_norm_W_z1_t_s1_t.name = 'min_col_norm_weight_z1_t_s1_t'

W_z2_t_s1_t = coder_1.params['W_z_2_t__coder_1'][:, :shared_nout_1]
col_norm_W_z2_t_s1_t = T.sqrt((W_z2_t_s1_t**2).sum(axis=0))
max_col_norm_W_z2_t_s1_t = col_norm_W_z2_t_s1_t.max()
mean_col_norm_W_z2_t_s1_t = col_norm_W_z2_t_s1_t.mean()
min_col_norm_W_z2_t_s1_t = col_norm_W_z2_t_s1_t.min()
max_col_norm_W_z2_t_s1_t.name = 'max_col_norm_weight_z2_t_s1_t'
mean_col_norm_W_z2_t_s1_t.name = 'mean_col_norm_weight_z2_t_s1_t'
min_col_norm_W_z2_t_s1_t.name = 'min_col_norm_weight_z2_t_s1_t'

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
    Monitoring(freq=10,
               ddout=[cost, recon_term, kl_1_term, kl_2_term, ll_term,
                      max_phi_sig_1, mean_phi_sig_1, min_phi_sig_1,
                      max_prior_sig_1, mean_prior_sig_1, min_prior_sig_1,
                      max_phi_sig_2, mean_phi_sig_2, min_phi_sig_2,
                      max_prior_sig_2, mean_prior_sig_2, min_prior_sig_2,
                      max_theta_sig, mean_theta_sig, min_theta_sig,
                      max_x, mean_x, min_x,
                      max_theta_mu, mean_theta_mu, min_theta_mu,
                      max_col_norm_W_x_t_s1_t,
                      mean_col_norm_W_x_t_s1_t,
                      min_col_norm_W_x_t_s1_t,
                      max_col_norm_W_z1_t_s1_t,
                      mean_col_norm_W_z1_t_s1_t,
                      min_col_norm_W_z1_t_s1_t,
                      max_col_norm_W_z2_t_s1_t,
                      mean_col_norm_W_z2_t_s1_t,
                      min_col_norm_W_z2_t_s1_t,
                      coeff_max, coeff_min, coeff_mean_max, coeff_mean_min],
               data=[Iterator(trdata, batch_size, start=0, end=1000)]),
    Picklize(freq=1000,
             path=save_path),
    EarlyStopping(freq=500, path=save_path),
    WeightNorm()
]

mainloop = Training(
    name='stacked_rnnvae_v3',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
