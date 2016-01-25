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
    EarlyStopping
)
from cle.cle.train.opt import Adam
from cle.cle.utils import flatten
from sk.datasets.blizzard import Blizzard


datapath = '/home/chungjun/data/blizzard/segmented/'
savepath = '/raid/chungjun/repos/sk/saved/vae/blizzard/'

batch_size = 100
num_sample = 10
inpsz = 200
latsz = 100
lat_emb = 1000
out_emb = 1000
k = 10
outsz = inpsz * k
enc_nout = 2000
dec_nout = 2000
pec_nout = 2000
debug = 1

model = Model()
trdata = Blizzard(name='train',
                  path=datapath,
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
x2 = T.concatenate([T.zeros((1, x.shape[1], x.shape[2])), x[:-1]], axis=0)
x2.name = 'x2'

encoder = LSTM(name='encoder',
               parent=['x_t'],
               parent_dim=[inpsz],
               batch_size=batch_size,
               nout=enc_nout,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

decoder = LSTM(name='decoder',
               parent=['z_t'],
               parent_dim=[latsz],
               batch_size=batch_size,
               nout=dec_nout,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

pecoder = LSTM(name='pecoder',
               parent=['x_tm1'],
               parent_dim=[inpsz],
               batch_size=batch_size,
               nout=pec_nout,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

phi_emb = FullyConnectedLayer(name='phi_emb',
                              parent=['encoder'],
                              parent_dim=[enc_nout],
                              nout=lat_emb,
                              unit='tanh',
                              init_W=init_W,
                              init_b=init_b)

phi_mu = FullyConnectedLayer(name='phi_mu',
                             parent=['phi_emb'],
                             parent_dim=[lat_emb],
                             nout=latsz,
                             unit='linear',
                             init_W=init_W,
                             init_b=init_b)

phi_sig = RealVectorLayer(name='phi_sig',
                          nout=latsz,
                          unit='softplus',
                          cons=1e-4,
                          init_b=init_b_sig)

prior = PriorLayer(name='prior',
                   parent=['phi_mu', 'phi_sig'],
                   parent_dim=[latsz, latsz],
                   use_sample=1,
                   num_sample=1,
                   nout=latsz)

prior_emb = FullyConnectedLayer(name='prior_emb',
                                parent=['decoder'],
                                parent_dim=[dec_nout],
                                nout=lat_emb,
                                unit='tanh',
                                init_W=init_W,
                                init_b=init_b)

prior_mu = FullyConnectedLayer(name='prior_mu',
                               parent=['prior_emb'],
                               parent_dim=[lat_emb],
                               nout=latsz,
                               unit='linear',
                               init_W=init_W,
                               init_b=init_b)

prior_sig = RealVectorLayer(name='prior_sig',
                            nout=latsz,
                            unit='softplus',
                            cons=1e-4,
                            init_b=init_b_sig)

kl = PriorLayer(name='kl',
                parent=['phi_mu', 'phi_sig', 'prior_mu', 'prior_sig'],
                parent_dim=[latsz, latsz, latsz, latsz],
                use_sample=0,
                nout=latsz)

theta_emb = FullyConnectedLayer(name='theta_emb',
                                parent=['pecoder'],
                                parent_dim=[pec_nout],
                                nout=out_emb,
                                unit='tanh',
                                init_W=init_W,
                                init_b=init_b)

theta_mu = FullyConnectedLayer(name='theta_mu',
                               parent=['theta_emb'],
                               parent_dim=[out_emb],
                               nout=outsz,
                               unit='linear',
                               init_W=init_W,
                               init_b=init_b)

theta_sig = RealVectorLayer(name='theta_sig',
                            nout=outsz,
                            unit='softplus',
                            cons=1e-4,
                            init_b=init_b_sig)

coeff = FullyConnectedLayer(name='coeff',
                            parent=['theta_emb'],
                            parent_dim=[out_emb],
                            nout=k,
                            unit='softmax',
                            init_W=init_W,
                            init_b=init_b)

nodes = [encoder, decoder, pecoder, phi_emb, phi_mu, phi_sig, prior,
         prior_emb, prior_mu, prior_sig, kl, theta_emb, theta_mu,
         theta_sig, coeff]
for node in nodes:
    node.initialize()
params = flatten([node.get_params().values() for node in nodes])

def inner_fn(x_t, x_tm1, enc_tm1, dec_tm1, pec_tm1, phi_sig_t, prior_sig_t,
             theta_sig_t):

    enc_t = encoder.fprop([[x_t], [enc_tm1]])

    phi_emb_t = phi_emb.fprop([enc_t])
    phi_mu_t = phi_mu.fprop([phi_emb_t])

    pec_t = pecoder.fprop([[x_tm1], [pec_tm1]])

    prior_emb_t = prior_emb.fprop([pec_t])
    prior_mu_t = prior_mu.fprop([prior_emb_t])

    z_t = prior.fprop([phi_mu_t, phi_sig_t])
    kl_t = kl.fprop([phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t])

    dec_t = decoder.fprop([[z_t], [dec_tm1]])

    theta_emb_t = theta_emb.fprop([dec_t])
    theta_mu_t = theta_mu.fprop([theta_emb_t])
    coeff_t = coeff.fprop([theta_emb_t])

    marginal_ll = []
    for i in xrange(num_sample):
        z_is = prior.fprop([phi_mu_t, phi_sig_t])
        theta_emb_t = theta_emb.fprop([dec_t])
        theta_mu_t = theta_mu.fprop([theta_emb_t])
        theta_sig_t = theta_sig.fprop([theta_emb_t])
        coeff_t = coeff.fprop([theta_emb_t])
        w = Gaussian(z_is, prior_mu_t, prior_sig_t) -\
            Gaussian(z_is, phi_mu_t, phi_sig_t)
        marginal_ll.append(GMM(x_t, theta_mu_t, theta_sig_t, coeff_t) + w)
    marginal_ll = T.concatenate(marginal_ll, axis=0).mean()

    return enc_t, dec_t, pec_t, kl_t, theta_mu_t, coeff_t, marginal_ll

prior_sig_t = prior_sig.fprop()
phi_sig_t = phi_sig.fprop()
theta_sig_t = theta_sig.fprop()
((enc_t, dec_t, pec_t, kl_t, theta_mu_t, coeff_t, marginal_ll), updates) =\
    theano.scan(fn=inner_fn,
                sequences=[x, x2],
                outputs_info=[encoder.get_init_state(),
                              decoder.get_init_state(),
                              pecoder.get_init_state(),
                              None, None, None, None],
                non_sequences=[phi_sig_t, prior_sig_t, theta_sig_t])
for k, v in updates.iteritems():
    k.default_update = v

reshaped_x = x.reshape((x.shape[0]*x.shape[1], -1))
reshaped_theta_mu = theta_mu_t.reshape((theta_mu_t.shape[0]*theta_mu_t.shape[1], -1))
reshaped_coeff = coeff_t.reshape((coeff_t.shape[0]*coeff_t.shape[1], -1))
reshaped_mask = mask.flatten()

kl_term = kl_t.reshape((kl_t.shape[0]*kl_t.shape[1], -1))
recon_term = GMM(reshaped_x, reshaped_theta_mu, theta_sig_t, reshaped_coeff)
# Apply mask
kl_term = kl_term[reshaped_mask.nonzero()].mean()
recon_term = recon_term[reshaped_mask.nonzero()].mean()
cost = recon_term + kl_term
ll_term = marginal_ll.mean()
cost.name = 'cost'
recon_term.name = 'recon_term'
kl_term.name = 'kl_term'
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

W_x_t_s_t = encoder.params['W_x_t__encoder'][:, :enc_nout]
col_norm_W_x_t_s_t = T.sqrt((W_x_t_s_t**2).sum(axis=0))
max_col_norm_W_x_t_s_t = col_norm_W_x_t_s_t.max()
mean_col_norm_W_x_t_s_t = col_norm_W_x_t_s_t.mean()
min_col_norm_W_x_t_s_t = col_norm_W_x_t_s_t.min()
max_col_norm_W_x_t_s_t.name = 'max_col_norm_W_x_t_s_t'
mean_col_norm_W_x_t_s_t.name = 'mean_col_norm_W_x_t_s_t'
min_col_norm_W_x_t_s_t.name = 'min_col_norm_W_x_t_s_t'

W_z_t_s_t = decoder.params['W_z_t__decoder'][:, :dec_nout]
col_norm_W_z_t_s_t = T.sqrt((W_z_t_s_t**2).sum(axis=0))
max_col_norm_W_z_t_s_t = col_norm_W_z_t_s_t.max()
mean_col_norm_W_z_t_s_t = col_norm_W_z_t_s_t.mean()
min_col_norm_W_z_t_s_t = col_norm_W_z_t_s_t.min()
max_col_norm_W_z_t_s_t.name = 'max_col_norm_W_z_t_s_t'
mean_col_norm_W_z_t_s_t.name = 'mean_col_norm_W_z_t_s_t'
min_col_norm_W_z_t_s_t.name = 'min_col_norm_W_z_t_s_t'

model.inputs = [x, mask]
model._params = params
model.nodes = nodes

optimizer = Adam(
    lr=0.001
)

extension = [
    GradientClipping(batch_size=batch_size),
    EpochCount(1000000),
    Monitoring(freq=10,
               ddout=[cost, recon_term, kl_term, ll_term,
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
                      min_col_norm_W_z_t_s_t],
               data=[Iterator(trdata, batch_size, start=0, end=1000)]),
    Picklize(freq=1000,
             path=savepath),
    EarlyStopping(freq=500, path=savepath)
]

mainloop = Training(
    name='rnnvae_free_theta_sig_decouple',
    data=Iterator(trdata, batch_size),
    model=model,
    optimizer=optimizer,
    cost=cost,
    outputs=[cost],
    extension=extension
)
mainloop.run()
