import ipdb
import theano
import theano.tensor as T

from cle.cle.layers import StemCell
from cle.cle.layers.cost import GaussianLayer
from cle.cle.utils import sharedX, tolist, totuple, unpack, predict
from sk.cle.models.sparse_vae.cost import KLGaussianStdGaussian, KLGaussianGaussian

from theano.compat.python2x import OrderedDict


class ClockworkPriorLayer(StemCell):
    """
    Clockwork prior layer which either computes
    the kl of sparse VAE or generates samples using
    normal distribution when mod(t, N)==0

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 N=1,
                 use_sample=False,
                 num_sample=1,
                 **kwargs):
        super(ClockworkPriorLayer, self).__init__(**kwargs)
        self.N = N
        self.use_sample = use_sample
        if use_sample:
            self.inner_fn = self.which_method('sample')
        else:
            self.inner_fn = self.which_method('cost')
        if use_sample:
            if num_sample is None:
                raise ValueError("If you are going to use sampling,\
                                  provide the number of samples.")
        self.num_sample = num_sample

    def which_method(self, which):
        return getattr(self, which)
 
    def cost(self, X):
        if len(X) != 2 and len(X) != 4:
            raise ValueError("The number of inputs does not match.")
        if len(X) == 2:
            return KLGaussianStdGaussian(X[0], X[1])
        elif len(X) == 4:
            return KLGaussianGaussian(X[0], X[1], X[2], X[3])

    def sample(self, X):
        if len(X) != 2 and len(X) != 4:
            raise ValueError("The number of inputs does not match.")
        mu = X[0]
        sig = X[1]
        mu = mu.dimshuffle(0, 'x', 1)
        sig = sig.dimshuffle(0, 'x', 1)
        epsilon = self.theano_rng.normal(size=(mu.shape[0],
                                               self.num_sample,
                                               mu.shape[-1]),
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + sig * epsilon
        z = z.reshape((z.shape[0] * z.shape[1], -1))
        return z

    def fprop(self, X):
        idx = X[0]
        X = X[1:]
        z = theano.ifelse.ifelse(T.neq(T.mod(idx, self.N), 0),
                                 T.zeros((X[0].shape[0]*self.num_sample,
                                          self.nout),
                                          dtype=X[0].dtype),
                                 self.inner_fn(X))
        z.name = self.name
        return z

    def __getstate__(self):
        dic = self.__dict__.copy()
        dic.pop('inner_fn')
        return dic
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.use_sample:
            self.inner_fn = self.which_method('sample')
        else:
            self.inner_fn = self.which_method('cost')
  
    def initialize(self):
        pass


class SpikenSlabPriorLayer(StemCell):
    """
    Spike and slab prior layer which either computes 
    the kl of sparse VAE or generates samples adaptively

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 N=1,
                 use_sample=False,
                 num_sample=1,
                 **kwargs):
        super(ClockworkPriorLayer, self).__init__(**kwargs)
        self.N = N
        self.use_sample = use_sample
        if use_sample:
            self.inner_fn = self.which_method('sample')
        else:
            self.inner_fn = self.which_method('cost')
        if use_sample:
            if num_sample is None:
                raise ValueError("If you are going to use sampling,\
                                  provide the number of samples.")
        self.num_sample = num_sample

    def which_method(self, which):
        return getattr(self, which)
 
    def cost(self, X):
        if len(X) != 2 and len(X) != 4:
            raise ValueError("The number of inputs does not match.")
        if len(X) == 2:
            return KLGaussianStdGaussian(X[0], X[1])
        elif len(X) == 4:
            return KLGaussianGaussian(X[0], X[1], X[2], X[3])

    def sample(self, X):
        if len(X) != 2 and len(X) != 4:
            raise ValueError("The number of inputs does not match.")
        mu = X[0]
        sig = X[1]
        mu = mu.dimshuffle(0, 'x', 1)
        sig = sig.dimshuffle(0, 'x', 1)
        epsilon = self.theano_rng.normal(size=(mu.shape[0],
                                               self.num_sample,
                                               mu.shape[-1]),
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + sig * epsilon
        z = z.reshape((z.shape[0] * z.shape[1], -1))
        return z

    def fprop(self, X):
        idx = X[0]
        X = X[1:]
        z = theano.ifelse.ifelse(T.neq(T.mod(idx, self.N), 0),
                                 T.zeros((X[0].shape[0]*self.num_sample,
                                          self.nout),
                                          dtype=X[0].dtype),
                                 self.inner_fn(X))
        z.name = self.name
        return z

    def __getstate__(self):
        dic = self.__dict__.copy()
        dic.pop('inner_fn')
        return dic
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.use_sample:
            self.inner_fn = self.which_method('sample')
        else:
            self.inner_fn = self.which_method('cost')
  
    def initialize(self):
        pass


class NCGMMLayer(GaussianLayer):
    """
    Noise collecting Gaussian mixture model layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 p_noise=0.1,
                 **kwargs):
        super(NCGMMLayer, self).__init__(**kwargs)
        self.p_noise = p_noise
   
    def cost(self, X):
        if len(X) != 4:
            raise ValueError("The number of inputs does not match.")
        cost = NCGMM(X[0], X[1], X[2], X[3], self.p_noise)
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()

    def sample(self, X):
        mu = X[0]
        sig = X[1]
        coeff = X[2]
        n_noise = T.cast(T.floor(coeff.shape[-1] * self.p_noise), 'int32')
        mu = T.concatenate(
            [mu, T.zeros((mu.shape[0],
                          n_noise*sig.shape[1]/coeff.shape[-1]))],
            axis=1
        )
        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))
        sig = sig.reshape((sig.shape[0],
                           sig.shape[1]/coeff.shape[-1],
                           coeff.shape[-1]))
        idx = predict(
            self.theano_rng.multinomial(
                pvals=coeff,
                dtype=coeff.dtype
            ),
            axis=1
        )
        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = sig[T.arange(sig.shape[0]), :, idx]
        sample = self.theano_rng.normal(size=mu.shape,
                                        avg=mu, std=sig,
                                        dtype=mu.dtype)
        return sample


class IndGMMLayer(GaussianLayer):
    """
    Gaussian mixture model layer with bell-shape indicators

    Parameters
    ----------
    .. todo::
    """
    def cost(self, X):
        if len(X) != 4:
            raise ValueError("The number of inputs does not match.")
        cost = NCGMM(X[0], X[1], X[2], X[3])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()

    def sample(self, X):
        ind, mu = X[0][:, 0], X[0][:, 1:]
        sig = X[1]
        coeff = X[2]
        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))
        sig = sig.reshape((sig.shape[0],
                           sig.shape[1]/coeff.shape[-1],
                           coeff.shape[-1]))
        idx = predict(
            self.theano_rng.multinomial(
                pvals=coeff,
                dtype=coeff.dtype
            ),
            axis=1
        )
        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = sig[T.arange(sig.shape[0]), :, idx]
        sample = self.theano_rng.normal(size=mu.shape,
                                        avg=mu, std=sig,
                                        dtype=mu.dtype)
        sample = T.concatenate([ind.dimshuffle(0, 'x'), sample], axis=1)
        return sample
