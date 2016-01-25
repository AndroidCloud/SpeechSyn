import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.utils.op import logsumexp


def NCGMM(y, mu, sig, coeff, p_noise=0.1):
    """
    Gaussian mixture model negative log-likelihood
    with noise collecting Gaussian

    Parameters
    ----------
    y      : TensorVariable
    mu     : FullyConnected (Linear)
    sig    : FullyConnected (Softplus)
    coeff  : FullyConnected (Softmax)
    """
    n_noise = T.cast(T.floor(coeff.shape[-1] * p_noise), 'int32')
    y = y.dimshuffle(0, 1, 'x')
    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]/(coeff.shape[-1]-n_noise),
                     coeff.shape[-1]-n_noise))
    sig = sig.reshape((sig.shape[0],
                       sig.shape[1]/coeff.shape[-1],
                       coeff.shape[-1]))
    vsig = sig[:, :, :-n_noise]
    uvsig = sig[:, :, -n_noise:]
    vinner = -0.5 * T.sum(T.sqr(y - mu) / vsig**2 + 2 * T.log(vsig) +
                          T.log(2 * np.pi), axis=1)
    uvinner = -0.5 * T.sum(T.sqr(y) / uvsig**2 + 2 * T.log(uvsig) +
                           T.log(2 * np.pi), axis=1)
    inner = T.concatenate([vinner, uvinner], axis=1)
    nll = -logsumexp(T.log(coeff) + inner, axis=1)
    return nll


def KLGaussianStdGaussian(mu, sig):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and standardized Gaussian dist.

    Parameters
    ----------
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    kl = 0.5 * (-2 * T.log(sig) + mu**2 + sig**2 - 1)
    return kl


def KLGaussianGaussian(mu1, sig1, mu2, sig2):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist.

    Parameters
    ----------
    mu1  : FullyConnected (Linear)
    sig1 : FullyConnected (Softplus)
    mu2  : FullyConnected (Linear)
    sig2 : FullyConnected (Softplus)
    """
    kl = 0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) + (sig1**2 + (mu1 - mu2)**2) /
                sig2**2 - 1)
    return kl
