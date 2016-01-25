import ipdb
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import theano
import theano.tensor as T

from cle.cle.data import DesignMatrix, TemporalSeries
from cle.cle.data.prep import SequentialPrepMixin
from cle.cle.utils import segment_axis, tolist, totuple
from onomatopoeia_utils import fetch_onomatopoeia

from scipy import signal
from scipy.io import wavfile


class Onomatopoeia(TemporalSeries, SequentialPrepMixin):
    """
    Onomatopoeia dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, tag_match="all", frame_size=1,
                 target_size=None, thresh=50000,
                 X_mean=None, X_std=None, **kwargs):
        self.frame_size = frame_size
        self.tag_match = tag_match
        if target_size is not None:
            self.target_size = target_size
        else:
            self.target_size = frame_size
        self.thresh = thresh
        self.X_mean = X_mean
        self.X_std = X_std
        super(Onomatopoeia, self).__init__(**kwargs)

    def load(self, data_path):
        data, tags = fetch_onomatopoeia(data_path)
        # hardcode split for now
        random_state = np.random.RandomState(1999)
        indices = np.arange(len(data))
        random_state.shuffle(indices)
        if self.name == "train":
            idx = int(.9 * float(len(data)))
            assert idx != len(data)
            data = data[indices[:idx]]
        elif self.name == "valid":
            idx = int(.1 * float(len(data)))
            assert idx != 0
            data = data[indices[-idx:]]
        else:
            raise ValueError("name = %s is not supported!" % self.name)
        raw_X = []
        for x in data:
            if len(x) < self.thresh:
                raw_X.append(np.asarray(x, dtype=theano.config.floatX))
            else:
                raw_X.append(np.asarray(x[:self.thresh], dtype=theano.config.floatX))
        raw_X = np.array(raw_X)
        pre_X, self.X_mean, self.X_std = self.global_normalize(raw_X, self.X_mean, self.X_std)
        X = np.array([segment_axis(x, self.frame_size, 0) for x in pre_X])
        return [X]

    def theano_vars(self):
        return [T.tensor3('x', dtype=theano.config.floatX),
                T.matrix('x_mask', dtype=theano.config.floatX)]

    def test_theano_vars(self):
        return [T.matrix('x', dtype=theano.config.floatX)]

    def slices(self, start, end):
        batches = [mat[start:end] for mat in self.data]
        mask = self.create_mask(batches[0].swapaxes(0, 1))
        batches = [self.zero_pad(batch) for batch in batches]
        return totuple([batches[0], mask])


if __name__ == "__main__":
    save_name = 'ground_truth_'
    #save_path = '/raid/chungjun/repos/sk/saved/rnnvae/data/onomatopoeia/ground_truth/'
    #data_path = '/raid/chungjun/data/ubisoft/onomatopoeia/'
    save_path = '/home/junyoung/repos/sk/saved/rnnvae/data/onomatopoeia/ground_truth/'
    data_path = '/home/junyoung/data/ubisoft/onomatopoeia/'
    frame_size = 100
    onomatopoeia = Onomatopoeia(name='train',
                                path=data_path,
                                frame_size=frame_size)
    batch = onomatopoeia.slices(start=0, end=10)
    ipdb.set_trace()
    X_mean = onomatopoeia.X_mean
    X_std = onomatopoeia.X_std
    X = onomatopoeia.data[0]
    sub_X = X
    sub_X = sub_X * X_std + X_mean
    for i, x in enumerate(sub_X):
        sample = x.flatten()
        wave_path = save_path + save_name + str(i) + '.wav'
        wavfile.write(wave_path, 16000, np.int16(sample))
        sample_path = save_path + save_name + str(i) + '.png'
        fig = plt.figure()
        plt.plot(sample)
        plt.savefig(sample_path, bbox_inches='tight', format='png')
