import ipdb
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import theano
import theano.tensor as T

from cle.cle.data import TemporalSeries
from cle.cle.data.prep import SequentialPrepMixin
from cle.cle.utils import segment_axis, tolist, totuple
from sk.datasets.blizzard_hdf5 import fetch_blizzard

from scipy.io import wavfile


class Blizzard(TemporalSeries, SequentialPrepMixin):
    """
    Blizzard dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, frame_size=1, target_size=None, use_log_space=0,
                 use_derivative=0, prep='normalize', cut_len=50000,
                 multi_source=0, shuffle=1, **kwargs):
        self.frame_size = frame_size
        if target_size is not None:
            self.target_size = target_size
        else:
            self.target_size = frame_size
        self.prep = prep
        self.use_log_space = use_log_space
        self.use_derivative = use_derivative
        self.multi_source = multi_source
        self.shuffle = shuffle
        super(Blizzard, self).__init__(**kwargs)

    def load(self, data_path):
        if self.name == 'train':
            data_path = data_path + 'sf_train_segmented_0.npy'
        elif self.name == 'valid':
            data_path = data_path + 'sf_valid_segmented_0.npy'
        data = np.load(data_path)
        raw_X = []
        for x in data:
            if len(x) < 50000:
                raw_X.append(np.asarray(x, dtype=theano.config.floatX))
            else:
                half_len = np.int(len(x) / 2.)
                raw_X.append(np.asarray(x[:half_len], dtype=theano.config.floatX))
                raw_X.append(np.asarray(x[half_len:], dtype=theano.config.floatX))
        raw_X = np.array(raw_X)
        if self.shuffle:
            idx = np.random.permutation(len(raw_X))
            raw_X = raw_X[idx]
        pre_X = self.apply_preprocessing(raw_X)
        if self.multi_source:
            X = [np.array([segment_axis(x, self.frame_size, 0) for x in X]) for X in pre_X]
        else:
            X = [np.array([segment_axis(x, self.frame_size, 0) for x in pre_X])]
        return X

    def apply_preprocessing(self, raw_X):
        self.X_mean = None
        self.X_std = None
        self.X_max = None
        self.X_min = None
        if self.use_log_space:
            raw_X = self.embed_log_space(raw_X)
        if self.use_derivative:
            raw_X = self.get_temporal_derivative(raw_X)
        if self.prep == 'normalize':
            pre_X, self.X_mean, self.X_std = self.global_normalize(raw_X)
        elif self.prep == 'standardize':
            pre_X, self.X_max, self.X_min = self.standardize(raw_X)
        else:
            pre_X = raw_X
        return pre_X

    def get_temporal_derivative(self, X):
        new_X = []
        for x in X:
            x_tm1 = np.concatenate([np.zeros(1)[None, :], x[:-1][None, :]], axis=1).flatten()
            dx = x - x_tm1
            new_X.append(dx)
        return new_X

    def embed_log_space(self, X):
        X = np.array([np.log(np.abs(x) + 1) * np.sign(x) for x in X])
        return X

    def theano_vars(self):
        return [T.tensor3('x', dtype=theano.config.floatX),
                T.matrix('x_mask', dtype=theano.config.floatX)]

    def test_theano_vars(self):
        return [T.matrix('x', dtype=theano.config.floatX)]

    def slices(self, start, end):
        batches = [mat[start:end] for mat in self.data]
        mask = self.create_mask(batches[0])
        batches = [self.zero_pad(batch) for batch in batches]
        return totuple([batches[0], mask])


class BlizzardForHigherOrder(Blizzard):
    """
    Blizzard dataset batch provider which outputs multiple channels

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, n_order=2, **kwargs):
        self.n_order = n_order
        super(BlizzardForHigherOrder, self).__init__(multi_source=1, **kwargs)

    def apply_preprocessing(self, raw_X):
        self.X_mean = None
        self.X_std = None
        self.X_max = None
        self.X_min = None
        if self.use_log_space:
            raw_X = self.embed_log_space(raw_X)
        if self.prep == 'normalize':
            pre_X, self.X_mean, self.X_std = self.global_normalize(raw_X)
        elif self.prep == 'standardize':
            pre_X, self.X_max, self.X_min = self.standardize(raw_X)
        else:
            pre_X = raw_X
        list_X = [pre_X]
        for i in xrange(self.n_order - 1):
            list_X.append(self.get_temporal_derivative(list_X[-1]))
        return list_X

    def theano_vars(self):
        outputs = []
        for i in xrange(self.n_order):
            variable_name = 'x_' + str(i)
            outputs.append(T.tensor3(variable_name, dtype=theano.config.floatX))
        return outputs + [T.matrix('x_mask', dtype=theano.config.floatX)]

    def test_theano_vars(self):
        outputs = []
        for i in xrange(self.n_order):
            variable_name = 'x_' + str(i)
            outputs.append(T.matrix(variable_name, dtype=theano.config.floatX))
        return outputs

    def slices(self, start, end):
        batches = [mat[start:end] for mat in self.data]
        mask = self.create_mask(batches[0])
        batches = [self.zero_pad(batch) for batch in batches]
        return totuple([batches[0], batches[1], mask])


class Blizzard_h5(TemporalSeries, SequentialPrepMixin):
    """
    Blizzard dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, frame_size=200, X_mean=None, X_std=None, shuffle=0,
                 **kwargs):
        self.frame_size = frame_size
        self.X_mean = X_mean
        self.X_std = X_std
        self.shuffle = shuffle
        super(Blizzard_h5, self).__init__(**kwargs)

    def load(self, data_path):
        X = fetch_blizzard(data_path, self.shuffle)
        if self.X_mean is None or self.X_std is None:
            prev_mean = None
            prev_var = None
            n_seen = 0
            n_inter = 10000
            range_end = np.int(np.ceil(len(X) / float(n_inter)))
            for i in xrange(range_end):
                n_seen += 1
                i_start = i*n_inter
                i_end = min((i+1)*n_inter, len(X))
                if prev_mean is None:
                    prev_mean = X[i_start:i_end].mean()
                    prev_var = 0.
                else:
                    curr_mean = prev_mean +\
                        (X[i_start:i_end] - prev_mean).mean() / n_seen
                    curr_var = prev_var +\
                        ((X[i_start:i_end] - prev_mean) *\
                        (X[i_start:i_end] - curr_mean)).mean()
                    prev_mean = curr_mean
                    prev_var = curr_var
                print "[%d / %d]" % (i, range_end)
            self.X_mean = prev_mean
            self.X_std = np.sqrt(prev_var / n_seen)
            ipdb.set_trace()
        return X

    def theano_vars(self):
        return T.tensor3('x', dtype=theano.config.floatX)

    def test_theano_vars(self):
        return T.matrix('x', dtype=theano.config.floatX)

    def slices(self, start, end):
        batch = np.array(self.data[start:end], dtype=theano.config.floatX)
        batch -= self.X_mean
        batch /= self.X_std
        batch = np.asarray([segment_axis(x, self.frame_size, 0) for x in batch])
        ipdb.set_trace()
        batch = batch.transpose(1, 0, 2)
        return totuple(batch)


def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = scipy.signal.butter(order, low, btype='high')
    return b, a


def butter_highpass_filter(x, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, x)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(x, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, x)
    return y


if __name__ == "__main__":
    save_name = 'ground_truth_'
    data_path = '/data/lisatmp3/chungjun/data/blizzard/'
    save_path = '/data/lisatmp/chungjun/nips2015/blizzard/pkl/'
    #save_path = '/raid/chungjun/repos/sk/saved/rnnvae/blizzard/ground_truth/'
    #data_path = '/raid/chungjun/data/blizzard/'
    #save_path = '/home/junyoung/repos/sk/saved/rnnvae/data/blizzard/ground_truth_shuffled/'
    #data_path = '/home/junyoung/data/blizzard/segmented/'
    frame_size = 200
    test_for_blizzard = 0
    test_for_multiple = 0
    test_for_h5 = 1
    use_derivative = 0
    use_log_space = 0
    if test_for_blizzard:
        blizzard = Blizzard(name='train',
                            path=data_path,
                            use_derivative=use_derivative,
                            use_log_space=use_log_space,
                            prep='normalize',
                            #prep='None',
                            frame_size=frame_size)
        batch = blizzard.slices(start=0, end=100)
        X_mean = blizzard.X_mean
        X_std = blizzard.X_std
        X = blizzard.data[0]
        if use_derivative:
            dx = X[0].flatten()
            dx_noise = dx + 0.01 * np.random.randn(32000)
            undx_noise = dx_noise * X_std + X_mean
            x_noise = np.cumsum(undx_noise)
            x_filtered = scipy.signal.lfilter([-1, 1] * 5, [1], x_noise)
            x_butterworth = butter_highpass(x_noise, 16000, 50)
        sub_X = X[:100]
        sub_X = sub_X * X_std + X_mean
        for i, x in enumerate(sub_X):
            sample = x.flatten()
            #wave_path = save_path + save_name + str(i) + '.wav'
            #wavfile.write(wave_path, 16000, np.int16(sample))
            sample_path = save_path + save_name + str(i) + '.png'
            fig = plt.figure()
            plt.plot(sample)
            plt.savefig(sample_path, bbox_inches='tight', format='png')
    elif test_for_multiple:
        blizzard = BlizzardForHigherOrder(name='train',
                                          path=data_path,
                                          n_order=2,
                                          prep='normalize',
                                          frame_size=frame_size)
        X_mean = blizzard.X_mean
        X_std = blizzard.X_std
        X = blizzard.data[0]
        ipdb.set_trace()
    elif test_for_h5:
        blizzard = Blizzard_h5(name='train',
                               path=data_path,
                               #frame_size=frame_size)
                               frame_size=frame_size,
                               X_mean=39.0267,
                               X_std=3177.812)
        X_mean = blizzard.X_mean
        X_std = blizzard.X_std
        X = blizzard.data[0]
        batch = blizzard.slices(0, 100)
        ipdb.set_trace()
