import ipdb
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
from iamondb_utils import fetch_iamondb


class IAMOnDB(TemporalSeries, SequentialPrepMixin):
    """
    IAMOnDB dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, prep='none', cond=False, **kwargs):
        self.prep = prep
        self.cond = cond
        super(IAMOnDB, self).__init__(**kwargs)

    def load(self, data_path):
        if self.name == "train":
            X, y, _, _ = fetch_iamondb(data_path)
            print("train")
            print(len(X))
            print(len(y))
        elif self.name == "valid":
            _, _, X, y = fetch_iamondb(data_path)
            print("valid")
            print(len(X))
            print(len(y))
        raw_X = X
        raw_X0 = []
        offset = True
        raw_new_X = []
        for item in raw_X:
            if offset:
                raw_X0.append(item[1:, 0])
                raw_new_X.append(item[1:, 1:] - item[:-1, 1:])
            else:
                raw_X0.append(item[:, 0])
                raw_new_X.append(item[:, 1:])
        raw_new_X, self.X_mean, self.X_std = self.global_normalize(raw_new_X)
        new_x = []
        for n in range(raw_new_X.shape[0]):
            new_x.append(np.concatenate((raw_X0[n][:, None], raw_new_X[n]),
                                        axis=-1).astype(theano.config.floatX))
        new_x = np.array(new_x)
        if self.prep == 'none':
            X = np.array(raw_X)
        if self.prep == 'normalize':
            X = new_x
            print X[0].shape
        elif self.prep == 'standardize':
            X, self.X_max, self.X_min = self.standardize(raw_X)
        self.labels = [np.array(y)]
        return [X]

    def theano_vars(self):
        if self.cond:
            return [T.tensor3('x', dtype=theano.config.floatX),
                    T.matrix('x_mask', dtype=theano.config.floatX),
                    T.tensor3('y', dtype=theano.config.floatX),
                    T.matrix('y_mask', dtype=theano.config.floatX)]
        else:
            return [T.tensor3('x', dtype=theano.config.floatX),
                    T.matrix('x_mask', dtype=theano.config.floatX)]

    def test_theano_vars(self):
        return [T.matrix('x', dtype=theano.config.floatX)]

    def slices(self, start, end):
        batches = [mat[start:end] for mat in self.data]
        label_batches = [mat[start:end] for mat in self.labels]
        mask = self.create_mask(batches[0].swapaxes(0, 1))
        batches = [self.zero_pad(batch) for batch in batches]
        label_mask = self.create_mask(label_batches[0].swapaxes(0, 1))
        label_batches = [self.zero_pad(batch) for batch in label_batches]
        if self.cond:
            return totuple([batches[0], mask, label_batches[0], label_mask])
        else:
            return totuple([batches[0], mask])

    def generate_index(self, X):
        maxlen = np.array([len(x) for x in X]).max()
        idx = np.arange(maxlen)
        return idx


if __name__ == "__main__":
    save_name = 'ground_truth_'
    save_path = '/u/kratarth/Documents/RNN_with_MNIST/NIPS/sk/datasets/handwriting/ground_truth/'
    data_path = '/data/lisatmp3/iamondb/'
    Pain = I(name='handwriting_train',
                        prep = 'normalize',
                        cond = True,
                        path=data_path)

    batch = Pain.slices(start=0, end=10826)
    X = Pain.data[0]
    sub_X = X
    for item in X:
        max_x = np.max(item[:,1])
        max_y = np.max(item[:,2])
        min_x = np.min(item[:,1])
        min_y = np.min(item[:,2])
    print np.max(max_x)
    print np.max(max_y)
    print np.min(min_x)
    print np.min(min_y)
    '''
    for i, x in enumerate(sub_X):
        sample = x.flatten()
        wavepath = save_path + save_name + str(i) + '.wav'
        wavfile.write(wavepath, 16000, np.int16(sample))
        samplepath = save_path + save_name + str(i) + '.png'
        fig = plt.figure()
        plt.plot(sample)
        plt.savefig(samplepath, bbox_inches='tight', format='png')
    '''
    ipdb.set_trace()


    def theano_vars(self):
        return T.tensor3('x', dtype=theano.config.floatX)

    def test_theano_vars(self):
        return T.matrix('x', dtype=theano.config.floatX)

    def slices(self, start, end):
        batch = np.array(self.data[self.idx[start:end]], dtype=theano.config.floatX)
        batch -= self.X_mean
        batch /= self.X_std
        batch = np.asarray([segment_axis(x, self.inpsz, 0) for x in batch])
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
