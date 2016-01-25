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
from sk.datasets.accent_hdf5 import fetch_accent_tbptt

from scipy.io import wavfile


class Accent_h5(TemporalSeries, SequentialPrepMixin):
    """
    Blizzard dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 X_mean=None,
                 X_std=None,
                 shuffle=0,
                 seq_len=8000,
                 use_window=0,
                 use_spec=0,
                 frame_size=200,
                 file_name='accent_tbptt',
                 batch_size=64,
                 range_start=0,
                 range_end=None,
                 **kwargs):
        self.X_mean = X_mean
        self.X_std = X_std
        self.shuffle = shuffle
        self.seq_len = seq_len
        self.use_window = use_window
        self.use_spec = use_spec
        self.frame_size = frame_size
        self.file_name = file_name
        if self.use_window:
            if self.use_spec:
                if not is_power2(self.frame_size):
                     raise ValueError("Provide a number which is power of 2,\
                                       for fast speed of DFT.")
            if np.mod(self.frame_size, 2)==0:
                self.overlap = self.frame_size / 2
            else:
                self.overlap = (self.frame_size - 1) / 2
            self.window = signal.hann(self.frame_size)[None, :].astype(theano.config.floatX)
        self.batch_size = batch_size
        self.range_start = range_start
        self.range_end = range_end
        super(Accent_h5, self).__init__(**kwargs)

    def load(self, data_path):
        X = fetch_accent_tbptt(data_path, self.seq_len, self.batch_size,
                               file_name=self.file_name+'.h5')
        if self.X_mean is None or self.X_std is None:
            prev_mean = None
            prev_var = None
            n_seen = 0
            n_inter = 10000
            range_start = self.range_start
            if self.range_end is not None:
                range_end = np.int(np.ceil(self.range_end / float(n_inter)))
            else:
                range_end = np.int(np.ceil(len(X) / float(n_inter)))
            for i in xrange(range_start, range_end):
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
                print "[%d / %d]" % (i+1, range_end)
            save_file_name = self.file_name + '_normal.npz'
            self.X_mean = prev_mean
            self.X_std = np.sqrt(prev_var / n_seen)
            np.savez(data_path + save_file_name, X_mean=self.X_mean, X_std=self.X_std)
        return X

    def theano_vars(self):
        return T.tensor3('x', dtype=theano.config.floatX)

    def test_theano_vars(self):
        return T.matrix('x', dtype=theano.config.floatX)

    def slices(self, start, end):
        batch = np.array(self.data[start:end], dtype=theano.config.floatX)
        if self.use_spec:
            batch = self._use_spec(batch)
            batch = self._log_magnitude(batch)
            batch = self._concatenate(batch)
        else:
            batch -= self.X_mean
            batch /= self.X_std
            if self.use_window:
                batch = self._use_window(batch)
            else:
                batch = np.asarray([segment_axis(x, self.frame_size, 0) for x in batch])
        batch = batch.transpose(1, 0, 2)
        return totuple(batch)

    def _use_window(self, batch):
        batch = np.asarray([self.window * segment_axis(x, self.frame_size,
                                                       self.overlap, end='pad')
                            for x in batch])
        return batch
 
    def _use_spec(self, batch):
        batch = np.asarray([self.numpy_rfft(self.window *
                                            segment_axis(x, self.frame_size,
                                                         self.overlap, end='pad'))
                            for x in batch])
        return batch

    def _log_magnitude(self, batch):
        batch_shape = batch.shape
        batch_reshaped = batch.reshape((batch_shape[0]*
                                        batch_shape[1],
                                        batch_shape[2]))
        # Transform into polar domain (magnitude & phase)
        mag, phase = R2P(batch_reshaped)
        log_mag = np.log10(mag + 1.)
        # Transform back into complex domain (real & imag)
        batch_normalized = P2R(log_mag, phase)
        new_batch = batch_normalized.reshape((batch_shape[0],
                                              batch_shape[1],
                                              batch_shape[2]))
        return new_batch

    def _concatenate(self, batch):
        batch_shape = batch.shape
        batch_reshaped = batch.reshape((batch_shape[0]*
                                        batch_shape[1],
                                        batch_shape[2]))
        batch_concatenated = complex_to_real(batch_reshaped)
        new_batch = batch_concatenated.reshape((batch_shape[0],
                                                batch_shape[1],
                                                batch_concatenated.shape[-1]))
        new_batch = new_batch.astype(theano.config.floatX)
        return new_batch



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
    data_path = '/home/junyoung/data/accent/accent_speech/'
    frame_size = 200
    seq_len = 8000
    use_window = 0
    use_spec = 0
    file_name = 'accent_tbptt'
    batch_size = 128
    normal_params = np.load(data_path + file_name + '_normal.npz')
    X_mean = normal_params['X_mean']
    X_std = normal_params['X_std']
    #X_mean = None
    #X_std = None
    range_start = 0
    range_end = 100000
    accent = Accent_h5(name='train',
                       path=data_path,
                       frame_size=frame_size,
                       seq_len=seq_len,
                       use_window=use_window,
                       use_spec=use_spec,
                       file_name=file_name,
                       X_mean=X_mean,
                       X_std=X_std,
                       range_start=range_start,
                       range_end=range_end,
                       batch_size=batch_size)
    ipdb.set_trace()
