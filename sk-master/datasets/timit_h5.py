import ipdb
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import theano
import theano.tensor as T

from cle.cle.data import DesignMatrix, TemporalSeries
from cle.cle.data.prep import SequentialPrepMixin
from cle.cle.utils import segment_axis, tolist, totuple
from cle.cle.utils.op import overlap_sum, complex_to_real
from sk.datasets.timit_hdf5 import fetch_timit

from scipy.io import wavfile


class TIMIT_h5(TemporalSeries, SequentialPrepMixin):
    """
    TIMIT dataset batch provider

    Parameters
    ----------
    frame_size: int, optional
        Dimension of one frame. Defaut is 1.
    prep: str, w
        Preprocessing of the dataset. Default is `'normalize'`
        for global normalization.
    """
    def __init__(self, 
                 X_mean=None,
                 X_std=None,
                 shuffle=0,
                 use_n_gram=0,
                 use_window=0,
                 use_spec=0,
                 load_phonetic_label=0,
                 load_spk_info=0,
                 frame_size=200,
                 file_name="_timit",
                 **kwargs):
        self.X_mean = X_mean
        self.X_std = X_std
        self.shuffle = shuffle
        self.use_n_gram = use_n_gram
        self.use_window = use_window
        self.use_spec = use_spec
        self.load_phonetic_label = load_phonetic_label
        self.load_spk_info = load_spk_info
        self.frame_size = frame_size
        self.file_name = file_name
        if use_window:
            if self.use_spec:
                if not is_power2(self.frame_size):
                     raise ValueError("Provide a number which is power of 2,\
                                       for fast speed of DFT.")
            if np.mod(self.frame_size, 2)==0:
                self.overlap = self.frame_size / 2
            else:
                self.overlap = (self.frame_size - 1) / 2
            self.window = signal.hann(self.frame_size)[None, :].astype(theano.config.floatX)
        super(TIMIT_h5, self).__init__(**kwargs)

    def load(self, data_path):
        if self.name not in ['train', 'valid', 'test']:
            raise ValueError(self.name + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")
        X, y= fetch_timit(data_path, self.shuffle, self.frame_size, self.name,
                          self.use_n_gram, self.file_name + '.h5')
        if (self.X_mean is None or self.X_std is None) and not self.use_spec:
            prev_mean = None
            prev_var = None
            n_seen = 0
            n_inter = 1
            range_end = np.int(np.ceil(len(X) / float(n_inter)))
            for i in xrange(range_end):
                n_seen += 1
                i_start = i*n_inter
                i_end = min((i+1)*n_inter, len(X))
                if prev_mean is None:
                    prev_mean = np.array(X[i_start:i_end]).mean()
                    prev_var = 0.
                else:
                    curr_mean = prev_mean +\
                        (np.array(X[i_start:i_end]) - prev_mean).mean() / n_seen
                    curr_var = prev_var +\
                        (np.array((X[i_start:i_end]) - prev_mean) *\
                         np.array((X[i_start:i_end]) - curr_mean)).mean()
                    prev_mean = curr_mean
                    prev_var = curr_var
                print "[%d / %d]" % (i+1, range_end)
            save_file_name = self.name + self.file_name + '_normal.npz'
            self.X_mean = prev_mean
            self.X_std = np.sqrt(prev_var / n_seen)
            np.savez(data_path + save_file_name, X_mean=self.X_mean, X_std=self.X_std)
 
        if self.load_spk_info:
            spk = np.load(speaker_path)
            spk = spk[idx]
            S = np.zeros((len(spk), 630))
            for i, s in enumerate(spk):
                S[i, s] = 1

        if self.load_spk_info and self.load_phonetic_label:
            return [X, Y, S]
        elif self.load_spk_info and not self.load_phonetic_label:
            return [X, S]
        elif not self.load_spk_info and self.load_phonetic_label:
            return [X, Y]
        elif not self.load_spk_info and not self.load_phonetic_label:
            return [X]

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

    def assign_n_gram_per_frame(self, unseg_Y):
        # Resolve multi label issue
        seg_Y = []
        for y in unseg_Y:
            this_y = np.zeros((y.shape[0],))
            for i in xrange(len(y)):
                try:
                    card_y, cnt_y = np.unique(y[i], return_counts=True)
                except TypeError as e:
                    card_y, cnt_y = self.count_unique(y[i])
                if len(card_y) == 1:
                    this_y[i] = card_y[0]
                elif len(card_y) != 2:
                    idx = np.argmin(cnt_y)
                    this_y[i] = card_y[idx]
            seg_Y.append(this_y)
        previous_label = 0.
        Y = []
        for y in seg_Y:
            this_y = np.zeros((y.shape[0], self.len_pho*5))
            global_cnt = 0
            while global_cnt != len(y) - 1:
                label_cnt = 1
                this_label = y[global_cnt]
                while this_label == y[global_cnt+label_cnt]:
                    if global_cnt+label_cnt == len(y) - 1:
                        break
                    else:
                        label_cnt += 1
                future_label = y[global_cnt+label_cnt]
                chunk_size = int(np.float(label_cnt/3.))
                start_end_idx = global_cnt + chunk_size
                middle_end_idx = start_end_idx + chunk_size
                this_y[global_cnt:global_cnt+label_cnt, int(previous_label)] = 1
                this_y[global_cnt:start_end_idx, int(self.len_pho+this_label)] = 1
                this_y[start_end_idx:middle_end_idx, int(self.len_pho*2+this_label)] = 1
                this_y[middle_end_idx:global_cnt+label_cnt, int(self.len_pho*3+this_label)] = 1
                this_y[global_cnt:global_cnt+label_cnt, int(self.len_pho*4+future_label)] = 1
                global_cnt += label_cnt
            Y.append(this_y)
        Y = np.array(Y)
        return Y

    def assign_phoneme_per_frame(self, unseg_Y):
        Y = []
        for y in unseg_Y:
            # Reflecting A, A_start, A_end
            this_y = np.zeros((y.shape[0], self.len_pho*3))
            for i in xrange(len(y)):
                try:
                    card_y, cnt_y = np.unique(y[i], return_counts=True)
                except TypeError as e:
                    card_y, cnt_y = self.count_unique(y[i])
                if len(card_y) == 1:
                    this_y[i, card_y[0]] = 1
                elif len(card_y) == 2:
                    idx = np.argmax(cnt_y)
                    if idx == 0:
                        this_y[i, card_y[idx]+self.len_pho] = 1
                    elif idx == 1:
                        this_y[i, card_y[idx]+self.len_pho*2] = 1
                else:
                    idx = np.argmax(cnt_y)
                    this_y[i, card_y[idx]] = 1
            Y.append(this_y)
        Y = np.array(Y)
        return Y

    def count_unique(self, keys):
        uniq_keys = np.unique(keys)
        bins = uniq_keys.searchsorted(keys)
        return uniq_keys, np.bincount(bins)

    def theano_vars(self):
        if self.load_spk_info and self.load_phonetic_label:
            return [T.tensor3('x', dtype=theano.config.floatX),
                    T.tensor3('y', dtype=theano.config.floatX),
                    T.matrix('spk', dtype=theano.config.floatX),
                    T.matrix('x_mask', dtype=theano.config.floatX)]
        elif self.load_spk_info and not self.load_phonetic_label:
            return [T.tensor3('x', dtype=theano.config.floatX),
                    T.matrix('spk', dtype=theano.config.floatX),
                    T.matrix('x_mask', dtype=theano.config.floatX)]
        elif not self.load_spk_info and self.load_phonetic_label:
            return [T.tensor3('x', dtype=theano.config.floatX),
                    T.tensor3('y', dtype=theano.config.floatX),
                    T.matrix('x_mask', dtype=theano.config.floatX)]
        elif not self.load_spk_info and not self.load_phonetic_label:
            return [T.tensor3('x', dtype=theano.config.floatX),
                    T.matrix('x_mask', dtype=theano.config.floatX)]

    def slices(self, start, end):
        batches = [mat[start:end] for mat in self.data]
        mask = self.create_mask(batches[0].swapaxes(0, 1))
        if self.load_spk_info:
            batches = [self.zero_pad(batch) for batch in batches[:-1]]
            spk = batches[-1]
            return totuple(batches + [spk, mask])
        else:
            batches = [self.zero_pad(batch) for batch in batches]
            return totuple(batches + [mask])

    def slices(self, start, end):
        batches = [mat[start:end] for mat in self.data]
        if self.use_spec:
            batches[0] = self._use_spec(batches[0])
            batches[0] = self._log_magnitude(batches[0])
            batches[0] = self._concatenate(batches[0])
        else:
            batches[0] -= self.X_mean
            batches[0] /= self.X_std
            if self.use_window:
                batches[0] = self._use_window(batches[0])
            else:
                batches[0] = np.asarray([segment_axis(x, self.frame_size, 0) for x in batches[0]])
        mask = self.create_mask(batches[0].swapaxes(0, 1))
        if self.load_spk_info:
            batches = [self.zero_pad(batch) for batch in batches[:-1]]
            spk = batches[-1]
            return totuple(batches + [spk, mask])
        else:
            batches = [self.zero_pad(batch) for batch in batches]
            return totuple(batches + [mask])


def P2R(magnitude, phase):
    return magnitude * np.exp(1j*phase)


def R2P(x):
    return np.abs(x), np.angle(x)


def is_power2(num):
    """
    States if a number is a power of two (Author: A.Polino)
    """
    return num != 0 and ((num & (num - 1)) == 0)


if __name__ == "__main__":
    save_name = 'ground_truth_'
    save_path = '/home/junyoung/repos/sk/saved/rnnvae/data/timit/ground_truth/'
    data_path = '/home/junyoung/data/timit/readable/'
    frame_size = 200
    use_n_gram = 1

    timit_data = TIMIT(name='valid',
                       path=data_path,
                       frame_size=frame_size,
                       shuffle=0,
                       use_n_gram=1,
                       use_window=0,
                       use_spec=0)

    batch = timit_data.slices(start=0, end=10)
    X_mean = timit_data.X_mean
    X_std = timit_data.X_std
    X = timit_data.data[0]
    sub_X = X[:100]
    sub_X = sub_X * X_std + X_mean
    for i, x in enumerate(sub_X):
        sample = x.flatten()
        wave_path = save_path + save_name + str(i) + '.wav'
        wavfile.write(wave_path, 16000, np.int16(sample))
        sample_path = save_path + save_name + str(i) + '.png'
        fig = plt.figure()
        plt.plot(sample)
        plt.savefig(sample_path, bbox_inches='tight', format='png')
        plt.close()
