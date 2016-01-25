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
from cle.cle.utils.op import overlap_sum, complex_to_real

from scipy import signal
from scipy.io import wavfile


class TIMIT(TemporalSeries, SequentialPrepMixin):
    """
    TIMIT dataset batch provider

    Parameters
    ----------
    frame_size: int, optional
        Dimension of one frame. Defaut is 1.
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
        super(TIMIT, self).__init__(**kwargs)

    def load(self, data_path):
        if self.name not in ['train', 'valid', 'test']:
            raise ValueError(self.name + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")

        speaker_info_list_path = os.path.join(data_path, 'spkrinfo.npy')
        #phoneme_list_path = os.path.join(data_path, 'reduced_phonemes.pkl')
        #word_list_path = os.path.join(data_path, 'words.pkl')
        #speaker_features_list_path = os.path.join(data_path,
        #                                          'spkr_feature_names.pkl')
        speaker_id_list_path = os.path.join(data_path, 'speakers_ids.pkl')
        raw_path = os.path.join(data_path, self.name + '_x_raw.npy')
        phoneme_path = os.path.join(data_path, self.name + '_x_phonemes.npy')
        #phone_path = os.path.join(data_path, self.name + '_x_phones.npy')
        #word_path = os.path.join(data_path, self.name + '_x_words.npy')
        speaker_path = os.path.join(data_path, self.name + '_spkr.npy')

        raw = np.load(raw_path)
        raw_X = []
        for x in raw:
            raw_X.append(np.asarray(x, dtype=theano.config.floatX))
        raw_X = np.array(raw_X)

        if self.shuffle:
            idx = np.random.permutation(len(raw_X))
            raw_X = raw_X[idx]
        else:
            idx = np.arange(len(raw_X))

        if not self.use_spec:
            pre_X, self.X_mean, self.X_std =\
                self.global_normalize(raw_X, self.X_mean, self.X_std)

        if self.use_window:
            if self.use_spec:
                X = self._use_spec(raw_X)
                X = self._log_magnitude(X)
                X = self._concatenate(X)
            else:
                X = self._use_window(pre_X)
        else:
            X = np.asarray([segment_axis(x, self.frame_size, 0) for x in pre_X])
 
        if self.load_spk_info:
            spk = np.load(speaker_path)
            spk = spk[idx]
            S = np.zeros((len(spk), 630))
            for i, s in enumerate(spk):
                S[i, s] = 1
 
        if self.load_phonetic_label:
            #pho = np.load(phone_path)
            pho = np.load(phoneme_path)
            self.len_pho = np.array([np.unique(x).max() for x in pho]).max() + 1
            unseg_Y = []
            for y in pho:
                unseg_Y.append(np.asarray(y, dtype=theano.config.floatX))
            unseg_Y = np.array(unseg_Y)
            unseg_Y = unseg_Y[idx]
            unseg_Y = np.array([segment_axis(y, self.frame_size, 0) for y in unseg_Y])
            if self.use_n_gram:
                Y = self.assign_n_gram_per_frame(unseg_Y)
            else:
                Y = self.assign_phoneme_per_frame(unseg_Y)

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
            return [T.ftensor3('x', dtype=theano.config.floatX),
                    T.ftensor3('y', dtype=theano.config.floatX),
                    T.fmatrix('spk', dtype=theano.config.floatX),
                    T.fmatrix('x_mask', dtype=theano.config.floatX)]
        elif self.load_spk_info and not self.load_phonetic_label:
            return [T.ftensor3('x'), T.fmatrix('spk'), T.fmatrix('x_mask')]
        elif not self.load_spk_info and self.load_phonetic_label:
            return [T.ftensor3('x'), T.ftensor3('y'), T.fmatrix('x_mask')]
        elif not self.load_spk_info and not self.load_phonetic_label:
            return [T.ftensor3('x'), T.fmatrix('x_mask')]

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

    train_data = TIMIT(name='train',
                        path=data_path,
                        frame_size=frame_size,
                        shuffle=0,
                        use_n_gram=1,
                        use_window=0,
                        use_spec=0)

    batch = train_data.slices(start=0, end=10)
    ipdb.set_trace()
    X_mean = train_data.X_mean
    X_std = train_data.X_std
    X = train_data.data[0]
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
