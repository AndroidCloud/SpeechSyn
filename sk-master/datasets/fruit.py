import ipdb
import fnmatch
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import tables
import tarfile
import theano
import theano.tensor as T

from cle.cle.data import DesignMatrix, TemporalSeries
from cle.cle.data.prep import SequentialPrepMixin
from cle.cle.utils import segment_axis, tolist, totuple
from cle.cle.utils.op import overlap_sum

from scipy import signal
from scipy.io import wavfile


class Fruit(TemporalSeries, SequentialPrepMixin):
    """
    Fruit dataset batch provider

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 frame_size=1,
                 prep='normalize',
                 **kwargs):
        self.frame_size = frame_size
        self.prep = prep
        super(Fruit, self).__init__(**kwargs)

    def load(self, data_path):
        dataset = 'audio.tar.gz'
        datafile = os.path.join(data_path, dataset)
        if not os.path.isfile(datafile):
            try:
                import urllib
                urllib.urlretrieve('http://google.com')
                url =\
                    'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
            except AttributeError:
                import urllib.request as urllib
                url =\
                    'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
            print("Downloading data from %s" % url)
            urllib.urlretrieve(url, datafile)
        if not os.path.exists(os.path.join(data_path, "audio")):
            tar = tarfile.open(datafile)
            os.chdir(data_path)
            tar.extractall()
            tar.close()
        h5_file_path = os.path.join(data_path, "saved_fruit.h5")
        if not os.path.exists(h5_file_path):
            data_path = os.path.join(data_path, "audio")
            audio_matches = []
            for root, dirnames, filenames in os.walk(data_path):
                for filename in fnmatch.filter(filenames, '*.wav'):
                    audio_matches.append(os.path.join(root, filename))
            random.seed(1999)
            random.shuffle(audio_matches)
            # http://mail.scipy.org/pipermail/numpy-discussion/2011-March/055219.html
            h5_file = tables.openFile(h5_file_path, mode='w')
            data_x = h5_file.createVLArray(h5_file.root, 'data_x',
                                           tables.Float32Atom(shape=()),
                                           filters=tables.Filters(1))
            data_y = h5_file.createVLArray(h5_file.root, 'data_y',
                                           tables.Int32Atom(shape=()),
                                           filters=tables.Filters(1))
            for wav_path in audio_matches:
                # Convert chars to int classes
                word = wav_path.split(os.sep)[-1][:6]
                chars = [ord(c) - 97 for c in word]
                data_y.append(np.array(chars, dtype='int32'))
                fs, d = wavfile.read(wav_path)
                data_x.append(d.astype(theano.config.floatX))
            h5_file.close()
        h5_file = tables.openFile(h5_file_path, mode='r')
        raw_X = np.array([np.asarray(x) for x in h5_file.root.data_x])
        cls = np.array([''.join([chr(y+97) for y in Y]) for Y in h5_file.root.data_y])

        if self.name != 'all':
            fruit_list = []
            if len(self.name) > 1:
                for i, fruit_name in enumerate(cls):
                    for name in self.name:
                        if name in fruit_name:
                            fruit_list.append(i)
            else:
                for i, fruit_name in enumerate(cls):
                    if self.name in fruit_name:
                        fruit_list.append(i)
        else:
            fruit_list = tolist(np.arange(len(raw_X)))
        raw_X = raw_X[fruit_list]
        if self.prep == 'normalize':
            pre_X, self.X_mean, self.X_std = self.global_normalize(raw_X)
        elif self.prep == 'standardize':
            pre_X, self.X_max, self.X_min = self.standardize(raw_X)
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
