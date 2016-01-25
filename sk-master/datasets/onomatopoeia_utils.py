# -*- coding: utf 8 -*-
from __future__ import division
import os
import numpy as np
import fnmatch
from scipy.io import wavfile
import cPickle
import wave

# wavio.py
# Author: Warren Weckesser
# License: BSD 3-Clause (http://opensource.org/licenses/BSD-3-Clause)
def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.fromstring(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


def readwav(file):
    """
    Read a wav file.

    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.

    This function does not read compressed wav files.
    """
    wav = wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)
    return rate, sampwidth, array


def writewav24(filename, rate, data):
    """Create a 24 bit wav file.

    data must be "array-like", either 1- or 2-dimensional.  If it is 2-d,
    the rows are the frames (i.e. samples) and the columns are the channels.

    The data is assumed to be signed, and the values are assumed to be
    within the range of a 24 bit integer.  Floating point values are
    converted to integers.  The data is not rescaled or normalized before
    writing it to the file.

    Example: Create a 3 second 440 Hz sine wave.

    >>> rate = 22050  # samples per second
    >>> T = 3         # sample duration (seconds)
    >>> f = 440.0     # sound frequency (Hz)
    >>> t = np.linspace(0, T, T*rate, endpoint=False)
    >>> x = (2**23 - 1) * np.sin(2 * np.pi * f * t)
    >>> writewav24("sine24.wav", rate, x)

    """
    a32 = np.asarray(data, dtype=np.int32)
    if a32.ndim == 1:
        # Convert to a 2D array with a single column.
        a32.shape = a32.shape + (1,)
    # By shifting first 0 bits, then 8, then 16, the resulting output
    # is 24 bit little-endian.
    a8 = (a32.reshape(a32.shape + (1,)) >> np.array([0, 8, 16])) & 255
    wavdata = a8.astype(np.uint8).tostring()

    w = wave.open(filename, 'wb')
    w.setnchannels(a32.shape[1])
    w.setsampwidth(3)
    w.setframerate(rate)
    w.writeframes(wavdata)
    w.close()

# all_tags = set([tag for d in file_names for tag in d.split("_")
#                 if "@" not in tag and not tag[-1].isdigit()])
all_tags = ['GSP', 'DZD', 'AGH', 'MAFR', 'DRT', 'BONNET', 'EXT', 'EXR',
            'BURGESS', 'BOO', 'PAI', 'AGR', 'PAN', 'SPR', 'WTF', 'TCK', 'CHC',
            'CHD', 'CHE', 'WRN', 'CHK', 'THR', 'ABONNY', 'ROBERTS', 'AHTABAI',
            'LOOK', 'APR', 'PYR', 'BLCKBRD', 'APO', 'FFH', 'HORNGLD', 'MENG',
            'GBD', 'DTH', 'PAIN', 'EMO', 'FLT', 'SKE', 'HPX', 'PRINS', 'AMA',
            'FAFR', 'EKENWAY', 'SCR', 'SHT', 'TIBURON', 'GRB', 'DSG', 'FOO',
            'AFR', 'FENG', 'MAJ', 'ANO', 'BREATH', 'ADEWALE', 'BLOC', 'UF',
            'TORRES', 'STR', 'HNG', 'RHONAD', 'SUP', 'COM', 'ASS', 'CGH', 'EFF',
            'CUR', 'WALPOLE', 'UNC', 'WAT', 'MED', 'ROGERS', 'RLF', 'MREAD',
            'HVY', 'NTL', 'SA', 'RSF', 'GNT', 'DIE', 'LOS', 'DUCASSE', 'LOW',
            'BAL', 'DIV', 'DIS', 'HIT', 'ONO', 'BIG', 'FAT', 'FAL', 'DRG',
            'CNT', 'KNE', 'AI', 'SLW', 'CNF', 'HGH', 'LGH', 'NRM', 'TFG', 'BOF',
            'LGT']

def fetch_onomatopoeia(data_path, tag_match="all"):
    ono_path = os.path.join(data_path, 'all_ono.npy')
    tags_path = os.path.join(data_path, 'all_ono_tags.pkl')
    if not os.path.exists(ono_path):
        # Converted to 16k, 16bit wav
        # original /data/lisa/data/ubi/speech/onomatopoeia/dataset/Audio Files
        audio_path = "/data/lisatmp3/kastner/ubi_data"
        data_matches = []
        for root, dirnames, filenames in os.walk(audio_path):
            for filename in fnmatch.filter(filenames, '*_*.wav'):
                data_matches.append(os.path.join(root, filename))

        file_names = [d.split("/")[-1] for d in data_matches]
        file_names = [d.replace("M_READ", "MREAD") for d in file_names]
        file_names = [d.replace("F_ENG", "FENG") for d in file_names]
        file_names = [d.replace("F_AFR", "FAFR") for d in file_names]
        file_names = [d.replace("M_ENG", "MENG") for d in file_names]
        file_names = [d.replace("M_AFR", "MAFR") for d in file_names]
        matching_tags = [[tag for tag in d.split("_")
                          if "@" not in tag and not tag[-1].isdigit()]
                         for d in file_names]
        audio = []
        for n, f in enumerate(data_matches):
            sr, d = wavfile.read(f)
            audio.append(d)
        f = open(tags_path, mode="wb")
        cPickle.dump(matching_tags, f)
        f.close()
        np.save(ono_path, np.array(audio))
    f = open(tags_path, mode="rb")
    matching_tags = cPickle.load(f)
    f.close()
    X = np.load(ono_path)
    if tag_match != "all":
        if type(tag_match) is list:
            for tag in tag_match:
                assert tag in all_tags
        else:
            assert tag_match in all_tags
        match_slicer = np.array([n for n, t in enumerate(matching_tags)
                                 if tag_match in t])
        X = X[match_slicer]
        matching_tags = [t for t in matching_tags if tag_match in t]
    return X, matching_tags
