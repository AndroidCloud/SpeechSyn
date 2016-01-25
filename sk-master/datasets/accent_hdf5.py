# -*- coding: utf 8 -*-
from __future__ import division
import ipdb
import os
import numpy as np
import tables
import numbers
import fnmatch

from cle.cle.utils import segment_axis
from scipy.io import wavfile


def fetch_accent(data_path, shuffle=0, sz=32000, file_name="full_accent.h5"):
    hdf5_path = os.path.join(data_path, file_name)
    random_state = np.random.RandomState(1999)
    if shuffle:
        hdf5_path = os.path.join(data_path, file_name[:-3] + "_shuffle.h5")
    if not os.path.exists(hdf5_path):
        data_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, '*.wav'):
                data_matches.append(os.path.join(root, filename))
        # Just group same languages, numbering will be in *alpha* not numeric
        # order within each language
        data_matches = sorted(data_matches)
        if shuffle:
            data_matches = random_state.shuffle(data_matches)
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        data = hdf5_file.createEArray(hdf5_file.root, 'data',
                                      tables.Int16Atom(),
                                      shape=(0, sz),
                                      filters=compression_filter,)
        for n, f in enumerate(data_matches):
            print("Processing file %i of %i" % (n, len(data_matches)))
            sr, d = wavfile.read(f)
            e = [r for r in range(0, len(d), sz)]
            e.append(None)
            starts = e[:-1]
            stops = e[1:]
            endpoints = zip(starts, stops)
            for i, j in endpoints:
                d_new = d[i:j]
                # zero pad
                if len(d_new) < sz:
                    d_large = np.zeros((sz,), dtype='int16')
                    d_large[:len(d_new)] = d_new
                    d_new = d_large
                data.append(d_new[None])
        hdf5_file.close()
    hdf5_file = tables.openFile(hdf5_path, mode='r')
    data = hdf5_file.root.data
    X = data
    return X


def fetch_accent_tbptt(data_path, sz=8000, batch_size=100,
                       file_name="accent_tbptt.h5"):
    hdf5_path = os.path.join(data_path, file_name)
    if not os.path.exists(hdf5_path):
        data_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, '*.wav'):
                if '._' not in filename:
                    data_matches.append(os.path.join(root, filename))
        # Just group same languages, numbering will be in *alpha* not numeric
        # order within each language
        data_matches = sorted(data_matches)
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        data = hdf5_file.createEArray(hdf5_file.root, 'data',
                                      tables.Int16Atom(),
                                      shape=(0, sz),
                                      filters=compression_filter,)
        large_d = None
        for n, f in enumerate(data_matches):
            print("Processing file %i of %i" % (n+1, len(data_matches)))
            try:
                sr, d = wavfile.read(f)
                if len(d.shape) > 1:
                    d = d[:, 0]
                if large_d is None:
                    large_d = d
                else:
                    large_d = np.concatenate([large_d, d])
            except ValueError:
                print("Not a proper wave file.")
        chunk_size = int(np.float(len(large_d) / batch_size))
        seg_d = segment_axis(large_d, chunk_size, 0)
        num_batch = int(np.float((seg_d.shape[-1] - 1)/float(sz)))
        for i in range(num_batch):
            this_batch = seg_d[:, i*sz:(i+1)*sz]
            for j in range(batch_size):
                data.append(this_batch[j][None])
        hdf5_file.close()
    hdf5_file = tables.openFile(hdf5_path, mode='r')
    data = hdf5_file.root.data
    X = data
    return X


if __name__ == "__main__":
    X = fetch_accent()
    from IPython import embed; embed()
    raise ValueError()
