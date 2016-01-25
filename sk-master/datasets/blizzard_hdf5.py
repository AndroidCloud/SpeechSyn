# -*- coding: utf 8 -*-
from __future__ import division
import ipdb
import os
import numpy as np
import tables
import numbers
import fnmatch
import scipy.signal
import theano

from cle.cle.utils import segment_axis

# A trick for monkeypatching an instancemethod that when method is a
# c-extension? there must be a better way


class _blizzardEArray(tables.EArray):
    pass


def fetch_blizzard(data_path, shuffle=0, sz=32000, file_name="full_blizzard.h5"):
    hdf5_path = os.path.join(data_path, file_name)
    if not os.path.exists(hdf5_path):
        data_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, 'data_*.npy'):
                data_matches.append(os.path.join(root, filename))
        # sort in proper order
        data_matches = sorted(data_matches,
                              key=lambda x: int(
                                  x.split("/")[-1].split("_")[-1][0]))

        # setup tables
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        data = hdf5_file.createEArray(hdf5_file.root, 'data',
                                      tables.Int16Atom(),
                                      shape=(0, sz),
                                      filters=compression_filter,)

        for n, f in enumerate(data_matches):
            print("Reading file %s" % (f))
            with open(f) as fp:
                # Array of arrays, ragged
                d = np.load(fp)
                if shuffle:
                    rnd_idx = np.random.permutation(len(d))
                    d = d[rnd_idx]
                for n, di in enumerate(d):
                    print("Processing line %i of %i" % (n+1, len(d)))
                    # Some of these are stereo??? wtf
                    if len(di.shape) > 1:
                        di = di[:, 0]
                    e = [r for r in range(0, len(di), sz)]
                    e.append(None)
                    starts = e[:-1]
                    stops = e[1:]
                    endpoints = zip(starts, stops)
                    #if shuffle:
                    #    rnd_idx = np.random.permutation(len(endpoints))
                    #    endpoints = endpoints[rnd_idx]
                    for i, j in endpoints:
                        di_new = di[i:j]
                        # zero pad
                        if len(di_new) < sz:
                            di_large = np.zeros((sz,), dtype='int16')
                            di_large[:len(di_new)] = di_new
                            di_new = di_large
                        data.append(di_new[None])
        hdf5_file.close()
    hdf5_file = tables.openFile(hdf5_path, mode='r')
    data = hdf5_file.root.data
    X = data
    return X


def fetch_blizzard_tbptt(data_path, sz=8000, batch_size=100,
                         file_name="blizzard_tbptt.h5"):
    hdf5_path = os.path.join(data_path, file_name)
    if not os.path.exists(hdf5_path):
        data_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, 'data_*.npy'):
                data_matches.append(os.path.join(root, filename))
        # sort in proper order
        data_matches = sorted(data_matches,
                              key=lambda x: int(
                                  x.split("/")[-1].split("_")[-1][0]))

        # setup tables
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        data = hdf5_file.createEArray(hdf5_file.root, 'data',
                                      tables.Int16Atom(),
                                      shape=(0, sz),
                                      filters=compression_filter,)

        for n, f in enumerate(data_matches):
            print("Reading file %s" % (f))
            with open(f) as fp:
                # Array of arrays, ragged
                d = np.load(fp)
                large_d = d[0]
                for i in xrange(1, len(d)):
                    print("Processing line %i of %i" % (i+1, len(d)))
                    di = d[i]
                    if len(di.shape) > 1:
                        di = di[:, 0]
                    large_d = np.concatenate([large_d, di])
                chunk_size = int(np.float(len(large_d) / batch_size))
                seg_d = segment_axis(large_d, chunk_size, 0)
                num_batch = int(np.float((seg_d.shape[-1] - 1)/float(sz)))
                for i in range(num_batch):
                    batch = seg_d[:, i*sz:(i+1)*sz]
                    for j in range(batch_size):
                        data.append(batch[j][None])
        hdf5_file.close()
    hdf5_file = tables.openFile(hdf5_path, mode='r')
    data = hdf5_file.root.data
    X = data
    return X


def fetch_blizzard_unify_spec(data_path, sz=8000, timestep=79, frame_size=200, overlap=100,
                                batch_size=100, file_name="blizzard_unify_spec.h5"):
    hdf5_path = os.path.join(data_path, file_name)
    if not os.path.exists(hdf5_path):
        data_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, 'data_*.npy'):
                data_matches.append(os.path.join(root, filename))
        # sort in proper order
        data_matches = sorted(data_matches,
                              key=lambda x: int(
                                  x.split("/")[-1].split("_")[-1][0]))

        # setup tables
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        data = hdf5_file.createEArray(hdf5_file.root, 'data',
                                      tables.Int16Atom(),
                                      shape=(0, timestep, frame_size),
                                      filters=compression_filter,)

        for n, f in enumerate(data_matches):
            print("Reading file %s" % (f))
            with open(f) as fp:
                # Array of arrays, ragged
                d = np.load(fp)
                large_d = d[0]
                for i in xrange(1, len(d)):
                    print("Processing line %i of %i" % (i+1, len(d)))
                    di = d[i]
                    if len(di.shape) > 1:
                        di = di[:, 0]
                    large_d = np.concatenate([large_d, di])
                chunk_size = int(np.float(len(large_d) / batch_size))
                seg_d = segment_axis(large_d, chunk_size, 0)
                num_batch = int(np.float((seg_d.shape[-1] - 1)/float(sz)))
                for i in range(num_batch):
                    batch = seg_d[:, i*sz:(i+1)*sz]
                    batch = np.array([segment_axis(x, frame_size, overlap,
                                                   end='pad') for x in batch])
                    batch = apply_window(batch)
                    batch = apply_fft(batch)
                    batch = log_magnitude(batch)
                    batch = apply_ifft(batch)
                    for j in range(batch_size):
                        data.append(batch[j][None])
        hdf5_file.close()
    hdf5_file = tables.openFile(hdf5_path, mode='r')
    data = hdf5_file.root.data
    X = data
    return X


def apply_window(batch):
    window = scipy.signal.hann(batch.shape[-1])
    batch = np.asarray([window * example for example in batch], dtype=theano.config.floatX)
    return batch


def apply_fft(batch):
    batch = np.asarray([np.fft.rfft(example) for example in batch], dtype=theano.config.floatX)
    return batch


def apply_ifft(batch):
    new_batch = np.asarray([np.fft.irfft(example) for example in batch], dtype=theano.config.floatX)
    return new_batch


def log_magnitude(batch):
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


def P2R(magnitude, phase):
    return magnitude * np.exp(1j*phase)


def R2P(x):
    return np.abs(x), np.angle(x)


if __name__ == "__main__":
    data_path = '/raid/chungjun/data/blizzard/'
    X = fetch_blizzard(data_path, 1)
    from IPython import embed; embed()
    raise ValueError()
