# -*- coding: utf 8 -*-
from __future__ import division
import ipdb
import os
import numpy as np
import tables
import numbers
import fnmatch

from cle.cle.utils import segment_axis

# A trick for monkeypatching an instancemethod that when method is a
# c-extension? there must be a better way


class _timitEArray(tables.EArray):
    pass


def fetch_timit(data_path, shuffle=0, frame_size=200, this_set="train",
                use_n_gram=1, file_name='_timit.h5'):
    file_name = this_set + file_name
    hdf5_path = os.path.join(data_path, file_name)
    if not os.path.exists(hdf5_path):
        raw_name = data_path + this_set + '_x_raw.npy'
        pho_name = data_path + this_set + '_x_phonemes.npy'
        raw_data = np.load(raw_name)
        pho_data = np.load(pho_name)
        if shuffle:
            idx = np.random.permutation(len(raw_data))
            raw_data = raw_data[idx]
            pho_data = pho_data[idx]
        len_pho = np.array([np.unique(x).max() for x in pho_data]).max() + 1
        pho_data = np.array([segment_axis(y, frame_size, 0) for y in pho_data])
        if use_n_gram:
            pho_data = assign_n_gram_per_frame(pho_data, len_pho)
        else:
            pho_data = assign_phoneme_per_frame(pho_data, len_pho)

        # setup tables
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        raw = hdf5_file.createVLArray(hdf5_file.root, 'raw',
                                      tables.Int16Atom(shape=()),
                                      filters=compression_filter,)
        pho = hdf5_file.createVLArray(hdf5_file.root, 'pho',
                                           tables.Int16Atom(shape=()),
                                           filters=compression_filter,)
        for x, y in zip(raw_data, pho_data):
            raw.append(x)
            pho.append(y.flatten())
        hdf5_file.close()
    hdf5_file = tables.openFile(hdf5_path, mode='r')
    X = hdf5_file.root.raw
    y = hdf5_file.root.pho
    return X, y


def assign_n_gram_per_frame(unseg_Y, len_pho):
    # Resolve multi label issue
    seg_Y = []
    for y in unseg_Y:
        this_y = np.zeros((y.shape[0],))
        for i in xrange(len(y)):
            try:
                card_y, cnt_y = np.unique(y[i], return_counts=True)
            except TypeError as e:
                card_y, cnt_y = count_unique(y[i])
            if len(card_y) == 1:
                this_y[i] = card_y[0]
            elif len(card_y) != 2:
                idx = np.argmin(cnt_y)
                this_y[i] = card_y[idx]
        seg_Y.append(this_y)
    previous_label = 0.
    Y = []
    for y in seg_Y:
        this_y = np.zeros((y.shape[0], len_pho*5))
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
            this_y[global_cnt:start_end_idx, int(len_pho+this_label)] = 1
            this_y[start_end_idx:middle_end_idx, int(len_pho*2+this_label)] = 1
            this_y[middle_end_idx:global_cnt+label_cnt, int(len_pho*3+this_label)] = 1
            this_y[global_cnt:global_cnt+label_cnt, int(len_pho*4+future_label)] = 1
            global_cnt += label_cnt
        Y.append(this_y)
    Y = np.array(Y)
    return Y


def assign_phoneme_per_frame(unseg_Y, len_pho):
    Y = []
    for y in unseg_Y:
        # Reflecting A, A_start, A_end
        this_y = np.zeros((y.shape[0], len_pho*3))
        for i in xrange(len(y)):
            try:
                card_y, cnt_y = np.unique(y[i], return_counts=True)
            except TypeError as e:
                card_y, cnt_y = count_unique(y[i])
            if len(card_y) == 1:
                this_y[i, card_y[0]] = 1
            elif len(card_y) == 2:
                idx = np.argmax(cnt_y)
                if idx == 0:
                    this_y[i, card_y[idx]+len_pho] = 1
                elif idx == 1:
                    this_y[i, card_y[idx]+len_pho*2] = 1
            else:
                idx = np.argmax(cnt_y)
                this_y[i, card_y[idx]] = 1
        Y.append(this_y)
    Y = np.array(Y)
    return Y


def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)


if __name__ == "__main__":
    data_path = '/home/junyoung/data/timit/readable/'
    X = fetch_timit(data_path, 1)
    from IPython import embed; embed()
    raise ValueError()
