#!/usr/bin/env python

import cPickle as pickle
import numpy as np
import os
import sys
import gzip

import random

def dostuff(inpath):

    data_files = [f for f in os.listdir(inpath) if f.endswith('.mel.npy')]
    random.shuffle(data_files)
    # data_files = data_files[:10]
    artists = set([f[:18] for f in data_files])
    artist_string_to_id = dict([(s,i) for i, s in enumerate(artists)])

    train_set_y = np.hstack([[artist_string_to_id[f[:18]]] * 129 for f in data_files[:250]])
    train_set_x = np.vstack([np.load(os.path.join(inpath, f)) for f in data_files[:250]])

    test_set_y = np.hstack([[artist_string_to_id[f[:18]]] * 129 for f in data_files[250:350]])
    test_set_x = np.vstack([np.load(os.path.join(inpath, f)) for f in data_files[250:350]])

    validation_set_y = np.hstack([[artist_string_to_id[f[:18]]] * 129 for f in data_files[350:]])
    validation_set_x = np.vstack([np.load(os.path.join(inpath, f)) for f in data_files[350:]])

    datasets = [(train_set_x, train_set_y), (validation_set_x, validation_set_y), (test_set_x, test_set_y)]
    return datasets


if __name__ == '__main__':
    datasets = dostuff(sys.argv[1])

    with open(sys.argv[2], 'w') as f:
        pickle.dump(datasets, f)
