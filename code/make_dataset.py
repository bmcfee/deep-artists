#!/usr/bin/env python

import cPickle as pickle
import numpy as np
import numpy
import os
import sys
import gzip
import sklearn.cross_validation
import random



def filtered_stratified_split(ids, splitter, Y, **kwargs):
    '''Cross-validation split  filtration. Ensures that points  of the
    same meta-id end up on the same side of the split
    input:

        ids:         n-by-1 mapping of data points to meta-id
        splitter:    handle to the cross-validation class (eg, StratifiedShuffleSplit)
        Y:           n-by-1 vector of class labels
        **kwargs:    arguments to the cross-validation class

    yields:
        (train, test) indices
    '''

    n = len(Y)

    indices = ('indices' in kwargs) and (kwargs['indices'])

    kwargs['indices'] = True

    def unfold(meta_ids, X_id, indices):
        split_ids = []
        for i in meta_ids:
            split_ids.extend(X_id[i])

        split_ids = numpy.array(split_ids)

        if not indices:
            z = numpy.zeros(n, dtype=bool)
            z[split_ids] = True
            return z

    # 1: make a new label vector Yid
    X_id = []
    Y_id = []

    last_id = None
    for i in xrange(len(ids)):
        if i > 0 and last_id == ids[i]:
            X_id[-1].append(i)
        else:
            last_id = ids[i]
            X_id.append([i])
            Y_id.append(Y[i])

    # 2: CV split on Yid
    splits = splitter(Y_id, **kwargs)

    # 3: Map CV indices back to Y space
    for meta_train, meta_test in splits:
        yield (unfold(meta_train, X_id, indices),
               unfold(meta_test, X_id, indices))



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





def dobetterstuff(inpath):
    data_files = [f for f in os.listdir(inpath) if f.endswith('.mel.npy')]
    random.shuffle(data_files)
    artists = set([f[:18] for f in data_files])
    artist_string_to_id = dict([(s,i) for i, s in enumerate(artists)])

    def get_split(datafiles___, splitpercent):
        # gen = filtered_stratified_split(datafiles___,
        #                                 sklearn.cross_validation.StratifiedShuffleSplit,
        #                                 [1] * len(datafiles___), n_iterations=1, test_size=splitpercent)
        gen = sklearn.cross_validation.ShuffleSplit(len(datafiles___), 1, splitpercent)
        for i_trs, i_tes in gen:
            return [datafiles___[i] for i in i_trs],  [datafiles___[i] for i in i_tes]

    training_files, test_files =  get_split(data_files, .2)
    training_files, validation_files = get_split(training_files, .2)

    print training_files
    print test_files
    print validation_files

    train_set_y = np.hstack([[artist_string_to_id[f[:18]]] * 129 for f in training_files])
    train_set_x = np.vstack([np.load(os.path.join(inpath, f)) for f in training_files])
    test_set_y = np.hstack([[artist_string_to_id[f[:18]]] * 129 for f in test_files])
    test_set_x = np.vstack([np.load(os.path.join(inpath, f)) for f in test_files])
    validation_set_y = np.hstack([[artist_string_to_id[f[:18]]] * 129 for f in validation_files])
    validation_set_x = np.vstack([np.load(os.path.join(inpath, f)) for f in validation_files])

    datasets = [(train_set_x, train_set_y), (validation_set_x, validation_set_y), (test_set_x, test_set_y)]
    return datasets




if __name__ == '__main__':
    # datasets = dostuff(sys.argv[1])
    datasets = dobetterstuff(sys.argv[1])
    with open(sys.argv[2], 'w') as f:
        pickle.dump(datasets, f)
