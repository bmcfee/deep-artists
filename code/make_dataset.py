#!/usr/bin/env python

import cPickle as pickle
import numpy as np
import os
import sys
import gzip

def dostuff(inpath):
    
    
    data_files = [f for f in os.listdir(inpath) if f.endswith('.mel.npy')]  # TODO FIXME
    # print data_files
    train_set_x = np.vstack([np.load(os.path.join(inpath, f)) for f in data_files])
    train_set_y = np.hstack([np.ones(129) *i for i in range(len(data_files))])  # FIXME I REALLY SUCK

    datasets = [(train_set_x, train_set_y), (train_set_x, train_set_y), (train_set_x, train_set_y)]
    
    return datasets


if __name__ == '__main__':
    datasets = dostuff(sys.argv[1])
    
    with gzip.open(sys.argv[2], 'w') as f:
        pickle.dump(datasets, f)
