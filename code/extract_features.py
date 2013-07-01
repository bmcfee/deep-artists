#!/usr/bin/env python
# encoding: utf-8
'''
Usage: ./extract_features.py input_dir output_dir
Takes  all  mp3  files  and outputs  mfcc-like  features  in  .mel.npy
matrices.
'''
import librosa
import numpy as np
import scipy
import os
import sys
from joblib import Parallel, delayed

PATCH_SIZE = 40


def file_extract_features(ff, output_dir):
    outname = os.path.join(output_dir, os.path.basename(ff)[:-4] + '.mel.npy')
    y, sr = librosa.load(os.path.join(input_dir, f))
    y = y[-22050*30:]
    D = librosa.feature.melspectrogram(y, n_fft=512, n_mels=64, fmax=8000)
    D = D[:,:(D.shape[1]/PATCH_SIZE) * PATCH_SIZE]
    D = D.reshape((D.shape[0], 40, -1), order='F')
    D = D.swapaxes(2,1).swapaxes(1,0)
    D = D.reshape((D.shape[0], -1))
    np.save(outname, D.astype(np.float32))
    print 'done', outname


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp3')]
    Parallel(n_jobs=4, pre_dispatch=16)(delayed(file_extract_features)(q, output_dir) for q in all_files)
