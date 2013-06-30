#!/usr/bin/env python
# encoding: utf-8

import librosa
import numpy as np
import scipy
import os
import sys

PATCH_SIZE = 40

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    c = 0
    for f in os.listdir(input_dir):
        if f.endswith('.mp3'):
            y, sr = librosa.load(os.path.join(input_dir, f))
            y = y[-22050*30:]
            D = librosa.feature.melspectrogram(y, n_fft=512, n_mels=64, fmax=8000)
            D = D[:,:(D.shape[1]/PATCH_SIZE) * PATCH_SIZE]
            D = D.reshape((D.shape[0], 40, -1), order='F')
            D = D.swapaxes(2,1).swapaxes(1,0)
            D = D.reshape((D.shape[0], -1))
            outname = os.path.join(output_dir, f[:-4] + '.mel.npy')
            np.save(outname, D.astype(np.float32))
            c += 1
            print c, outname
