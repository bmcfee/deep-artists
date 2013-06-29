#!/usr/bin/env python
# CREATED:2013-06-29 10:29:10 by Brian McFee <brm2132@columbia.edu>
# whee! 

import numpy as np
import theano
import theano.tensor as T

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class MellyLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        self.input = input

        W_values = np.asarray(rng.uniform(  low=0, 
                                            high=1,
                                            size=(n_in, n_out),
                                            dtype=theano.config.floatX))

        W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = np.ones( (n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_out = T.dot(input, self.W) + self.b
        self.output = T.log(lin_out)
        self.params = [self.W, self.b]

class ReLConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
        
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in  = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))

        # Initialize filters
        W_bound = np.sqrt(6./(fan_in + fan_out))

        self.W = theano.shared(np.asarray(rng.uniform(  low=-W_bound,
                                                        high=W_bound,
                                                        size=filter_shape),
                                          dtype=theano.config.floatX),
                                borrow=True)

        # Initialize bias terms
        b_values = np.zeros((filter_shape[0], ), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(input=input, filters=self.W,
                                filter_shape=filter_shape,
                                image_shape=image_shape)

        pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize, ignore_border=True)

        self.output = T.maximum(0.0, pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]


