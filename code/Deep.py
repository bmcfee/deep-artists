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

#         b_values = np.ones( (n_out,), dtype=theano.config.floatX)
#         b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
#         self.b = b

        linear_out  = T.tensordot(input, self.W, axes=(2, 1))
        self.output = T.log(linear_out + 1.0)
#         self.params = [self.W, self.b]
        self.params = [self.W]
        self.L2_sqr = (self.W **2).sum()



class ReLUConvPoolLayer(object):
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
        self.L2_sqr = (self.W **2).sum()

        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        self.output_size = (image_shape[0], filter_shape[0], 
                                (image_shape[2] - filter_shape[2] + 1) / poolsize[0],
                                (image_shape[3] - filter_shape[3] + 1) / poolsize[1])


class Deepotron(object):

    def __init__(self,  input_shape, 
                        rng=None,
                        log_layer=64, 
                        batch_size=40,
                        n_filters=[3], 
                        filter_sizes=[(9,9)]):

        assert len(n_filters) == len(filter_sizes)

        n_layers = len(n_filters)

        if rng is None:
            rng = np.random.RandomState(23455)


        x = T.matrix('x')
        y = T.ivector('y')

        layer0_input = x.reshape((input_shape[0], 1, input_shape[1], input_shape[2]))

        # n_in needs to be the height of a spectrogram patch
        layers = [MellyLayer(rng, input=layer0_input, n_in=input_shape[2], n_out=log_layer)]

        # def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):

#         image_shape = (batch_size, 1, log_layer, input_shape[-1])
#         channels_in = image_shape[1]
#         for i in range(n_layers):
#             layers.append(
#                 ReLUConvPoolLayer(  
#                         rng=rng, 
#                         input=layers[-1].output,
#                         image_shape=image_shape,
#                         filter_shape=(n_filters[i], channels_in, filter_sizes[i][0], filter_sizes[i][1])),
#                         poolsize=(2,2)
#             )

#             image_shape = layers[-1].output_size
#             channels_in = image_shape[1]



