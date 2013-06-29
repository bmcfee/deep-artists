#!/usr/bin/env python
# CREATED:2013-06-29 10:29:10 by Brian McFee <brm2132@columbia.edu>
# whee!
import cPickle
import gzip
import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T
import numpy
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from itertools import chain

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



# logistic regression copied from theano examples
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class Deepotron(object):

    def __init__(self,
                 input,
                 n_classes_out,
                 rng=None,
                 log_layer_size=64):
        if rng is None:
            rng = np.random.RandomState(23455)
        x = T.matrix('x')
        # layer0_input = x.reshape((input.shape[0], 1, input.shape[1], input.shape[2]))
        layer0_input = x.dimshuffle(0, 'x', 1, 2)

        # n_in needs to be the height of a spectrogram patch
        self.layers = [MellyLayer(rng, input=layer0_input, n_in=input.shape[2], n_out=log_layer_size)]
        self.logRegressionLayer = LogisticRegression(
            input=self.layers[-1].output,
            n_in=log_layer_size,
            n_out=n_classes_out)
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.params = list(chain.from_iterable([l.params for l in self.layers])) + self.logRegressionLayer.params



def dostuff():
    data_files = [f for f in os.listdir('data') if f.endswith('.npy')][:5]  # TODO FIXME
    # print data_files
    train_set_x = np.vstack([np.load(os.path.join('data', f)) for f in data_files])
    train_set_y = np.hstack([np.ones(64) *i for i in range(len(data_files))])  # FIXME I REALLY SUCK

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    rng = numpy.random.RandomState(1234)
    classifier = Deepotron(input=x,
                           n_classes_out=np.max(train_set_y)+1,
                           rng=rng)
    cost = classifier.negative_log_likelihood(y) + classifier.layers[0].L2_sqr
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    train_model = theano.function(inputs=[index], outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]})


    for epoch in range(10):
        for minibatch_index in xrange(20):
            minibatch_avg_cost = train_model(minibatch_index)
            print 'minicost:', minibatch_avg_cost








if __name__ == '__main__':
    dostuff()










#         image_shape = (batch_size, 1, log_layer_size, input.shape[-1])
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
