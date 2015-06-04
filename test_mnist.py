import sys
import os
import time
import numpy
import numpy as np
import pickle

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

import theano
import theano.tensor as T

from logistic_regression import LogisticRegression
from fully_connected import HiddenLayer
from conv_layer import LeNetConvPoolLayer


mnist = fetch_mldata('MNIST original', data_home='data')

X = mnist.data.astype(theano.config.floatX) / 255
Y = mnist.target.astype(np.int32)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=10000)
X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=5000)

batch_size = 500
n_train_batches = X_train.shape[0] / batch_size
n_valid_batches = X_validate.shape[0] / batch_size
n_test_batches = X_test.shape[0] / batch_size
train_set_x = theano.shared(X_train, borrow=True)
train_set_y = theano.shared(Y_train, borrow=True)
valid_set_x = theano.shared(X_validate, borrow=True)
valid_set_y = theano.shared(Y_validate, borrow=True)
test_set_x = theano.shared(X_test, borrow=True)
test_set_y = theano.shared(Y_test, borrow=True)

nkerns=[20, 50]
n_epochs=200
learning_rate=0.1
rng = numpy.random.RandomState(1234)
L2_reg = 0.0001

x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                    # [int] labels
index = T.lscalar()  # index to a [mini]batch

######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

# Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
# (28, 28) is the size of MNIST images.
layer0_input = x.reshape((batch_size, 1, 28, 28))

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
# maxpooling reduces this further to (24/2, 24/2) = (12, 12)
# 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(batch_size, 1, 28, 28),
    filter_shape=(nkerns[0], 1, 5, 5),
    poolsize=(2, 2)
)

# Construct the second convolutional pooling layer
# filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
# maxpooling reduces this further to (8/2, 8/2) = (4, 4)
# 4D output tensor is thus of shape (nkerns[0], nkerns[1], 4, 4)
layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[0], 12, 12),
    filter_shape=(nkerns[1], nkerns[0], 5, 5),
    poolsize=(2, 2)
)

# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
# or (500, 50 * 4 * 4) = (500, 800) with the default values.
layer2_input = layer1.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=nkerns[1] * 4 * 4,
    n_out=500,
    activation=T.tanh
)

# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

L2_sqr = (layer0.W ** 2).sum() + (layer1.W ** 2).sum() + (layer2.W ** 2).sum() + (layer3.W ** 2).sum()

# the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood(y) + L2_reg * L2_sqr

# create a function to compute the mistakes that are made by the model
test_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

# create a list of all model parameters to be fit by gradient descent
params = layer3.params + layer2.params + layer1.params + layer0.params

#print 'total params:', sum([tmp.get_value().size for tmp in params])
#sys.exit(1)

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i], grads[i]) pairs.
updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

train_model = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)


###############
# TRAIN MODEL #
###############
print '... training'
# early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                       # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant
validation_frequency = min(n_train_batches, patience / 2)
                              # go through this many
                              # minibatche before checking the network
                              # on the validation set; in this case we
                              # check every epoch

best_validation_loss = numpy.inf
best_iter = 0
test_score = 0.
start_time = time.clock()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

        iter = (epoch - 1) * n_train_batches + minibatch_index

        if iter % 100 == 0:
            print 'training @ iter = ', iter
        cost_ij = train_model(minibatch_index)

        if (iter + 1) % validation_frequency == 0:

            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i
                                 in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, time %.2fm, minibatch %i/%i, validation error %f %%' %
                  (epoch, (time.clock()-start_time)/60.0, minibatch_index + 1, n_train_batches,
                   this_validation_loss * 100.))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = [
                    test_model(i)
                    for i in xrange(n_test_batches)
                ]
                test_score = numpy.mean(test_losses)
                print(('     epoch %i, time %.2fm, minibatch %i/%i, test error of '
                       'best model %f %%') %
                      (epoch, (time.clock()-start_time)/60.0, minibatch_index + 1, n_train_batches,
                       test_score * 100.))

        #if patience <= iter:
        #    done_looping = True
        #    break

end_time = time.clock()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iteration %i, '
      'with test performance %f %%' %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
print ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
pickle.dump(((layer0.W.get_value(), layer0.b.get_value()), (layer1.W.get_value(), layer1.b.get_value()), (layer2.W.get_value(), layer2.b.get_value()), (layer3.W.get_value(), layer3.b.get_value())), open('model.pkl', 'w'), -1)
