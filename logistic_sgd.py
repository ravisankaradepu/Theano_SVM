from __future__ import print_function
__docformat__ = 'restructedtext en'
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
        train_set=list(train_set)
    zz=numpy.where(numpy.logical_and(train_set[1]>=0, train_set[1]<=1))
    train_set[0] = train_set[0][zz]
    train_set[1] = train_set[1][zz]
    zz=numpy.where(train_set[1] == 0)
    train_set[1][zz] = -1
    train_set=tuple(train_set)

    valid_set=list(valid_set)
    zz=numpy.where(numpy.logical_and(valid_set[1]>=0, valid_set[1]<=1))
    valid_set[0] = valid_set[0][zz]
    valid_set[1] = valid_set[1][zz]
    zz=numpy.where(valid_set[1] == 0)
    valid_set[1][zz] = -1
    valid_set=tuple(valid_set)

    test_set=list(test_set)
    zz=numpy.where(numpy.logical_and(test_set[1]>=0, test_set[1]<=1))
    test_set[0] = test_set[0][zz]
    test_set[1] = test_set[1][zz]
    zz=numpy.where(test_set[1] == 0)
    test_set[1][zz] = -1
    test_set=tuple(test_set)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = numpy.asarray(data_x)
        shared_y = numpy.asarray(data_y)
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
