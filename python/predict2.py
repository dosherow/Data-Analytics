# homework 7 Drew Osherow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# training and test data sets
portfolio_train = 'portfolio_training.csv'
portfolio_test = 'portfolio_test.csv'

# load data sets
# 1st row of both csv files contains the header information:
# number of data points (n), number of features or X variables (p),
# name of 2 classes
# last column is target (Y variable taking integer values) for classification
# 1st 5 columns are features (X variables taking floating-point values)

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = portfolio_train,
    target_dtype = np.int,
    features_dtype = np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = portfolio_test,
    target_dtype = np.int,
    features_dtype = np.float32)

# specify that all features (X variables) have real-valued data

feature_columns = [tf.contrib.layers.real_valued_column('', dimension=5)]

# build a 4-layer DNN with 10, 20, 10 ReLU neurons
# in the 3 hidden layers, respectively
# the output layer contains the softmax neuron (a natural extension of sigmoid
# to the case of multiple classes) which outputs probabilities (for the 3 classes)

classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                            hidden_units = [10, 20, 10],
                                            n_classes = 2,
                                            dropout = 0.4,
                                            model_dir = '/tmp/portfolio_model')

# fit deep learning model (learning network weights using the SGD algorithm with mini-batch

classifier.fit(x = training_set.data,
               y = training_set.target,
               steps = 2000)

# prediction with deep learning
# evaluate classification accuracy
accuracy_score = classifier.evaluate(x = test_set.data,
                                     y = test_set.target)['accuracy']
# print out classification accuracy (1 - misclassification error rate)
print('Accuracy: {0:f}'.format(accuracy_score))