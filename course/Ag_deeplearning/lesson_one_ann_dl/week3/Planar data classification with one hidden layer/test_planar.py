# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


X, Y = load_planar_dataset()  # X: 2x400

plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)


# print X, Y

# print len(X[0])

# print np.array(X.T)
# print np.shape(X.T)[0]

# plt.show()


# X_assess, Y_assess = layer_sizes_test_case()
# print X_assess, Y_assess
# print np.shape(X_assess)
# print np.shape(Y_assess)


# print np.random.randn(2, 3)


# w = np.random.randn(4, 2) * 0.01

# X_assess, parameters = forward_propagation_test_case()
# W2 = parameters['W2']
# b2 = parameters['b2']

# print np.shape(X_assess)

# W1 = parameters['W1']
# print W1
# print '\n'
# print X_assess


# print '\n'
# Z1 = np.dot(W1, X_assess)

# print Z1

# print '\n'

# A1 = np.tanh(Z1)

# print 'W2'
# print W2

# print 'A1'
# print A1


# Z2 = np.dot(W2, A1) + b2

# print 'Z2'
# print Z2[0][0]

# A2 = np.tanh(Z2[0][0])
# print 'A2'
# print A2
# print np.mean(A2)


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    # START CODE HERE ### (≈ 5 lines of code)
    # parameters = None
    # W1 = None
    # b1 = None
    # W2 = None
    # b2 = None
    # ### END CODE HERE ###

    # # Loop (gradient descent)

    # for i in range(0, num_iterations):

    #     # START CODE HERE ### (≈ 4 lines of code)
    #     # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
    #     A2, cache = None

    #     # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
    #     cost = None

    #     # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
    #     grads = backward_propagation(parameters, cache, X, Y)

    #     # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
    #     parameters = update_parameters(parameters, grads)

    #     ### END CODE HERE ###

    #     # Print the cost every 1000 iterations
    #     if print_cost and i % 1000 == 0:
    #         print ("Cost after iteration %i: %f" % (i, cost))

    # return parameters


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    # START CODE HERE ### (≈ 3 lines of code)
    n_x = np.shape(X)[0]  # size of input layer
    n_h = 4  # random?
    n_y = np.shape(Y)[0]  # size of output layer
    ## END CODE HERE ###
    return (n_x, n_h, n_y)


# X_assess, Y_assess = nn_model_test_case()

# parameters = nn_model(X_assess, Y_assess, 4,
    # num_iterations=10000, print_cost=False)

# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


for i in xrange(0, 2):
    print i
