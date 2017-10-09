# coding: utf-8

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

import numpy as np

import tensorflow as tf
from pre_setting import reset_graph


def train_by_BGD(housing, scaled_housing_data_plus_bias):

    print scaled_housing_data_plus_bias[0]
    print(scaled_housing_data_plus_bias.mean(axis=0))
    print(scaled_housing_data_plus_bias.mean(axis=1))
    print(scaled_housing_data_plus_bias.mean())
    print(scaled_housing_data_plus_bias.shape)

    n_epochs = 1000
    learning_rate = 0.01
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

    theta = tf.Variable(tf.random_uniform(
        [n + 1, 1], -1.0, 1.0, seed=42), name="theta")

    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

    gradients = 2 / m * tf.matmul(tf.transpose(X), error)
    training_op = tf.assign(theta, theta - learning_rate * gradients)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)

        best_theta = theta.eval()

    print best_theta


def train_autodiff_BGD(housing, scaled_housing_data_plus_bias):
    n_epochs = 1000
    learning_rate = 0.01
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform(
        [n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

    # gradients = tf.gradients(mse, [theta])[0]
    # training_op = tf.assign(theta, theta - learning_rate * gradients)


# Using a GradientDescentOptimizer replace the preceding      ##########3
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    ####################     Using a momentum optimizer     #####################
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    # training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)

        best_theta = theta.eval()

    print("Best theta:")
    print(best_theta)


if __name__ == '__main__':
    reset_graph()
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    # train_by_BGD(housing, scaled_housing_data_plus_bias)

    train_autodiff_BGD(housing, scaled_housing_data_plus_bias)
