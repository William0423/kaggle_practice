# coding: utf-8

import tensorflow as tf
from pre_setting import reset_graph

from sklearn.datasets import fetch_california_housing

import numpy as np

from sklearn.preprocessing import StandardScaler


def placeholder_nodes():
    A = tf.placeholder(tf.float32, shape=(None, 3))
    B = A + 5
    with tf.Session() as sess:
        B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
        B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

    print(B_val_1)

    print(B_val_2)


def train_by_MbGD():
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
    n_epochs = 1000
    learning_rate = 0.01
    reset_graph()

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

    theta = tf.Variable(tf.random_uniform(
        [n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))

    def fetch_batch(epoch, batch_index, batch_size):
        # not shown in the book
        np.random.seed(epoch * n_batches + batch_index)
        indices = np.random.randint(m, size=batch_size)  # not shown
        X_batch = scaled_housing_data_plus_bias[indices]  # not shown
        y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
        return X_batch, y_batch

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()
        print best_theta


if __name__ == '__main__':
    # placeholder_nodes()

    train_by_MbGD()
