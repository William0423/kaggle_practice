# coding:utf-8

import tensorflow as tf
from pre_setting import reset_graph
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.preprocessing import StandardScaler

reset_graph()


housing = fetch_california_housing()
m, n = housing.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias,
                dtype=tf.float32, name="X")            # not shown
y = tf.constant(housing.target.reshape(-1, 1),
                dtype=tf.float32, name="y")            # not shown
theta = tf.Variable(tf.random_uniform(
    [n + 1, 1], -1.0, 1.0, seed=42), name="theta")
# not shown
y_pred = tf.matmul(X, theta, name="predictions")
# not shown
error = y_pred - y
# not shown
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate)            # not shown
# not shown
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval()
                  )                                # not shown
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)

    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")


with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval()  # not shown in the book

np.allclose(best_theta, best_theta_restored)

saver = tf.train.Saver({"weights": theta})
