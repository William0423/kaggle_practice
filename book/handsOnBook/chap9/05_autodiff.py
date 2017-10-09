
import numpy as np

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def np_func(a, b):
    z = 0
    for i in range(100):
        z = a * np.cos(z + i) + z * np.sin(b - i)
    print z


def tf_func():
    a = tf.Variable(0.2, name="a")
    b = tf.Variable(0.3, name="b")
    z = tf.constant(0.0, name="z0")
    for i in range(100):
        z = a * tf.cos(z + i) + z * tf.sin(b - i)

    grads = tf.gradients(z, [a, b])
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        print(z.eval())
        print(sess.run(grads))


if __name__ == '__main__':
    np_func(0.2, 0.3)
    tf_func()
