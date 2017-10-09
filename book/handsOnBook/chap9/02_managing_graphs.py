# coding:utf-8

import tensorflow as tf
from pre_setting import reset_graph


def demo_one():
    print "#############   demo_1  ##############"
    x1 = tf.Variable(1)
    print x1.graph is tf.get_default_graph()


def demo_two():
    print "#############   demo_2  #########"
    graph = tf.Graph()
    with graph.as_default():
        x2 = tf.Variable(2)
    print x2.graph is graph
    print x2.graph is tf.get_default_graph()


def demo_three():
    w = tf.constant(3)
    x = w + 2
    y = x + 5
    z = x * 3

    with tf.Session() as sess:
        print(y.eval())  # 10
        # It is important to note that it will not reuse the result of the previous evaluation of w and x . In short, the preceding code evaluates w and x twice.
        print(z.eval())  # 15

        '''
	If you want to evaluate y and z efficiently, without evaluating w and x twice as in the
	previous code, you must ask TensorFlow to evaluate both y and z in just one graph
	run, as shown in the following code:
	'''
    with tf.Session() as sess:
        y_val, z_val = sess.run([y, z])
        print(y_val)  # 10
        print(z_val)  # 15


if __name__ == '__main__':
    reset_graph()
    demo_one()
    demo_two()
    demo_three()
