# coding:utf-8

import tensorflow as tf

from pre_setting import reset_graph

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print node1, node2


def demo_one(x, y, f):
    sess = tf.Session()
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    print result
    sess.close()


def demo_two(x, y, f):
    '''B
the session is automatically closed at the end of the block.
    '''
    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()
        print result

    # equivalent
    init = tf.global_variables_initializer()  # prepare an init node
    with tf.Session() as sess:
        init.run()  # actually initialize all the variables
        result = f.eval()


def demo_three(x, y, f):
    '''
The only difference from a regular Session is that when an InteractiveSes
sion is created it automatically sets itself as the default session, so you don’t need a
with block
    '''
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    init.run()
    result = f.eval()
    print result


if __name__ == '__main__':
    '''
Creating Your First Graph and Running It in a Session
    '''
    reset_graph()
    x = tf.Variable(3, name='x')
    y = tf.Variable(4, name='y')
    f = x * x * y + y + 2
    # demo_one(x, y, f)
    # demo_two(x, y, f)
    demo_three(x, y, f)
