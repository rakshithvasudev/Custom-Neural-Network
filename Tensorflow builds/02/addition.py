import tensorflow as tf
import numpy as np

a = tf.constant(value=np.random.rand(1000, 5).tolist(), name='first_matrix')
b = tf.constant(value=np.random.rand(5, 10000).tolist(), name='second_matrix')

c = tf.matmul(a, b)

with tf.Session() as sess:
    print(sess.run(c))
