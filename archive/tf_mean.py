import tensorflow as tf
import numpy as np

a = tf.placeholder(
	shape=(None, 28, 28, 3),
	dtype=tf.float32,
)

b = tf.reduce_mean(a, [0, 1, 3])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(b, {a: np.random.randn(5, 28, 28, 3)}).shape)
