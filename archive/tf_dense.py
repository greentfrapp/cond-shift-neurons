import tensorflow as tf
import numpy as np


a = tf.placeholder(
	shape=(None, 5, 4, 3, 2),
	dtype=tf.float32,
)

b = tf.layers.dense(
	inputs=a,
	units=10,
)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(b, {a: np.random.randn(7,5,4,3,2)}).shape)