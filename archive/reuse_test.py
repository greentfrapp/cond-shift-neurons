import tensorflow as tf
import numpy as np


class Model():

	def __init__(self, inputs):
		self.output = tf.layers.dense(
			inputs=inputs,
			units=1,
			activation=None,
			name="model_dense",
			reuse=tf.AUTO_REUSE,
		)

sess = tf.Session()

a = tf.placeholder(
	shape=(None, 2),
	dtype=tf.float32,
	name="a",
)

with tf.variable_scope('1'):
	model_1 = Model(a)
with tf.variable_scope('1'):
	model_2 = Model(a)

sess.run(tf.global_variables_initializer())

print(sess.run([model_1.output, model_2.output], {a: np.ones((1, 2))}))

