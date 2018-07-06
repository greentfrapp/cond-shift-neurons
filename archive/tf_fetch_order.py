import tensorflow as tf

sess = tf.Session()

a = tf.Variable(
	initial_value=2,
	dtype=tf.float32,
)

b = tf.Variable(
	initial_value=5,
	dtype=tf.float32,
)

assign_op = a.assign(b)

sess.run(tf.global_variables_initializer())

fetch_arg = [a, assign_op]
# fetch_arg = [assign_op, a]

print(sess.run(fetch_arg)[0])