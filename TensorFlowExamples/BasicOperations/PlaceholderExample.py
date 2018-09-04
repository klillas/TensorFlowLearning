import tensorflow as tf


class PlaceholderExample:
    def run(self):
        x = tf.placeholder(tf.float32, name="x")
        y = tf.placeholder(tf.float32, name="y")

        z = tf.multiply(x, y, name="z")

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            print("Placeholder example result: ", session.run(z, feed_dict={x: 2.1, y: 3.0}))
