import tensorflow as tf


class ConstantExample:
    def run(self):
        x = tf.constant(1.0, name="x", dtype=tf.float32)
        a = tf.constant(2.0, name="a", dtype=tf.float32)
        b = tf.constant(3.0, name="b", dtype=tf.float32)

        y = tf.Variable(tf.add(tf.multiply(a, b), x))

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            print("Constant example result: ", session.run(y))
