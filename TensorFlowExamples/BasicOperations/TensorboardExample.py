import tensorflow as tf
from subprocess import call
import os


class TensorboardExample:
    def run(self):
        x = tf.placeholder(dtype = tf.float32, name = "x")
        y = tf.placeholder(dtype = tf.float32, name = "y")

        z = tf.multiply(x, y)

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("logs", session.graph)

            session.run(init)
            print("Tensorboard example result: ", session.run(z, feed_dict={x: 2.0, y: 3.0}))