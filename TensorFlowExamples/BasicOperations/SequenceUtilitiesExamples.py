import tensorflow as tf
import numpy as np

class SequenceUtilitiesExamples:
    def convert(self, v, t=tf.float32):
        return tf.convert_to_tensor(v, t)

    def run(self):
        x = self.convert(np.array(
            [
                [2, 2, 1, 3],
                [4, 5, 6, 1],
                [0, 1, 1, -2],
                [6, 2, 3, 0]
            ]
        ))

        y = self.convert(np.array([1, 2, 3, 5, 7]))

        z = self.convert((np.array([1, 0, 4, 6, 2])))

        arg_min = tf.argmin(x, 1)
        arg_max = tf.argmax(x, 1)
        unique = tf.unique(y)
        diff = tf.setdiff1d(y, z)

        with tf.Session() as session:
            print("Sequence utilities examples:")
            print("argmin and argmax input matrix: ")
            print(session.run(x))
            print("Argmin axis 1: ", session.run(arg_min))
            print("Argmax axis 1: ", session.run(arg_max))
            print("")
            print("Unique input array: ", session.run(y))
            print("Unique: ", session.run(unique))
            print("")
            print("Diff input arays: ")
            print(session.run(y))
            print(session.run(z))
            print("Diff: ", session.run(diff))