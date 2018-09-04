import tensorflow as tf
import numpy as np


class MatrixReduction:
    def _convert(self, v, t = tf.float32):
        return tf.convert_to_tensor(v, dtype = t)

    def run(self):
        x = self._convert(np.array([
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9)
        ]), tf.int32)

        bool_tensor = self._convert([
            (True, False, True),
            (False, False, False),
            (True, True, True)
        ], tf.bool)

        red_sum_0 = tf.reduce_sum(x)
        red_sum = tf.reduce_sum(x, axis = 1)

        red_prod_0 = tf.reduce_prod(x)
        red_prod = tf.reduce_prod(x, axis = 1)

        red_min_0 = tf.reduce_min(x)
        red_min = tf.reduce_min(x, axis = 1)

        red_max_0 = tf.reduce_max(x)
        red_max = tf.reduce_max(x, axis = 1)

        red_mean_0 = tf.reduce_mean(x)
        red_mean = tf.reduce_mean(x, axis = 1)

        red_bool_all_0 = tf.reduce_all(bool_tensor)
        red_bool_all = tf.reduce_all(bool_tensor, axis = 1)

        red_bool_any_0 = tf.reduce_any(bool_tensor)
        red_bool_any = tf.reduce_any(bool_tensor, axis = 1)

        with tf.Session() as session:
            print("Original matrix: ", session.run(x))

            print("Reduce sum without passed axis parameter: ", session.run(red_sum_0))
            print("Reduce sum with passed axis = 1: ", session.run(red_sum))

            print("Reduce product without passed axis parameter: ", session.run(red_prod_0))
            print("Reduce product with passed axis parameter = 1: ", session.run(red_prod))

            print("Reduce min without passed axis parameter: ", session.run(red_min_0))
            print("Reduce min with passed axis parameter = 1: ", session.run(red_min))

            print("Reduce max without passed axis parameter: ", session.run(red_max_0))
            print("Reduce max with passed axis parameter = 1: ", session.run(red_max))

            print("Reduce mean without passed axis parameter: ", session.run(red_mean_0))
            print("Reduce mean with passed axis parameter = 1: ", session.run(red_mean))

            print("Original boolean matrix: ", session.run(bool_tensor))

            print("Reduce bool all without passed axis parameter: ", session.run(red_bool_all_0))
            print("Reduce bool all with passed axis parameter = 1: ", session.run(red_bool_all))

            print("Reduce bool any without passed axis parameter: ", session.run(red_bool_any_0))
            print("Reduce bool any with passed axis parameter = 1: ", session.run(red_bool_any))