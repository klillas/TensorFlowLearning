import tensorflow as tf
import numpy as np


class NumpyToTensorConversion:
    def run(self):
        numpyArray = np.array([1, 2, 3, 4, 5])
        numpyMatrix = np.array(np.random.randn(4, 4), dtype='float32')

        tensor = tf.convert_to_tensor(numpyArray, dtype=tf.float64)
        tensorMatrix = tf.convert_to_tensor(numpyMatrix)
        tensorOperationWithNumpyArrays = tf.multiply(numpyArray, numpyArray)

        with tf.Session() as session:
            print("NumpyToTensorConversion output: ")
            print("Tensor array: ", session.run(tensor))
            print("Tensor array pos 0: ", session.run(tensor[0]))
            print("Tensor matrix: ", session.run(tensorMatrix))
            print("Tensor operation using Numpy arrays as input: ", session.run(tensorOperationWithNumpyArrays))