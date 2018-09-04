import tensorflow as tf
import numpy as np

class BasicMatrixOperations:
    def _convert(self, v, t=tf.float32):
        return tf.convert_to_tensor(v, t)

    def run(self):
        m1 = self._convert(np.array(np.random.rand(4, 4), dtype='float32'))
        m2 = self._convert(np.array(np.random.rand(4, 4), dtype='float32'))
        m3 = self._convert(np.array(np.random.rand(4, 4), dtype='float32'))
        m4 = self._convert(np.array(np.random.rand(4, 4), dtype='float32'))
        m5 = self._convert(np.array(np.random.rand(4, 4), dtype='float32'))
        m6 = self._convert(np.array(np.random.rand(4, 4), dtype='float32'))

        m_transpose = tf.transpose(m1)
        m_mul = tf.matmul(m1, m2)
        m_det = tf.matrix_determinant(m3)
        m_inv = tf.matrix_inverse(m4)
        m_solve = tf.matrix_solve(m5, [[1], [1], [1], [1]])
        m_random_shuffle = tf.random_shuffle(m6)
        m_top_k = tf.nn.top_k(m1, k=3)

        with tf.Session() as session:
            print("Basic matrix operations result:")

            print("Transpose input: ")
            print(session.run(m1))
            print("Transpose: ", session.run(m_transpose))

            print("")
            print("Multiply input: ")
            print(session.run(m1), session.run(m2))
            print("Multiply: ", session.run(m_mul))

            print("")
            print("Inverse input: ")
            print(session.run(m4))
            print("Inverse: ")
            print(session.run(m_inv))

            print("")
            print("Determinant input: ")
            print(session.run(m3))
            print("Determinant: ")
            print(session.run(m_det))

            print("")
            print("Solve input: ")
            print(session.run(m5))
            print("Solve: ")
            print(session.run(m_solve))

            print("")
            print("Random shuffle input: ")
            print(session.run(m6))
            print("Random shuffle: ")
            print(session.run(m_random_shuffle))

            print("")
            topk_values, topk_indices = session.run(m_top_k)
            print("Top k input: ")
            print(session.run(m1))
            print("Top k with k = 3: ")
            print("Indices: ")
            print(topk_indices)
            print("Values: ")
            print(topk_values)