import tensorflow as tf
import numpy as np

class SegmentationExample:
    def _convert(self, v, t=tf.float32):
        return tf.convert_to_tensor(v, t)

    def run(self):
        seg_ids = tf.constant([0, 0, 1, 2, 2])

        tens1 = self._convert(np.array([
            (2, 5, 3, -5),
            (0, 3, -2, 5),
            (4, 3, 5, 3),
            (6, 1, 4, 0),
            (6, 1, 4, 0)]),
            tf.int32)

        tens2 = self._convert(np.array([1, 2, 3, 4, 5]), tf.int32)

        seg_sum = tf.segment_sum(tens1, seg_ids)
        seg_sum_1 = tf.segment_sum(tens2, seg_ids)

        with tf.Session() as session:
            print("Segmentation ids: ", session.run(seg_ids))
            print("Matrix to segment: ")
            print(session.run(tens1))
            print("Result: ")
            print(session.run(seg_sum))

            print("Array to segment: ", session.run(tens2))
            print("Result: ", session.run(seg_sum_1))