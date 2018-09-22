import numpy as np


class CNNData:
    data_x = 0
    labels = 0

    def __init__(self, data_x, labels):
        self.data_x = np.array(data_x, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int32)