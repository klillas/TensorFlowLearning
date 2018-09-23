import numpy as np


class CNNData:
    data_x = None
    labels = None
    labels_one_hot = None

    def __init__(self, data_x, labels, labels_one_hot):
        self.data_x = np.array(data_x, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int32)
        self.labels_one_hot = np.array(labels_one_hot, dtype=np.int32)