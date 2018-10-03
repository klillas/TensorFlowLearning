import numpy as np


class SemanticSegmentationData:
    data_x = None
    labels = None

    def __init__(self, data_x, labels):
        self.data_x = np.array(data_x, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int32)