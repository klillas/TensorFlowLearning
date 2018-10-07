import numpy as np
from scipy import misc

class SemanticSegmentationData:
    data_x = None
    labels = None
    labels_one_hot = None
    label_count = None

    def __init__(self, data_x, labels, label_count):
        self.label_count = label_count
        self.data_x = np.array(data_x, dtype=np.uint8)
        self.labels = np.array(labels, dtype=np.int32)
        self.labels_one_hot = np.eye(self.label_count)[self.labels]

    def exportImage(self, path, id):
        misc.imsave(path, self.data_x[id])