import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage import data, color, io, img_as_float

class SemanticSegmentationData:
    data_x = None
    labels = None
    labels_one_hot = None
    label_count = None
    example_count = None

    def __init__(self, data_x, labels, label_count):
        self.label_count = label_count
        self.example_count = data_x.shape[0]
        self.data_x = np.array(data_x, dtype=np.uint8)
        self.labels = np.array(labels, dtype=np.int32)
        labels_index_255_changed = self.labels.copy()
        np.place(labels_index_255_changed, self.labels == 255, self.label_count)
        eye = np.eye(self.label_count)
        eye = np.append(np.eye(self.label_count), np.zeros((1, label_count)), axis=0)
        self.labels_one_hot = eye[labels_index_255_changed]
