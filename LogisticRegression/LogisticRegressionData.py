import numpy as np


class LogisticRegressionData:
    data_x = 0
    data_y = 0

    @property
    def x_max(self):
        return np.amax(self.data_x, axis=0)

    @property
    def x_min(self):
        return np.amin(self.data_x, axis=0)

    def __init__(self, data_x, data_y, feature_scale = True):
        self.data_x = np.array(data_x, dtype=np.float32)
        self.data_y = np.array(data_y, dtype=np.float32)

        if feature_scale:
            self.data_x = (self.data_x-self.x_min)/(self.x_max-self.x_min)
