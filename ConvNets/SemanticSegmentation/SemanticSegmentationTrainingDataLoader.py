from scipy import misc
import numpy as np
import glob
import re
import os
import time
from random import *

from ConvNets.SemanticSegmentation.SemanticSegmentationData import SemanticSegmentationData


class SemanticSegmentationTrainingDataLoader:
    training_data_path = None
    real_world_test_path = None
    image_width = None
    image_height = None
    image_channels = None
    label_count = None
    training_set_ratio = None
    batch_size = None
    probability_delete_example = None

    _labels = None
    _data_x = None
    _real_world_examples = None

    def load_picture_data(self, file_path):
        picture_data = misc.imread(file_path)
        return picture_data

    def initialize(self, batch_size, probability_delete_example):
        self.training_data_path = "C:/temp/training/"
        self.real_world_test_path = "C:/temp/RealWorldTest/"
        self.image_width = 256
        self.image_height = 192
        self.image_channels = 3
        self.label_count = 2
        self.training_set_ratio = 0.98
        self.batch_size = batch_size
        self.probability_delete_example = probability_delete_example
        self._labels = np.zeros(shape=(self.batch_size, self.image_height * self.image_width), dtype=np.uint8)
        self._data_x = np.zeros(shape=(self.batch_size, self.image_height, self.image_width, self.image_channels), dtype=np.uint8)
        self._load_real_world_training()

    def delete_all_existing_training_data(self):
        datFiles = glob.glob(self.training_data_path + "*.dat")
        jpgFiles = glob.glob(self.training_data_path + "*.jpg")
        for datFile in datFiles:
            if re.search("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", datFile) == None:
                raise ValueError("Unexpected file to delete: {}".format(datFile))
            os.remove(datFile)

        for jpgFile in jpgFiles:
            if re.search("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", jpgFile) == None:
                raise ValueError("Unexpected file to delete: {}".format(jpgFile))
            os.remove(jpgFile)

    def load_next_batch(self, delete_batch_source=False):
        datFiles = glob.glob(self.training_data_path + "*.dat")
        while len(datFiles) < 500:
            time.sleep(.500)
            datFiles = glob.glob(self.training_data_path + "*.dat")

        training_ids = np.random.choice(len(datFiles), self.batch_size, replace=False)
        for i in range(self.batch_size):
            training_id = training_ids[i]
            dat_file = datFiles[training_id]
            file_id = re.search("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", dat_file).group()

            leftEyeImagePath = self.training_data_path + file_id + "_CameraLeftEye.jpg"
            self._data_x[i] = misc.imread(leftEyeImagePath)

            labelsPath = self.training_data_path + file_id + "_labels.dat"
            self._labels[i] = np.fromfile(labelsPath, dtype=np.uint8, count=self.image_height*self.image_width).reshape((1, self.image_height * self.image_width))

            if random() < self.probability_delete_example or delete_batch_source == True:
                os.remove(leftEyeImagePath)
                os.remove(labelsPath)


        semantic_segmentation_data = SemanticSegmentationData(
            self._data_x,
            self._labels,
            self.label_count
        )

        return semantic_segmentation_data

    def get_real_world_training_examples(self):
        return self._real_world_examples

    def _load_real_world_training(self):
        jpgFiles = glob.glob(self.real_world_test_path + "*.jpg")
        self._real_world_examples = np.zeros(shape=(len(jpgFiles), self.image_height, self.image_width, self.image_channels), dtype=np.uint8)
        for i in range(len(jpgFiles)):
            self._real_world_examples[i] = misc.imread(jpgFiles[i])