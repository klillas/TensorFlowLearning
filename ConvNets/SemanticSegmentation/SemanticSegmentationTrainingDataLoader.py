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
    image_width = None
    image_height = None
    image_channels = None
    label_count = None
    training_set_ratio = None
    batch_size = None
    probability_delete_example = None

    _labels = None
    _data_x = None

    def load_picture_data(self, file_path):
        picture_data = misc.imread(file_path)
        return picture_data

    def initialize(self, batch_size, probability_delete_example):
        self.training_data_path = "C:/temp/training/"
        self.image_width = 256
        self.image_height = 192
        self.image_channels = 3
        self.label_count = 2
        self.training_set_ratio = 0.98
        self.batch_size = batch_size
        self.probability_delete_example = probability_delete_example
        self._labels = np.zeros(shape=(self.batch_size, self.image_height * self.image_width), dtype=np.uint8)
        self._data_x = np.zeros(shape=(self.batch_size, self.image_height, self.image_width, self.image_channels), dtype=np.uint8)


    def load_next_batch(self):
        datFiles = glob.glob("c:/temp/training/*.dat")
        while len(datFiles) < 500:
            time.sleep(.500)
            datFiles = glob.glob("c:/temp/training/*.dat")

        training_ids = np.random.choice(len(datFiles), self.batch_size, replace=False)
        for i in range(self.batch_size):
            training_id = training_ids[i]
            dat_file = datFiles[training_id]
            file_id = str(int(re.search(r'\d+', dat_file).group()))

            leftEyeImagePath = self.training_data_path + file_id + "_CameraLeftEye.jpg"
            rightEyeImagePath = self.training_data_path + file_id + "_CameraRightEye.jpg"
            self._data_x[i] = misc.imread(leftEyeImagePath)

            labelsPath = self.training_data_path + file_id + "_labels.dat"
            self._labels[i] = np.fromfile(labelsPath, dtype=np.uint8, count=self.image_height*self.image_width).reshape((1, self.image_height * self.image_width))

            if random() < self.probability_delete_example:
                os.remove(leftEyeImagePath)
                os.remove(rightEyeImagePath)
                os.remove(labelsPath)


        semantic_segmentation_data = SemanticSegmentationData(
            self._data_x,
            self._labels,
            self.label_count
        )

        return semantic_segmentation_data

        #shuffle_array = np.arange(0, data_set_size)
        #np.random.shuffle(shuffle_array)

        #data_x = data_x[shuffle_array]
        #labels = labels[shuffle_array]


    def generate_traindata_from_depthvision_pictures(self):
        trainingDataPath = "C:/temp/training/"
        data_set_size = 15000
        image_width = 256
        image_height = 192
        image_channels = 3
        label_count = 2
        training_set_ratio = 0.98
        labels = np.zeros(shape=(data_set_size, image_height * image_width), dtype=np.uint8)
        data_x = np.zeros(shape=(data_set_size, image_height, image_width, image_channels),dtype=np.uint8)
        for i in range(data_set_size):
            if i % 1000 == 0:
                print("{} examples loaded".format(i))
            leftEyeImage = misc.imread(trainingDataPath + str(i) + "_CameraLeftEye.jpg")
            #rightEyeImage = misc.imread(trainingDataPath + str(i) + "_CameraRightEye.jpg")
            #bothEyes = np.concatenate((leftEyeImage, rightEyeImage))
            data_x[i] = leftEyeImage

            # labels[i] = np.loadtxt(fname=trainingDataPath + str(i) + "_labels.dat", dtype=np.uint8).reshape((1, image_height, image_width))
            labels[i] = np.fromfile(trainingDataPath + str(i) + "_labels.dat", dtype=np.uint8, count=image_height*image_width).reshape((1, image_height * image_width))

        shuffle_array = np.arange(0, data_set_size)
        np.random.shuffle(shuffle_array)

        data_x = data_x[shuffle_array]
        labels = labels[shuffle_array]

        training_data_size = int(data_set_size * training_set_ratio)

        semantic_segmentation_data_train = SemanticSegmentationData(
            data_x[0:training_data_size],
            labels[0:training_data_size],
            label_count
        )
        semantic_segmentation_data_validation = SemanticSegmentationData(
            data_x[training_data_size:data_set_size],
            labels[training_data_size:data_set_size],
            label_count
        )
        return semantic_segmentation_data_train, semantic_segmentation_data_validation, image_height, image_width, image_channels