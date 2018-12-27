import threading
import queue

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
    minimum_available_training_set_size = None

    cached_datfiles = None
    cached_datfiles_original_size = None
    batch_queue = None

    _real_world_examples = None

    def load_picture_data(self, file_path):
        picture_data = misc.imread(file_path)
        return picture_data

    def initialize(self, batch_size, probability_delete_example, minimum_available_training_set_size):
        self.training_data_path = "C:\\temp\\training\\"
        self.real_world_test_path = "C:\\temp\\RealWorldTest\\"
        self.image_width = 256
        self.image_height = 192
        self.image_channels = 3
        self.label_count = 2
        self.training_set_ratio = 0.98
        self.batch_size = batch_size
        self.probability_delete_example = probability_delete_example
        self._load_real_world_training()
        self.minimum_available_training_set_size = minimum_available_training_set_size
        self.cached_datfiles = glob.glob(self.training_data_path + "*.dat")
        self.cached_datfiles_original_size = len(self.cached_datfiles)
        self.batch_queue = queue.Queue()

        asynch_load_next_batch_thread = threading.Thread(name='daemon', target=self._asynch_load_next_batch)
        asynch_load_next_batch_thread.setDaemon(True)
        asynch_load_next_batch_thread.start()

        asynch_load_next_batch_thread = threading.Thread(name='daemon', target=self._asynch_load_next_batch)
        asynch_load_next_batch_thread.setDaemon(True)
        asynch_load_next_batch_thread.start()

        asynch_load_next_batch_thread = threading.Thread(name='daemon', target=self._asynch_load_next_batch)
        asynch_load_next_batch_thread.setDaemon(True)
        asynch_load_next_batch_thread.start()

        asynch_load_next_batch_thread = threading.Thread(name='daemon', target=self._asynch_load_next_batch)
        asynch_load_next_batch_thread.setDaemon(True)
        asynch_load_next_batch_thread.start()

    def _asynch_load_next_batch(self):
        while True:
            while self.batch_queue.qsize() > 100:
                time.sleep(.100)

            while len(self.cached_datfiles) < self.minimum_available_training_set_size or len(self.cached_datfiles) / self.cached_datfiles_original_size < 0.8:
                time.sleep(.500)
                self.cached_datfiles = glob.glob(self.training_data_path + "*.dat")
                self.cached_datfiles_original_size = len(self.cached_datfiles)

            labels = np.zeros(shape=(self.batch_size, self.image_height * self.image_width), dtype=np.uint8)
            data_x = np.zeros(shape=(self.batch_size, self.image_height, self.image_width, self.image_channels), dtype=np.uint8)

            for i in range(self.batch_size):
                training_id = np.random.choice(len(self.cached_datfiles), 1, replace=False)[0]
                dat_file = self.cached_datfiles[training_id]
                file_id = re.search("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", dat_file).group()

                leftEyeImagePath = self.training_data_path + file_id + "_CameraLeftEye.jpg"
                data_x[i] = misc.imread(leftEyeImagePath)

                labelsPath = self.training_data_path + file_id + "_labels.dat"
                labels[i] = np.fromfile(labelsPath, dtype=np.uint8, count=self.image_height*self.image_width).reshape((1, self.image_height * self.image_width))
                # TODO: This is guaranteed to mess up the indexes in self.last_batch_datfiles_indexes. Fix!
                if random() < self.probability_delete_example:
                    self._delete_training_data(training_id)

            self.batch_queue.put(SemanticSegmentationData(
                data_x,
                labels,
                self.label_count
            ))

    def _delete_training_data(self, cached_datfile_index):
        dat_file = self.cached_datfiles[cached_datfile_index]
        file_id = re.search("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", dat_file).group()

        leftEyeImagePath = self.training_data_path + file_id + "_CameraLeftEye.jpg"

        labelsPath = self.training_data_path + file_id + "_labels.dat"

        if os.path.exists(leftEyeImagePath):
            os.remove(leftEyeImagePath)
        if os.path.exists(labelsPath):
            os.remove(labelsPath)
        self.cached_datfiles.pop(cached_datfile_index)

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

        self.cached_datfiles = []

    def load_next_batch(self, delete_batch_source=False):
        semantic_segmentation_data = self.batch_queue.get(block=True)
        #if delete_batch_source:
            #for dat_index in self.last_batch_datfiles_indexes:
                # TODO: because of an earlier todo in this file, this needs to be commented out for the time being... Will impact the validation cost calculation slightly, but with 100 000 training data size it should not be so bad.
                # self._delete_training_data(dat_index)

        return semantic_segmentation_data

    def get_real_world_training_examples(self):
        return self._real_world_examples

    def _load_real_world_training(self):
        jpgFiles = glob.glob(self.real_world_test_path + "*.jpg")
        self._real_world_examples = np.zeros(shape=(len(jpgFiles), self.image_height, self.image_width, self.image_channels), dtype=np.uint8)
        for i in range(len(jpgFiles)):
            self._real_world_examples[i] = misc.imread(jpgFiles[i])