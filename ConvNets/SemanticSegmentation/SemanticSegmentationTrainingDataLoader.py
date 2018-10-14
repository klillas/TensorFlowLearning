from scipy import misc
import numpy as np

from ConvNets.SemanticSegmentation.SemanticSegmentationData import SemanticSegmentationData


class SemanticSegmentationTrainingDataLoader:

    def generate_traindata_from_depthvision_pictures(self):
        trainingDataPath = "C:/temp/training/"
        data_set_size = 5000
        image_width = 256
        image_height = 192
        image_channels = 3
        label_count = 2
        training_set_ratio = 0.98
        labels = np.zeros(shape=(data_set_size, image_height * image_width), dtype=np.uint8)
        data_x = np.zeros(shape=(data_set_size, image_height, image_width, image_channels),dtype=np.uint8)
        for i in range(data_set_size):
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