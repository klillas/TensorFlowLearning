from scipy import misc
import numpy as np

class SemanticSegmentationTrainingDataLoader:

    def generate_traindata_from_depthvision_pictures(self):
        trainingDataPath = "C:/temp/training/"
        training_size = 50
        image_width = 545
        image_height = 358
        image_channels = 3
        labels = np.zeros(shape=(training_size, image_height, image_width), dtype=np.uint8)
        training_data = np.zeros(shape=(training_size, image_height, image_width, image_channels), dtype=np.uint8)
        for i in range(labels.shape[0]):
            leftEyeImage = misc.imread(trainingDataPath + str(i) + "_CameraLeftEye.jpg")
            #rightEyeImage = misc.imread(trainingDataPath + str(i) + "_CameraRightEye.jpg")
            #bothEyes = np.concatenate((leftEyeImage, rightEyeImage))
            training_data[i] = leftEyeImage

            # labels[i] = np.loadtxt(fname=trainingDataPath + str(i) + "_labels.dat", dtype=np.uint8).reshape((1, image_height, image_width))
            labels[i] = np.fromfile(trainingDataPath + str(i) + "_labels.dat", dtype=np.uint8, count=image_width*image_height).reshape((1, image_height, image_width))
        return training_data, labels, training_size, image_height, image_width, image_channels