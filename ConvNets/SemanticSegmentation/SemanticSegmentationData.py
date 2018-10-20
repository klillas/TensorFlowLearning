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
        self.labels_one_hot = np.eye(self.label_count)[self.labels]

    def exportImageWithLabels(self, path, id, predicted_labels):
        misc.imsave(path, self.data_x[id])

    def overlay_image_with_labels(self, image_data, predicted_labels):
        color_mask = np.zeros((predicted_labels.shape[0], predicted_labels.shape[1], 3))
        color_mask[np.where(predicted_labels == 1)] = [1, 0, 0]
        color_mask[np.where(predicted_labels == 2)] = [0, 1, 0]
        color_mask[np.where(predicted_labels == 3)] = [0, 0, 1]

        alpha = 0.7
        # Convert the input image and color mask to Hue Saturation Value (HSV)
        # colorspace
        img_hsv = color.rgb2hsv(image_data)
        color_mask_hsv = color.rgb2hsv(color_mask)

        # Replace the hue and saturation of the original image
        # with that of the color mask
        img_hsv[..., 0] = color_mask_hsv[..., 0]
        img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

        img_masked = color.hsv2rgb(img_hsv)

        # Display the output
        #f, (ax0) = plt.subplots(1, 1, subplot_kw={'xticks': [], 'yticks': []})
        #ax0.imshow(self.data_x[id], cmap=plt.cm.gray)
        #ax1.imshow(color_mask)
        #ax0.imshow(img_masked)
        #plt.show()

        return img_masked