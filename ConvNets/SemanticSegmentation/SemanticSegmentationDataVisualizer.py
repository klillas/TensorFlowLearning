from scipy import misc
from skimage import data, color, io, img_as_float
import numpy as np

class SemanticSegmentationDataVisualizer:

    def export_image_with_labels(self, path, image_data, predicted_labels):
        misc.imsave(path, self.overlay_image_with_labels(image_data, predicted_labels))

    def overlay_image_with_labels(self, image_data, predicted_labels):
        color_mask = np.zeros((predicted_labels.shape[0], predicted_labels.shape[1], 3))
        color_mask[np.where(predicted_labels == 0)] = [128, 128, 128]
        color_mask[np.where(predicted_labels == 1)] = [128, 0, 0]
        color_mask[np.where(predicted_labels == 2)] = [128, 128, 0]
        color_mask[np.where(predicted_labels == 3)] = [0, 0, 192]
        color_mask[np.where(predicted_labels == 4)] = [128, 64, 128]
        color_mask[np.where(predicted_labels == 5)] = [64, 0, 128]


        alpha = 0.8
        # Convert the input image and color mask to Hue Saturation Value (HSV)
        # colorspace
        img_hsv = color.rgb2hsv(image_data)
        color_mask_hsv = color.rgb2hsv(color_mask)

        # Replace the hue and saturation of the original image
        # with that of the color mask
        img_hsv[..., 0] = color_mask_hsv[..., 0]
        img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

        img_masked = color.hsv2rgb(img_hsv)
        return img_masked

    def generate_ground_truth_image(self, predicted_labels):
        ground_truth = np.zeros((predicted_labels.shape[0], predicted_labels.shape[1], 3))
        ground_truth[np.where(predicted_labels == 0)] = [128, 128, 128]
        ground_truth[np.where(predicted_labels == 1)] = [128, 0, 0]
        ground_truth[np.where(predicted_labels == 2)] = [128, 128, 0]
        ground_truth[np.where(predicted_labels == 3)] = [0, 0, 192]
        ground_truth[np.where(predicted_labels == 4)] = [128, 64, 128]
        ground_truth[np.where(predicted_labels == 5)] = [64, 0, 128]

        return ground_truth
