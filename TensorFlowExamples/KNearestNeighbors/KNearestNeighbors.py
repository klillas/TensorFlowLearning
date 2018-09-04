import os
import numpy as np
import tensorflow as tf
from six.moves import xrange


class KNearestNeighbors:
    def run_learning(self, train_data_path, k):
        data, labels = self._load_data(train_data_path)
        m_tot = int(data.shape[0])
        m_train = int(m_tot * 0.85)
        m_test = int(m_tot - m_train)

        test_data = data[0:m_test]
        test_labels = labels[0:m_test]

        training_data = data[m_test:]
        training_labels = labels[m_test:]

        print("Training set statistics: ")
        print("Number of label 0: {}".format(training_labels.shape[0] - np.sum(training_labels)))
        print("Number of label 1: {}".format(np.sum(training_labels)))

        print("")
        print("Test set statistics: ")
        print("Number of label 0: {}".format(test_labels.shape[0] - np.sum(test_labels)))
        print("Number of label 1: {}".format(np.sum(test_labels)))

        training_tensor = tf.placeholder("float", training_data.shape)
        test_tensor = tf.placeholder("float", test_data.shape[1])

        _, knn_nearest_neighbors_indices = tf.nn.top_k(-tf.reduce_sum(tf.abs(tf.add(training_tensor, tf.negative(test_tensor))), axis = 1), k=k)
        knn_label_predictions = tf.gather(training_labels, knn_nearest_neighbors_indices)
        knn_prediction = tf.round(tf.divide(tf.reduce_sum(knn_label_predictions), k))

        with tf.Session() as session:
            missed = 0
            predictions = [0, 0]

            for i in xrange(len(test_data)):
                if test_labels[i] == 0:
                    continue

                knn_prediction_result = session.run(knn_prediction, feed_dict={training_tensor: training_data, test_tensor: test_data[i]})

                predictions[int(knn_prediction_result)] = predictions[int(knn_prediction_result)] + 1;

                if knn_prediction_result != test_labels[i]:
                    missed += 1
                    print("Miss number ", i, ". Prediction: ", knn_prediction_result, ". Actual: ", test_labels[i], ".")

                if i % 100 == 0:
                    print("Testing {} out of {}. {} predictions for label 0. {} predictions for label 1.".format(i, len(test_data), predictions[0], predictions[1]))

                tf.summary.FileWriter("../logs", session.graph)

        print("Missed: {} -- Total: {}".format(missed, len(test_data)))
        print("Incorrect label 0: ", predictions[0])
        print("Incorrect label 1: ", predictions[1])


    def _load_data(self, file_path):
        """
        Loads the data and labels from a csv file with , as delimiter
        :param file_path: Path to csv file
        :return:
        Matrix of data
        Array of labels
        """
        csv_data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
        data = []
        labels = []
        np.random.seed(12345)
        np.random.shuffle(csv_data)

        for d in csv_data:
            data.append(d[1:-1])
            labels.append(d[-1])

        return np.array(data), np.array(labels)