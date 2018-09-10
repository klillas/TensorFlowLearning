import numpy as np

class IOHelper:
    def loadCsvData(self, file_path, trainPart=0.9, validationPart=0.1, delimiter=",", skip_header=1, labelColumn=-1):
        """
        Loads the data and labels from a csv file with , as delimiter
        :param file_path: Path to csv file
        :param delimiter: The delimiter character
        :param skip_header: 1 if skip first row, 0 otherwise
        :param labelColumn: Index of label column
        :return:
        Matrix of training data
        Array of taining labels
        Matrix of validation data
        Array of validation label
        """
        csv_data = np.genfromtxt(file_path, delimiter=delimiter, skip_header=skip_header)
        train_size = csv_data.shape[0]
        #csv_data = csv_data[0:1000]
        training_data = []
        training_labels = []
        validation_data = []
        validation_labels = []
        # np.random.seed(12345)
        np.random.shuffle(csv_data)

        for d in csv_data[0:int(train_size*trainPart)]:
            training_data.append(d[0:-1])
            training_labels.append([d[-1]])

        for d in csv_data[int(train_size*trainPart):]:
            validation_data.append(d[0:-1])
            validation_labels.append([d[-1]])

        return np.array(training_data), np.array(training_labels), np.array(validation_data), np.array(validation_labels)