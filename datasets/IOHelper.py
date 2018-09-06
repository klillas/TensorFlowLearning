import numpy as np

class IOHelper:
    def loadCsvData(self, file_path, delimiter=",", skip_header=1, labelColumn=-1):
        """
        Loads the data and labels from a csv file with , as delimiter
        :param file_path: Path to csv file
        :param delimiter: The delimiter character
        :param skip_header: 1 if skip first row, 0 otherwise
        :param labelColumn: Index of label column
        :return:
        Matrix of data
        Array of labels
        """
        csv_data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
        #csv_data = csv_data[0:1000]
        data = []
        labels = []
        # np.random.seed(12345)
        np.random.shuffle(csv_data)

        for d in csv_data:
            data.append(d[0:-1])
            labels.append([d[-1]])

        return np.array(data), np.array(labels)