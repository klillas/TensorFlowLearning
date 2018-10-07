import tensorflow as tf
import numpy as np
#impot scikit as sc
from ConvNets.SemanticSegmentation import SemanticSegmentationData


class SemanticSegmentation:
    data_train = None
    data_validation = None

    tf_ph_x = None
    tf_ph_labels = None
    tf_ph_labels_one_hot = None
    tf_ph_droput_keep_prob = None

    tf_tensor_cost = None
    tf_tensor_model = None
    tf_tensor_train = None
    tf_tensor_correctness = None
    tf_tensor_global_step = None

    hyper_param_width = None
    hyper_param_height = None
    hyper_param_image_channels = None
    hyper_param_label_size = None
    hyper_param_learning_rate = None
    hyper_param_train_batch_size = None
    hyper_param_model_name = None

    def initialize(self, data_train: SemanticSegmentationData, data_validation: SemanticSegmentationData, image_height, image_width, image_channels, learning_rate, batch_size, hyper_param_model_name):
        self.hyper_param_width = image_width
        self.hyper_param_height = image_height
        self.hyper_param_image_channels = image_channels
        self.hyper_param_label_size = data_train.label_count
        self.hyper_param_learning_rate = learning_rate
        self.hyper_param_train_batch_size = batch_size
        self.hyper_param_model_name = hyper_param_model_name

        self.data_train = data_train
        self.data_validation = data_validation

        self.tf_ph_x = tf.placeholder(tf.float32, [None, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels], name="x")
        self.tf_ph_labels = tf.placeholder(tf.int32, [None, self.hyper_param_height * self.hyper_param_width], name="labels")
        self.tf_ph_labels_one_hot = tf.placeholder(tf.int32, [None, self.hyper_param_height * self.hyper_param_width, self.hyper_param_label_size], name="labels")
        self.tf_ph_droput_keep_prob = tf.placeholder(tf.float32)

        self.tf_tensor_global_step = tf.Variable(0, trainable=False, name='global_step')

        self._initialize_model()
        self._initialize_cost()
        self._initialize_optimization()
        self._initialize_predictor()

    def train_own_model(self):
        init = tf.global_variables_initializer()
        summary_merged = tf.summary.merge_all()

        with tf.Session() as session:
            session.run(init)

            writer_train = tf.summary.FileWriter("./logs/" + self.hyper_param_model_name + "/train", session.graph)
            writer_validation = tf.summary.FileWriter("./logs/" + self.hyper_param_model_name + "/validation")

            for i in range(0, 1000000):
                #####Train#####
                session.run(tf.assign(self.tf_tensor_global_step, i))
                if self.hyper_param_train_batch_size > 0:
                    trainIds = np.random.randint(
                        self.data_train.data_x.shape[0],
                        size=self.hyper_param_train_batch_size)
                else:
                    trainIds = np.random.randint(
                        self.data_train.data_x.shape[0],
                        size=self.data_train.data_x.shape[0])

                result = session.run(self.tf_tensor_train, feed_dict={
                    self.tf_ph_x: self.data_train.data_x[trainIds, :],
                    self.tf_ph_labels_one_hot: self.data_train.labels_one_hot[trainIds, :],
                    self.tf_ph_droput_keep_prob: 0.5
                })

                if (i % 100 == 0):
                    print("")
                    print("")
                    print("#################################################################")
                    print("Validation test")
                    self._calculcate_and_log_statistics(
                        "Train cost",
                        session,
                        writer_train,
                        self.data_train.data_x,
                        self.data_train.labels,
                        self.data_train.labels_one_hot
                    )

                    self._calculcate_and_log_statistics(
                        "Validation cost",
                        session,
                        writer_validation,
                        self.data_validation.data_x,
                        self.data_validation.labels,
                        self.data_validation.labels_one_hot
                    )

                    writer_train.flush()
                    writer_validation.flush()

    def _calculcate_and_log_statistics(self, cost_description, session, summary_writer, data_x, labels, labels_one_hot):
        cost_batches = []
        correctness_batches = []
        prediction_batches = np.zeros(labels.shape)
        for j in range(data_x.shape[0])[0::self.hyper_param_train_batch_size]:
            training_cost_item, batch_correctness, batch_predictor = session.run([self.tf_tensor_cost, self.tf_tensor_correctness, self.tf_tensor_predictor], feed_dict={
                self.tf_ph_x: data_x[j:j + self.hyper_param_train_batch_size],
                self.tf_ph_labels_one_hot: labels_one_hot[j:j + self.hyper_param_train_batch_size],
                self.tf_ph_labels: labels[j:j + self.hyper_param_train_batch_size],
                self.tf_ph_droput_keep_prob: 1.0
            })
            cost_batches.append(training_cost_item)
            correctness_batches.append(batch_correctness)
            prediction_batches[j:j + self.hyper_param_train_batch_size] = batch_predictor

        average_cost = np.mean(cost_batches)
        average_correctness = np.mean(correctness_batches)

        costs_summary = tf.Summary()
        costs_summary.value.add(tag="Cost", simple_value=average_cost)
        summary_writer.add_summary(costs_summary, tf.train.global_step(session, self.tf_tensor_global_step))
        print("Statistics for {}".format(cost_description))
        print("Average cost = {}".format(average_cost))

        correctness_summary = tf.Summary()
        correctness_summary.value.add(tag="Correctness", simple_value=average_correctness)
        summary_writer.add_summary(correctness_summary, tf.train.global_step(session, self.tf_tensor_global_step))
        print("Correctness = {}".format(average_correctness))

        total_label_1_correct_predictions = 0
        for i in range(labels.shape[0]):
            if (np.where(prediction_batches[i] == 1)[0].shape[0] > 0):
                total_label_1_correct_predictions = total_label_1_correct_predictions + np.where(np.take(labels, np.where(prediction_batches[0] == 1)[0]) == 1)[0].shape[0]
        print("Label 1 correct guess percentage {}".format(total_label_1_correct_predictions / np.where(prediction_batches == 1)[0].shape[0]))
        #print("Total percentage of correctly labelled pixels: {}".format(np.where(prediction_batches == labels)[0].shape[0] / (labels.shape[0] * labels.shape[1])))

        print("")
        print("")


    def _initialize_predictor(self):
        self.tf_tensor_predictor = tf.argmax(input=self.tf_tensor_model, axis=2, output_type=tf.int32)
        self.tf_tensor_correctness = tf.count_nonzero(tf.equal(self.tf_tensor_predictor, self.tf_ph_labels)) / tf.cast(tf.shape(self.tf_ph_labels)[0] * tf.shape(self.tf_ph_labels)[1], dtype=tf.int64)

    def _initialize_optimization(self):
        # Calculate distance from actual labels using cross entropy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.tf_tensor_model,
            labels=self.tf_ph_labels_one_hot)
        # Take mean for total loss
        loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

        # The model implements this operation to find the weights/parameters that would yield correct pixel labels
        self.tf_tensor_train = tf.train.AdamOptimizer(learning_rate=self.hyper_param_learning_rate).minimize(loss_op, name="fcn_train_op")


    def _initialize_model(self):
        model = tf.layers.conv2d(
            inputs=self.tf_ph_x,
            filters=16,
            kernel_size=[4, 4],
            padding="same",
            activation=tf.nn.relu
        )

        model = tf.layers.max_pooling2d(
            inputs=model,
            pool_size=[4, 4],
            strides=4
        )

        model = tf.layers.conv2d(
            inputs=model,
            filters=32,
            kernel_size=[4, 4],
            padding="same",
            activation=tf.nn.relu
        )

        model = tf.layers.max_pooling2d(
            inputs=model,
            pool_size=[4, 4],
            strides=4
        )

        model = tf.layers.conv2d(
            inputs=model,
            filters=64,
            kernel_size=[4, 4],
            padding="same",
            activation=tf.nn.relu
        )

        model = tf.layers.max_pooling2d(
            inputs=model,
            pool_size=[4, 4],
            strides=4
        )

        model = tf.reshape(
            model,
            (-1, model.shape[1] * model.shape[2] * model.shape[3]))

        model = tf.nn.dropout(
            model,
            keep_prob=self.tf_ph_droput_keep_prob
        )

        model = tf.layers.dense(
            inputs=model,
            units=24*32
        )

        model = tf.reshape(
            model,
            (-1, 24, 32, 1))

        model = tf.layers.conv2d_transpose(
            model,
            filters=self.hyper_param_label_size,
            kernel_size=16,
            strides=(32, 32),
            padding='SAME')

        model = tf.reshape(
            model,
            (-1, model.shape[1] * model.shape[2], model.shape[3]),
            name="fcn_logits")

        self.tf_tensor_model = model

    def _initialize_cost(self):
        # Calculate distance from actual labels using cross entropy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.tf_tensor_model, labels=self.tf_ph_labels_one_hot)
        # Take mean for total loss
        self.tf_tensor_cost = tf.reduce_mean(cross_entropy, name="fcn_loss")