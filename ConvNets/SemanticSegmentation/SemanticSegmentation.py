import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
from ConvNets.SemanticSegmentation import SemanticSegmentationData, SemanticSegmentationTrainingDataLoader
from datetime import datetime
from datetime import timedelta
import sys

from memory_profiler import profile

from TensorFlowLearning.ConvNets.SemanticSegmentation.SemanticSegmentationDataVisualizer import \
    SemanticSegmentationDataVisualizer
from TensorFlowLearning.ConvNets.SemanticSegmentation.SemanticSegmentationModelFactory import \
    SemanticSegmentationModelFactory


class SemanticSegmentation:
    data_generator = None

    tf_ph_x = None
    tf_ph_labels = None
    tf_ph_labels_one_hot = None
    tf_ph_droput_keep_prob = None
    tf_ph_image = None
    tf_ph_learning_rate = None

    tf_tensor_cost = None
    tf_tensor_model = None
    tf_tensor_train = None

    tf_variable_global_step = None
    tf_variable_image = None
    tf_variable_best_cost = None

    tf_summary_image_predictions = None
    tf_summary_image_base_truth = None
    tf_summary_real_image_predictions = None

    tf_session = None
    tf_graph = None

    hyper_param_width = None
    hyper_param_height = None
    hyper_param_image_channels = None
    hyper_param_label_size = None
    hyper_param_learning_rate = None
    hyper_param_adaptive_learning_rate_active = None
    hyper_param_adaptive_learning_rate = None
    hyper_param_train_batch_size = None
    hyper_param_model_name = None
    hyper_param_load_existing_model = None
    hyper_param_save_model_interval_seconds = None
    hyper_param_dropout_keep_prob = None
    hyper_param_epoch_start = None
    hyper_param_validation_batch_size = None
    hyper_param_validation_every_n_steps = None
    hyper_param_max_epochs = None

    tf_model_saver = None

    validation_data = None

    semantic_segmentation_data_visualizer = None

    def initialize(
            self,
            data_generator: SemanticSegmentationTrainingDataLoader,
            image_height,
            image_width,
            image_channels,
            learning_rate,
            batch_size,
            hyper_param_model_name,
            load_existing_model,
            save_model_interval_seconds,
            dropout_keep_prob,
            validation_batch_size,
            validation_every_n_steps,
            adaptive_learning_rate_active,
            adaptive_learning_rate,
            max_epochs):
        self.data_generator = data_generator
        self.hyper_param_width = image_width
        self.hyper_param_height = image_height
        self.hyper_param_image_channels = image_channels
        self.hyper_param_label_size = data_generator.label_count
        self.hyper_param_learning_rate = learning_rate
        self.hyper_param_train_batch_size = batch_size
        self.hyper_param_model_name = hyper_param_model_name
        self.hyper_param_load_existing_model = load_existing_model
        self.hyper_param_save_model_interval_seconds = save_model_interval_seconds
        self.hyper_param_dropout_keep_prob = dropout_keep_prob
        self.hyper_param_validation_batch_size = validation_batch_size
        self.hyper_param_validation_every_n_steps = validation_every_n_steps
        self.hyper_param_adaptive_learning_rate_active = adaptive_learning_rate_active
        self.hyper_param_adaptive_learning_rate = adaptive_learning_rate
        self.hyper_param_max_epochs = max_epochs

        self.semantic_segmentation_data_visualizer = SemanticSegmentationDataVisualizer()

        model_factory = SemanticSegmentationModelFactory()

        #if load_existing_model == True:
            # Old examples could have been used in training. Delete everything before loading the validation data batch
            # TODO: Refactor this to be a setting, because you do not always want to delete all examples even when loading an existing model
            # self.data_generator.delete_all_existing_training_data()

        if self.hyper_param_validation_batch_size % self.hyper_param_train_batch_size != 0:
            raise ValueError("Train batch size must be evenly divisible with validation batch size")

        #self.validation_data = np.zeros((self.hyper_param_validation_batch_size, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels))
        self.validation_data = []
        for batch_index in range(self.hyper_param_validation_batch_size)[0::self.hyper_param_train_batch_size]:
            self.validation_data.append(self.data_generator.load_next_batch(delete_batch_source=True))

        self.session = tf.Session()

        if self.hyper_param_load_existing_model == False:
            self.tf_ph_x = tf.placeholder(tf.float32, [None, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels], name="x")
            self.tf_ph_labels = tf.placeholder(tf.int32, [None, self.hyper_param_height * self.hyper_param_width], name="labels")
            self.tf_ph_labels_one_hot = tf.placeholder(tf.int32, [None, self.hyper_param_height * self.hyper_param_width, self.hyper_param_label_size], name="labels_one_hot")
            self.tf_ph_droput_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.tf_ph_image = tf.placeholder(tf.float32, (self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels), name="image")
            self.tf_ph_learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
            self.tf_variable_global_step = tf.Variable(0, trainable=False, name='global_step')
            self.tf_variable_best_cost = tf.Variable(sys.float_info.max, trainable=False, name='best_cost')
            self.tf_variable_image = tf.Variable(np.zeros((50, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels)), name="variable_image")

        if self.hyper_param_load_existing_model == True:
            loader_path = "./stored_models/"

            file_to_restore = tf.train.latest_checkpoint(loader_path) + ".meta"
            self.tf_model_saver = tf.train.import_meta_graph(file_to_restore)
            self.tf_model_saver.restore(self.session, tf.train.latest_checkpoint(loader_path))
            self.tf_graph = self.session.graph

            self.tf_variable_image = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == 'variable_image:0'][0]
            self.tf_ph_x = self.tf_graph.get_tensor_by_name("x:0")
            self.tf_ph_labels = self.tf_graph.get_tensor_by_name("labels:0")
            self.tf_ph_labels_one_hot = self.tf_graph.get_tensor_by_name("labels_one_hot:0")
            self.tf_ph_droput_keep_prob = self.tf_graph.get_tensor_by_name("dropout_keep_prob:0")
            self.tf_ph_image = self.tf_graph.get_tensor_by_name("image:0")
            self.tf_variable_global_step = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == 'global_step:0'][0]
            self.tf_variable_best_cost = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == 'best_cost:0'][0]
            self.tf_ph_learning_rate = self.tf_graph.get_tensor_by_name("learning_rate:0")

        self.tf_summary_image_predictions = tf.summary.image("example predictions", self.tf_variable_image, 50)
        self.tf_summary_image_base_truth = tf.summary.image("example base truth", self.tf_variable_image, 50)
        self.tf_summary_real_image_predictions = tf.summary.image("real world predictions", self.tf_variable_image, self.data_generator.get_real_world_training_examples().shape[0]*2)

        #self._initialize_model_minimal()
        self._initialize_model_U_net()
        #self._initialize_test_model()
        self.tf_tensor_cost = model_factory.initialize_cost(
            load_existing_model=self.hyper_param_load_existing_model,
            model=self.tf_tensor_model,
            labels_one_hot=self.tf_ph_labels_one_hot,
            graph=self.tf_graph)
        self._initialize_optimization()
        self._initialize_predictor()

        if self.hyper_param_load_existing_model == False:
            self.session.run(tf.global_variables_initializer())
        self.tf_model_saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=1)

    def train_own_model(self):
        summary_merged = tf.summary.merge_all()
        time_model_last_saved = datetime.now()

        self.session.graph.finalize()

        writer_train = tf.summary.FileWriter("./logs/" + self.hyper_param_model_name + "/train", self.session.graph)
        writer_validation = tf.summary.FileWriter("./logs/" + self.hyper_param_model_name + "/validation")

        last_iteration_cost = sys.float_info.max
        for i in range(self.session.run(self.tf_variable_global_step), self.hyper_param_max_epochs):
            data_train = self.data_generator.load_next_batch()
            if i % 25 == 0:
                print("Training step {}".format(i))
            self.tf_variable_global_step.load(i, session=self.session)
            #####Train#####
            #self.exportImageWithLabels("c:/temp/pic.jpg", 0,  np.reshape(self.data_train.labels[0], (self.hyper_param_height, self.hyper_param_width)))

            self.hyper_param_learning_rate
            _ = self.session.run(self.tf_tensor_train, feed_dict={
                self.tf_ph_x: data_train.data_x,
                self.tf_ph_labels_one_hot: data_train.labels_one_hot,
                self.tf_ph_droput_keep_prob: self.hyper_param_dropout_keep_prob,
                self.tf_ph_learning_rate: self.hyper_param_learning_rate
            })

            batch_training_cost = self.session.run(self.tf_tensor_cost, feed_dict={
                self.tf_ph_x: data_train.data_x,
                self.tf_ph_labels_one_hot: data_train.labels_one_hot,
                self.tf_ph_droput_keep_prob: 1.0
            })

            if self.hyper_param_adaptive_learning_rate_active:
                if batch_training_cost < last_iteration_cost:
                    self.hyper_param_learning_rate = self.hyper_param_learning_rate * (1 + self.hyper_param_adaptive_learning_rate)
                else:
                    self.hyper_param_learning_rate = self.hyper_param_learning_rate * (1 - self.hyper_param_adaptive_learning_rate)
                print("New learning rate {0}".format(self.hyper_param_learning_rate))
                last_iteration_cost = batch_training_cost

            costs_summary = tf.Summary()
            costs_summary.value.add(tag="Cost", simple_value=batch_training_cost)
            writer_train.add_summary(costs_summary, tf.train.global_step(self.session, self.tf_variable_global_step))

            if (i % self.hyper_param_validation_every_n_steps == 0):
                # TODO: Just here now to avoid runtime error. Needs to use its separate batch of validation examples
                data_validation = self.data_generator.load_next_batch()
                print("")
                print("")
                print("#################################################################")
                print("Train batch cost: {}".format(batch_training_cost))
                print("")
                print("Validation test")
                self._calculcate_and_log_statistics(
                    "Validation cost",
                    writer_validation,
                    data_validation,
                    plotPrediction=True,
                    calculate_cost=True
                )

                writer_validation.flush()

            #if (i % 5 == 0):
            #    self._calculcate_and_log_statistics(
            #        "Train cost",
            #        writer_train,
            #        # TODO: Use only a static subset of the total trainset now. Fix an incremental train calculation to get rid of this limitation.
            #        self.data_train,
            #        plotPrediction = False,
            #        calculate_cost = True
            #    )
            #    writer_train.flush()

            #if datetime.now() > (time_model_last_saved + timedelta(seconds=self.hyper_param_save_model_interval_seconds)) \
            #        and batch_training_cost < self.session.run(self.tf_variable_best_cost):
            if datetime.now() > (time_model_last_saved + timedelta(seconds=self.hyper_param_save_model_interval_seconds)):
                time_model_last_saved = datetime.now()
                self.tf_variable_best_cost.load(batch_training_cost, session=self.session)
                self.tf_model_saver.save(
                    sess=self.session,
                    save_path="./stored_models/" + self.hyper_param_model_name,
                    global_step=tf.train.global_step(self.session, self.tf_variable_global_step))
                print("New lowest cost found, model state saved")


    def _calculcate_and_log_statistics(self, cost_description, summary_writer, semantic_segmentation_data: SemanticSegmentationData, plotPrediction, calculate_cost):
        if calculate_cost == True:
            cost_batches = []
            semantic_segmentation_data = self.validation_data
            for j in range((int)(self.hyper_param_validation_batch_size / self.hyper_param_train_batch_size)):
                # print("Calculating cost of example {}".format(j))
                training_cost_item = self.session.run(self.tf_tensor_cost, feed_dict={
                    self.tf_ph_x: semantic_segmentation_data[j].data_x,
                    self.tf_ph_labels_one_hot: semantic_segmentation_data[j].labels_one_hot,
                    self.tf_ph_labels: semantic_segmentation_data[j].labels,
                    self.tf_ph_droput_keep_prob: 1.0
                })
                cost_batches.append(training_cost_item)

            average_cost = np.mean(cost_batches)

            costs_summary = tf.Summary()
            costs_summary.value.add(tag="Cost", simple_value=average_cost)
            summary_writer.add_summary(costs_summary, tf.train.global_step(self.session, self.tf_variable_global_step))


        if plotPrediction == True:
            # TODO: Make the amount of predictions user definable (All the static 50 assignments here and in the initializer
            # TODO: This only works if the batch size is equal to 50 or evenly divisible, so have to fix that
            prediction_batches = np.zeros((50, self.hyper_param_height * self.hyper_param_width))
            image_batches = np.zeros((50, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels))
            overlay_image = np.zeros((50, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels))
            ground_truth_batches = np.zeros((50, self.hyper_param_height * self.hyper_param_width))
            for j in range(50)[0::self.hyper_param_train_batch_size]:
                semantic_segmentation_data = self.data_generator.load_next_batch(delete_batch_source=True)
                batch_predictor = self.session.run(self.tf_tensor_predictor, feed_dict={
                    self.tf_ph_x: semantic_segmentation_data.data_x,
                    self.tf_ph_droput_keep_prob: 1.0
                })
                prediction_batches[j:j + self.hyper_param_train_batch_size] = batch_predictor
                image_batches[j:j + self.hyper_param_train_batch_size] = semantic_segmentation_data.data_x
                ground_truth_batches[j:j + self.hyper_param_train_batch_size] = semantic_segmentation_data.labels

            for i in range(50):
                overlay_image[i] = self.semantic_segmentation_data_visualizer.generate_ground_truth_image(np.reshape(prediction_batches[i], (self.hyper_param_height, self.hyper_param_width)))
            self.tf_variable_image.load(overlay_image, self.session)

            image_summary = self.session.run(self.tf_summary_image_predictions)
            summary_writer.add_summary(image_summary, tf.train.global_step(self.session, self.tf_variable_global_step))

            for i in range(0, 50, 2):
                overlay_image[i] = self.semantic_segmentation_data_visualizer.generate_ground_truth_image(np.reshape(ground_truth_batches[i], (self.hyper_param_height, self.hyper_param_width)))
                overlay_image[i+1] = self.semantic_segmentation_data_visualizer.overlay_image_with_labels(image_batches[i], np.reshape(ground_truth_batches[i], (self.hyper_param_height, self.hyper_param_width)))
            self.tf_variable_image.load(overlay_image, self.session)

            image_summary = self.session.run(self.tf_summary_image_base_truth)
            summary_writer.add_summary(image_summary, tf.train.global_step(self.session, self.tf_variable_global_step))


            # TODO: Make the amount of predictions user definable (All the static 50 assignments here and in the initializer
            # TODO: This only works if the batch size is equal to 50 or evenly divisible, so have to fix that
            real_world_examples = self.data_generator.get_real_world_training_examples()
            overlay_image = np.zeros((50, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels))
            for j in range(0, real_world_examples.shape[0] * 2, 2):
                image_prediction = self.session.run(self.tf_tensor_predictor, feed_dict={
                    self.tf_ph_x: np.reshape(real_world_examples[int(j/2)], (1, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels)),
                    self.tf_ph_labels: semantic_segmentation_data.labels,
                    self.tf_ph_droput_keep_prob: 1.0
                })
                overlay_image[j] = self.semantic_segmentation_data_visualizer.generate_ground_truth_image(np.reshape(image_prediction, (self.hyper_param_height, self.hyper_param_width)))
                overlay_image[j+1] = self.semantic_segmentation_data_visualizer.overlay_image_with_labels(
                    real_world_examples[int(j/2)],
                    np.reshape(image_prediction, (self.hyper_param_height, self.hyper_param_width)))

            self.tf_variable_image.load(overlay_image, self.session)

            image_summary = self.session.run(self.tf_summary_real_image_predictions)
            summary_writer.add_summary(image_summary, tf.train.global_step(self.session, self.tf_variable_global_step))

        print("")

    def _initialize_model_U_net(self):
        if self.hyper_param_load_existing_model == False:
            model_factory = SemanticSegmentationModelFactory()
            self.tf_tensor_model = model_factory.initialize_model_U_net(self.hyper_param_label_size, self.tf_ph_x, self.tf_ph_droput_keep_prob)

        if self.hyper_param_load_existing_model == True:
            self.tf_tensor_model = self.tf_graph.get_tensor_by_name("fcn_logits:0")


    def _initialize_predictor(self):
        if self.hyper_param_load_existing_model == False:
            self.tf_tensor_predictor = tf.argmax(input=self.tf_tensor_model, axis=2, output_type=tf.int32, name="predictor")

        if self.hyper_param_load_existing_model == True:
            self.tf_tensor_predictor = self.tf_graph.get_tensor_by_name("predictor:0")

    def _initialize_optimization(self):
        if self.hyper_param_load_existing_model == False:
            # Calculate distance from actual labels using cross entropy
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.tf_tensor_model,
                labels=self.tf_ph_labels_one_hot)
            # Take mean for total loss
            loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

            # The model implements this operation to find the weights/parameters that would yield correct pixel labels
            self.tf_tensor_train = tf.train.AdamOptimizer(learning_rate=self.tf_ph_learning_rate).minimize(loss_op, name="fcn_train_op")

        if self.hyper_param_load_existing_model == True:
            self.tf_tensor_train = self.tf_graph.get_operation_by_name("fcn_train_op")


    def predict_and_create_image(self, path, image_data):
        prediction = self.session.run(self.tf_tensor_predictor, feed_dict={
            self.tf_ph_x: np.reshape(image_data, (-1, image_data.shape[0], image_data.shape[1], image_data.shape[2])),
            self.tf_ph_droput_keep_prob: 1.0
        })

        self.semantic_segmentation_data_visualizer.export_image_with_labels(path, image_data, np.reshape(prediction, (self.hyper_param_height, self.hyper_param_width)))