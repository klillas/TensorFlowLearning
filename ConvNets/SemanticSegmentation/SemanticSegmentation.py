import tensorflow as tf
import numpy as np
from ConvNets.SemanticSegmentation import SemanticSegmentationData
from datetime import datetime
from datetime import timedelta


class SemanticSegmentation:
    data_train = None
    data_validation = None

    tf_ph_x = None
    tf_ph_labels = None
    tf_ph_labels_one_hot = None
    tf_ph_droput_keep_prob = None
    tf_ph_image = None

    tf_tensor_cost = None
    tf_tensor_model = None
    tf_tensor_train = None

    tf_variable_global_step = None
    tf_variable_image = None

    tf_summary_image_predictions = None

    tf_session = None
    tf_graph = None

    hyper_param_width = None
    hyper_param_height = None
    hyper_param_image_channels = None
    hyper_param_label_size = None
    hyper_param_learning_rate = None
    hyper_param_train_batch_size = None
    hyper_param_model_name = None
    hyper_param_load_existing_model = None
    hyper_param_save_model_interval_seconds = None
    hyper_param_dropout_keep_prob = None
    hyper_param_epoch_start = None

    tf_model_saver = None

    def initialize(
            self,
            data_train: SemanticSegmentationData,
            data_validation: SemanticSegmentationData,
            image_height,
            image_width,
            image_channels,
            learning_rate,
            batch_size,
            hyper_param_model_name,
            load_existing_model,
            save_model_interval_seconds,
            dropout_keep_prob):
        self.hyper_param_width = image_width
        self.hyper_param_height = image_height
        self.hyper_param_image_channels = image_channels
        self.hyper_param_label_size = data_train.label_count
        self.hyper_param_learning_rate = learning_rate
        self.hyper_param_train_batch_size = batch_size
        self.hyper_param_model_name = hyper_param_model_name
        self.hyper_param_load_existing_model = load_existing_model
        self.hyper_param_save_model_interval_seconds = save_model_interval_seconds
        self.hyper_param_dropout_keep_prob = dropout_keep_prob

        self.data_train = data_train
        self.data_validation = data_validation

        self.session = tf.Session()

        if self.hyper_param_load_existing_model == False:
            self.tf_ph_x = tf.placeholder(tf.float32, [None, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels], name="x")
            self.tf_ph_labels = tf.placeholder(tf.int32, [None, self.hyper_param_height * self.hyper_param_width], name="labels")
            self.tf_ph_labels_one_hot = tf.placeholder(tf.int32, [None, self.hyper_param_height * self.hyper_param_width, self.hyper_param_label_size], name="labels_one_hot")
            self.tf_ph_droput_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.tf_ph_image = tf.placeholder(tf.float32, (self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels), name="image")
            self.tf_variable_global_step = tf.Variable(0, trainable=False, name='global_step')
            self.tf_variable_image = tf.Variable(np.zeros((30, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels)), name="variable_image")

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

        self.tf_summary_image_predictions = tf.summary.image("example prediction", self.tf_variable_image, 30)

        self._initialize_model_minimal()
        self._initialize_cost()
        self._initialize_optimization()
        self._initialize_predictor()

        if self.hyper_param_load_existing_model == False:
            self.session.run(tf.global_variables_initializer())
            self.tf_model_saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)


    def train_own_model(self):
        summary_merged = tf.summary.merge_all()
        time_model_last_saved = datetime.now()

        self.session.graph.finalize()

        writer_train = tf.summary.FileWriter("./logs/" + self.hyper_param_model_name + "/train", self.session.graph)
        writer_validation = tf.summary.FileWriter("./logs/" + self.hyper_param_model_name + "/validation")

        batch_training_cost_buffer = np.zeros(((int) (self.data_train.data_x.shape[0] / self.hyper_param_train_batch_size)))
        first_iteration = True

        for i in range(self.session.run(self.tf_variable_global_step), 1000000):
            print("Training step {}".format(i))
            self.tf_variable_global_step.load(i, session=self.session)
            #####Train#####
            if self.hyper_param_train_batch_size > 0:
                trainIds = np.random.randint(
                    self.data_train.data_x.shape[0],
                    size=self.hyper_param_train_batch_size)
            else:
                trainIds = np.random.randint(
                    self.data_train.data_x.shape[0],
                    size=self.data_train.data_x.shape[0])

            _, batch_training_cost = self.session.run([self.tf_tensor_train, self.tf_tensor_cost], feed_dict={
                self.tf_ph_x: self.data_train.data_x[trainIds, :],
                self.tf_ph_labels_one_hot: self.data_train.labels_one_hot[trainIds, :],
                self.tf_ph_droput_keep_prob: self.hyper_param_dropout_keep_prob
            })

            if first_iteration == True:
                # TODO: This is a temporary solution until we can store the actual cost in a tensor and restore it
                batch_training_cost_buffer.fill(batch_training_cost)
            np.roll(batch_training_cost_buffer, 1)
            batch_training_cost_buffer[0] = batch_training_cost
            mean_training_cost = np.mean(batch_training_cost_buffer)

            costs_summary = tf.Summary()
            costs_summary.value.add(tag="Cost", simple_value=mean_training_cost)
            writer_train.add_summary(costs_summary, tf.train.global_step(self.session, self.tf_variable_global_step))

            if (i % 5 == 0):
                print("")
                print("")
                print("#################################################################")
                print("Train batch cost: {}".format(mean_training_cost))
                print("")
                print("Validation test")
                self._calculcate_and_log_statistics(
                    "Validation cost",
                    writer_validation,
                    self.data_validation,
                    plotPrediction = True,
                    calculate_cost = True
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

            if datetime.now() > (time_model_last_saved + timedelta(seconds=self.hyper_param_save_model_interval_seconds)):
                time_model_last_saved = datetime.now()
                self.tf_model_saver.save(
                    sess=self.session,
                    save_path="./stored_models/" + self.hyper_param_model_name,
                    global_step=tf.train.global_step(self.session, self.tf_variable_global_step))
                print("Model state saved")

            first_iteration = False

    def _calculcate_and_log_statistics(self, cost_description, summary_writer, semantic_segmentation_data: SemanticSegmentationData, plotPrediction, calculate_cost):
        if calculate_cost == True:
            prediction_batches = np.zeros(semantic_segmentation_data.labels.shape)
            #cost_batches = np.zeros(semantic_segmentation_data.data_x.shape[0])
            cost_batches = []
            for j in range(semantic_segmentation_data.data_x.shape[0])[0::self.hyper_param_train_batch_size]:
                # print("Calculating cost of example {}".format(j))
                training_cost_item, batch_predictor = self.session.run([self.tf_tensor_cost, self.tf_tensor_predictor], feed_dict={
                    self.tf_ph_x: semantic_segmentation_data.data_x[j:j + self.hyper_param_train_batch_size],
                    self.tf_ph_labels_one_hot: semantic_segmentation_data.labels_one_hot[j:j + self.hyper_param_train_batch_size],
                    self.tf_ph_labels: semantic_segmentation_data.labels[j:j + self.hyper_param_train_batch_size],
                    self.tf_ph_droput_keep_prob: 1.0
                })
                cost_batches.append(training_cost_item)
                prediction_batches[j:j + self.hyper_param_train_batch_size] = batch_predictor

            average_cost = np.mean(cost_batches)

            costs_summary = tf.Summary()
            costs_summary.value.add(tag="Cost", simple_value=average_cost)
            summary_writer.add_summary(costs_summary, tf.train.global_step(self.session, self.tf_variable_global_step))
            print("Statistics for {}".format(cost_description))
            print("Average cost = {}".format(average_cost))
            #semantic_segmentation_data.exportImage("c:/temp/picture.jpg", 0, prediction_batches[0])


        if plotPrediction == True:
            overlay_image = np.zeros((30, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels))
            for i in range(30):
                overlay_image[i] = semantic_segmentation_data.overlay_image_with_labels(i, np.reshape(prediction_batches[i], (self.hyper_param_height, self.hyper_param_width)))
            self.tf_variable_image.load(overlay_image, self.session)

            image_summary = self.session.run(self.tf_summary_image_predictions)
            summary_writer.add_summary(image_summary, tf.train.global_step(self.session, self.tf_variable_global_step))

            #for i in range(30):
            #    overlay_image[i] = semantic_segmentation_data.overlay_image_with_labels(i, np.reshape(semantic_segmentation_data.labels[i], (self.hyper_param_height, self.hyper_param_width)))
            #self.tf_variable_image.load(overlay_image, session)

            #image_summary = session.run(self.tf_summary_image_predictions)
            #summary_writer.add_summary(image_summary, tf.train.global_step(session, self.tf_variable_global_step))

            print("")

        print("")
        print("")


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
            self.tf_tensor_train = tf.train.AdamOptimizer(learning_rate=self.hyper_param_learning_rate).minimize(loss_op, name="fcn_train_op")

        if self.hyper_param_load_existing_model == True:
            self.tf_tensor_train = self.tf_graph.get_operation_by_name("fcn_train_op")


    def _initialize_model_minimal(self):
        if self.hyper_param_load_existing_model == False:
            model = tf.layers.conv2d(
                inputs=self.tf_ph_x,
                filters=2,
                kernel_size=[8, 8],
                padding="same",
                activation=tf.nn.relu,
                name="Layer1"
            )
            model_conv1_lowres = tf.layers.max_pooling2d(
                inputs=model,
                pool_size=[8, 8],
                strides=8
            )
            model_conv1_lowres_flat = tf.reshape(model_conv1_lowres, (-1, model_conv1_lowres.shape[1] * model_conv1_lowres.shape[2] * model_conv1_lowres.shape[3]))

            model = tf.layers.max_pooling2d(
                inputs=model,
                pool_size=[3, 3],
                strides=3
            )

            model = tf.layers.conv2d(
                inputs=model,
                filters=8,
                kernel_size=[4, 4],
                padding="same",
                activation=tf.nn.relu,
                name="Layer2"
            )

            model_conv2_lowres = tf.layers.max_pooling2d(
                inputs=model,
                pool_size=[8, 8],
                strides=8
            )
            model_conv2_lowres_flat = tf.reshape(model_conv2_lowres, (-1, model_conv2_lowres.shape[1] * model_conv2_lowres.shape[2] * model_conv2_lowres.shape[3]))

            model = tf.layers.max_pooling2d(
                inputs=model,
                pool_size=[3, 3],
                strides=3
            )

            model = tf.layers.conv2d(
                inputs=model,
                filters=32,
                kernel_size=[2, 2],
                padding="same",
                activation=tf.nn.relu,
                name="Layer3"
            )

            model = tf.layers.max_pooling2d(
                inputs=model,
                pool_size=[3, 3],
                strides=3
            )

            model = tf.reshape(
                model,
                (-1, model.shape[1] * model.shape[2] * model.shape[3]))

            model = tf.concat([model, model_conv1_lowres_flat, model_conv2_lowres_flat], axis=1)

            model = tf.nn.dropout(
                model,
                keep_prob=self.tf_ph_droput_keep_prob
            )

            model = tf.layers.dense(
                inputs=model,
                units=2014,
                name="Layer4"
            )

            model = tf.nn.dropout(
                model,
                keep_prob=self.tf_ph_droput_keep_prob
            )

            model = tf.layers.dense(
                inputs=model,
                units=96*128,
                name="Layer5"
            )

            model = tf.reshape(
                model,
                (-1, 96, 128, 1))

            model = tf.layers.conv2d_transpose(
                model,
                filters=self.hyper_param_label_size,
                kernel_size=16,
                strides=((int)(self.hyper_param_height / model.shape[1].value), (int)(self.hyper_param_width / model.shape[2].value)),
                padding='SAME',
                name="Layer6")

            model = tf.reshape(
                model,
                (-1, model.shape[1] * model.shape[2], model.shape[3]),
                name="fcn_logits")

            self.tf_tensor_model = model

        if self.hyper_param_load_existing_model == True:
            self.tf_tensor_model = self.tf_graph.get_tensor_by_name("fcn_logits:0")


    def _initialize_cost(self):
        if self.hyper_param_load_existing_model == False:
            # Calculate distance from actual labels using cross entropy
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.tf_tensor_model, labels=self.tf_ph_labels_one_hot, name="cross_entropy")
            # Take mean for total loss
            self.tf_tensor_cost = tf.reduce_mean(cross_entropy, name="fcn_loss")

        if self.hyper_param_load_existing_model == True:
            self.tf_tensor_cost = self.tf_graph.get_tensor_by_name("fcn_loss:0")