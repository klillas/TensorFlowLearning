import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
from ConvNets.SemanticSegmentation import SemanticSegmentationData, SemanticSegmentationTrainingDataLoader
from datetime import datetime
from datetime import timedelta
from scipy import misc
from skimage import data, color, io, img_as_float


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

    tf_summary_image_predictions = None
    tf_summary_real_image_predictions = None

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
    hyper_param_validation_batch_size = None
    hyper_param_validation_every_n_steps = None

    tf_model_saver = None

    validation_data = None

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
            validation_every_n_steps):
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

        if load_existing_model == True:
            # Old examples could have been used in training. Delete everything before loading the validation data batch
            self.data_generator.delete_all_existing_training_data()

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
            self.tf_ph_learning_rate = self.tf_graph.get_tensor_by_name("learning_rate:0")

        self.tf_summary_image_predictions = tf.summary.image("example prediction", self.tf_variable_image, 50)
        self.tf_summary_real_image_predictions = tf.summary.image("real world prediction", self.tf_variable_image, self.data_generator.get_real_world_training_examples().shape[0])

        #self._initialize_model_minimal()
        self._initialize_model_U_net()
        self._initialize_cost()
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

        train_cost_last_print = 0
        for i in range(self.session.run(self.tf_variable_global_step), 1000000):
            data_train = self.data_generator.load_next_batch()
            if i % 25 == 0:
                print("Training step {}".format(i))
            self.tf_variable_global_step.load(i, session=self.session)
            #####Train#####
            #self.exportImageWithLabels("c:/temp/pic.jpg", 0,  np.reshape(self.data_train.labels[0], (self.hyper_param_height, self.hyper_param_width)))

            _ = self.session.run(self.tf_tensor_train, feed_dict={
                self.tf_ph_x: data_train.data_x,
                self.tf_ph_labels_one_hot: data_train.labels_one_hot,
                self.tf_ph_droput_keep_prob: self.hyper_param_dropout_keep_prob,
                self.tf_ph_learning_rate: self.hyper_param_learning_rate
            })

            batch_training_cost = self.session.run(self.tf_tensor_cost, feed_dict={
                self.tf_ph_x: data_train.data_x,
                self.tf_ph_labels_one_hot: data_train.labels_one_hot,
                self.tf_ph_droput_keep_prob: 1.0,
                self.tf_ph_learning_rate: self.hyper_param_learning_rate
            })

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
                print("Train batch cost delta: {}".format(batch_training_cost - train_cost_last_print))
                train_cost_last_print = batch_training_cost
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

            if datetime.now() > (time_model_last_saved + timedelta(seconds=self.hyper_param_save_model_interval_seconds)):
                time_model_last_saved = datetime.now()
                self.tf_model_saver.save(
                    sess=self.session,
                    save_path="./stored_models/" + self.hyper_param_model_name,
                    global_step=tf.train.global_step(self.session, self.tf_variable_global_step))
                print("Model state saved")


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
            overlay_image = np.zeros((50, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels))
            for j in range(50)[0::self.hyper_param_train_batch_size]:
                semantic_segmentation_data = self.data_generator.load_next_batch(delete_batch_source=True)
                batch_predictor = self.session.run(self.tf_tensor_predictor, feed_dict={
                    self.tf_ph_x: semantic_segmentation_data.data_x,
                    self.tf_ph_droput_keep_prob: 1.0
                })
                prediction_batches[j:j + self.hyper_param_train_batch_size] = batch_predictor
                overlay_image[j:j + self.hyper_param_train_batch_size] = semantic_segmentation_data.data_x

            for i in range(50):
                overlay_image[i] = self._overlay_image_with_labels(overlay_image[i], np.reshape(prediction_batches[i], (self.hyper_param_height, self.hyper_param_width)))
            self.tf_variable_image.load(overlay_image, self.session)

            image_summary = self.session.run(self.tf_summary_image_predictions)
            summary_writer.add_summary(image_summary, tf.train.global_step(self.session, self.tf_variable_global_step))



            # TODO: Make the amount of predictions user definable (All the static 50 assignments here and in the initializer
            # TODO: This only works if the batch size is equal to 50 or evenly divisible, so have to fix that
            real_world_examples = self.data_generator.get_real_world_training_examples()
            overlay_image = np.zeros((50, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels))
            for j in range(real_world_examples.shape[0]):
                image_prediction = self.session.run(self.tf_tensor_predictor, feed_dict={
                    self.tf_ph_x: np.reshape(real_world_examples[j], (1, self.hyper_param_height, self.hyper_param_width, self.hyper_param_image_channels)),
                    self.tf_ph_labels: semantic_segmentation_data.labels,
                    self.tf_ph_droput_keep_prob: 1.0
                })
                overlay_image[j] = self._overlay_image_with_labels(real_world_examples[j], np.reshape(image_prediction, (self.hyper_param_height, self.hyper_param_width)))

            self.tf_variable_image.load(overlay_image, self.session)

            image_summary = self.session.run(self.tf_summary_real_image_predictions)
            summary_writer.add_summary(image_summary, tf.train.global_step(self.session, self.tf_variable_global_step))

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

    def _model_add_convolution(self, model, filter_multiplier=0, filter_size=0, kernel_size=[3, 3], strides=1):
        model_filter_size = model.shape[3].value
        new_filter_size = filter_size
        if filter_multiplier != 0:
            new_filter_size = int(model_filter_size * filter_multiplier)
        model = tf.layers.conv2d(
            inputs=model,
            filters=new_filter_size,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            activation=tf.nn.relu
        )
        return model

    def _model_add_max_pooling(self, model, pool_size=[2, 2], strides=2):
        model = tf.layers.max_pooling2d(
            inputs=model,
            pool_size=pool_size,
            strides=strides
        )
        return model

    def _model_add_deconvolution(self, model, size_multiplier):
        model = tf.layers.conv2d_transpose(
            model,
            filters=model.shape[3],
            strides=size_multiplier,
            kernel_size=(size_multiplier, size_multiplier),
            padding='same')
        return model

    def _initialize_model_U_net(self):
        if self.hyper_param_load_existing_model == False:
            model = tf.nn.dropout(self.tf_ph_x, keep_prob=self.tf_ph_droput_keep_prob)

            model = self._model_add_convolution(model=model, filter_size=16)
            model = self._model_add_convolution(model=model, filter_multiplier=1)
            output_layer1 = model
            model = self._model_add_max_pooling(model=model, pool_size=[4, 4], strides=4)
            #model = tf.nn.dropout(model, keep_prob=self.tf_ph_droput_keep_prob)

            model = self._model_add_convolution(model=model, filter_multiplier=2)
            model = self._model_add_convolution(model=model, filter_multiplier=1)
            output_layer2 = model
            model = self._model_add_max_pooling(model=model)
            #model = tf.nn.dropout(model, keep_prob=self.tf_ph_droput_keep_prob)

            model = self._model_add_convolution(model=model, filter_multiplier=2)
            model = self._model_add_convolution(model=model, filter_multiplier=1)


            ###########Deconvolution############
            model = self._model_add_deconvolution(model=model, size_multiplier=2)
            model = self._model_add_convolution(model=model, filter_multiplier=0.5)
            model = tf.concat([model, output_layer2], axis=3)
            model = self._model_add_convolution(model=model, filter_multiplier=1)
            model = self._model_add_convolution(model=model, filter_multiplier=0.5)

            model = self._model_add_deconvolution(model=model, size_multiplier=4)
            model = self._model_add_convolution(model=model, filter_multiplier=0.5)
            model = tf.concat([model, output_layer1], axis=3)
            model = self._model_add_convolution(model=model, filter_multiplier=1)
            model = self._model_add_convolution(model=model, filter_multiplier=0.5)

            model = self._model_add_convolution(model=model, filter_size=self.hyper_param_label_size)

            model = tf.reshape(model, (-1, model.shape[1] * model.shape[2], self.hyper_param_label_size), name="fcn_logits")

            self.tf_tensor_model = model

        if self.hyper_param_load_existing_model == True:
            self.tf_tensor_model = self.tf_graph.get_tensor_by_name("fcn_logits:0")


    def _initialize_model_minimal(self):
        if self.hyper_param_load_existing_model == False:
            model = tf.layers.conv2d(
                inputs=self.tf_ph_x,
                filters=64,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu
            )

            model = tf.layers.conv2d(
                inputs=model,
                filters=64,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu
            )

            model = tf.layers.max_pooling2d(
                inputs=model,
                pool_size=[2, 2],
                strides=2
            )

            model = tf.layers.conv2d(
                inputs=model,
                filters=128,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu
            )

            model = tf.layers.conv2d(
                inputs=model,
                filters=128,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu
            )

            model = tf.layers.max_pooling2d(
                inputs=model,
                pool_size=[2, 2],
                strides=2
            )

            model = tf.layers.conv2d(
                inputs=model,
                filters=256,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu
            )

            model = tf.layers.conv2d(
                inputs=model,
                filters=256,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu
            )

            model = tf.layers.max_pooling2d(
                inputs=model,
                pool_size=[3, 3],
                strides=3
            )

            model = tf.reshape(
                model,
                (-1, model.shape[1] * model.shape[2] * model.shape[3]))

            #model = tf.concat([model, model_conv1_lowres_flat, model_conv2_lowres_flat], axis=1)

            #model = tf.nn.dropout(
            #    model,
            #    keep_prob=self.tf_ph_droput_keep_prob
            #)

            model = tf.layers.dense(
                inputs=model,
                units=1024
            )

            #model = tf.nn.dropout(
            #    model,
            #    keep_prob=self.tf_ph_droput_keep_prob
            #)

            model = tf.layers.dense(
                inputs=model,
                units=48*64,
                name="Dense2"
            )

            model = tf.reshape(
                model,
                (-1, 48, 64, 1))

            model = tf.layers.conv2d_transpose(
                model,
                filters=self.hyper_param_label_size,
                kernel_size=64,
                strides=((int)(self.hyper_param_height / model.shape[1].value), (int)(self.hyper_param_width / model.shape[2].value)),
                padding='SAME')

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



    def predict_and_create_image(self, path, image_data):
        prediction = self.session.run(self.tf_tensor_predictor, feed_dict={
            self.tf_ph_x: np.reshape(image_data, (-1, image_data.shape[0], image_data.shape[1], image_data.shape[2])),
            self.tf_ph_droput_keep_prob: 1.0
        })

        self._export_image_with_labels(path, image_data, np.reshape(prediction, (self.hyper_param_height, self.hyper_param_width)))

    def _export_image_with_labels(self, path, image_data, predicted_labels):
        misc.imsave(path, self._overlay_image_with_labels(image_data, predicted_labels))

    def _overlay_image_with_labels(self, image_data, predicted_labels):
        color_mask = np.zeros((predicted_labels.shape[0], predicted_labels.shape[1], 3))
        color_mask[np.where(predicted_labels == 1)] = [232, 88, 35]
        color_mask[np.where(predicted_labels == 2)] = [41, 48, 90]
        color_mask[np.where(predicted_labels == 3)] = [246, 164, 3]
        color_mask[np.where(predicted_labels == 4)] = [166, 169, 130]
        color_mask[np.where(predicted_labels == 5)] = [96, 157, 186]


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

        # Display the output
        #f, (ax0) = plt.subplots(1, 1, subplot_kw={'xticks': [], 'yticks': []})
        #ax0.imshow(self.data_x[id], cmap=plt.cm.gray)
        #ax1.imshow(color_mask)
        #ax0.imshow(img_masked)
        #plt.show()

        return img_masked