import numpy as np
import tensorflow as tf
from tensorflow import train
from datetime import datetime
from datetime import timedelta

from ConvNets.ConvNetMNistSolver.CNNData import CNNData


class ConvNetMNistSolver:
    data_train = None
    data_validate = None

    tf_ph_x = None
    tf_ph_label_one_hot = None
    tf_ph_labels = None

    tf_tensor_model = None
    tf_tensor_cost = None
    tf_tensor_train = None
    tf_tensor_predictor = None
    tf_tensor_global_step = None
    tf_tensor_correctness = None

    tf_model_saver = None

    hyper_param_label_size = None
    hyper_param_picture_width = None
    hyper_param_picture_height = None
    hyper_param_train_batch_size = None
    hyper_param_learn_rate = None
    hyper_param_model_name = None
    hyper_param_load_existing_model = None
    hyper_param_save_model_interval_seconds = None


    def train_own_model(self):
        init = tf.global_variables_initializer()
        summary_merged = tf.summary.merge_all()

        with tf.Session() as session:
            session.run(init)
            if self.hyper_param_load_existing_model:
                self.tf_model_saver.restore(sess=session, save_path=tf.train.latest_checkpoint("./stored_models"))
                #self.tf_model_saver.restore(sess=session, save_path="./stored_models/" + self.hyper_param_model_name + ".ckpt")

            time_model_last_saved = datetime.now()

            writer_train = tf.summary.FileWriter("./logs/" + self.hyper_param_model_name + "/train", session.graph)
            writer_validation = tf.summary.FileWriter("./logs/" + self.hyper_param_model_name + "/validation")

            for i in range(session.run(self.tf_tensor_global_step), 1000000):
                #with tf.device("/gpu:0"):
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
                    self.tf_ph_label_one_hot: self.data_train.labels_one_hot[trainIds, :]
                })

                if i % 100 == 0:
                    print("Train step ", i, " finished")

                if i % 100 == 0:
                    print("Validation test")
                    self._calculcate_and_log_statistics(
                        "Train cost",
                        writer_train,
                        session,
                        self.data_train.data_x,
                        self.data_train.labels,
                        self.data_train.labels_one_hot
                    )

                    self._calculcate_and_log_statistics(
                        "Validation cost",
                        writer_validation,
                        session,
                        self.data_validate.data_x,
                        self.data_validate.labels,
                        self.data_validate.labels_one_hot
                    )

                    writer_train.flush()
                    writer_validation.flush()

                    print("Validation test finished")
                    print("")
                    print("")

                    if datetime.now() > (time_model_last_saved + timedelta(seconds=self.hyper_param_save_model_interval_seconds)):
                        time_model_last_saved = datetime.now()
                        self.tf_model_saver.save(sess=session, save_path="./stored_models/" + self.hyper_param_model_name + ".ckpt", global_step=tf.train.global_step(session, self.tf_tensor_global_step))
                        print("Model state saved")

    def _calculcate_and_log_statistics(self, cost_description, summary_writer, session, data_x, labels, labels_one_hot):
        cost_batches = []
        correctness_batches = []
        for j in range(data_x.shape[0])[0::self.hyper_param_train_batch_size]:
            training_cost_item, batch_correctness = session.run([self.tf_tensor_cost, self.tf_tensor_correctness], feed_dict={
                self.tf_ph_x: data_x[j:j + 10],
                self.tf_ph_label_one_hot: labels_one_hot[j:j + 10],
                self.tf_ph_labels: labels[j:j + 10]
            })
            cost_batches.append(training_cost_item)
            correctness_batches.append(batch_correctness)

        average_cost = np.mean(cost_batches)
        average_correctness = np.mean(correctness_batches)

        costs_summary = tf.Summary()
        costs_summary.value.add(tag="Cost", simple_value=average_cost)
        summary_writer.add_summary(costs_summary, tf.train.global_step(session, self.tf_tensor_global_step))
        print("{} = {}".format(cost_description, average_cost))

        correctness_summary = tf.Summary()
        correctness_summary.value.add(tag="Correctness", simple_value=average_correctness)
        summary_writer.add_summary(correctness_summary, tf.train.global_step(session, self.tf_tensor_global_step))
        print("Correctness = {}".format(average_correctness))

    def initialize_own_model(
            self,
            data_train: CNNData,
            data_validate: CNNData,
            hyper_param_label_size = 10,
            hyper_param_picture_height = 28,
            hyper_param_picture_width = 28,
            hyper_param_train_batch_size = 500,
            hyper_param_learn_rate = 0.03,
            hyper_param_model_name = "ConvNetModel",
            hyper_param_load_existing_model = False,
            hyper_param_save_model_interval_seconds = 300
    ):
        self.data_train = data_train
        self.data_validate = data_validate
        self.hyper_param_label_size = hyper_param_label_size
        self.hyper_param_picture_height = hyper_param_picture_height
        self.hyper_param_picture_width = hyper_param_picture_width
        self.hyper_param_train_batch_size = hyper_param_train_batch_size
        self.hyper_param_learn_rate = hyper_param_learn_rate
        self.hyper_param_model_name = hyper_param_model_name
        self.hyper_param_load_existing_model = hyper_param_load_existing_model
        self.hyper_param_save_model_interval_seconds = hyper_param_save_model_interval_seconds

        self.tf_tensor_global_step = tf.Variable(0, trainable=False, name='global_step')
        self.tf_ph_x = tf.placeholder(tf.float32, [None, self.hyper_param_picture_width, self.hyper_param_picture_height, 1], name='x')
        self.tf_ph_label_one_hot = tf.placeholder(tf.int32, [None, self.hyper_param_label_size], name='labels_one_hot')
        self.tf_ph_labels = tf.placeholder(tf.int32, [None], name='labels')
        self._initialize_model()
        self._initialize_tensor_cost()
        self._initialize_train_optimizer()
        self._initialize_predictor()

        self.tf_model_saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)


    def _initialize_model(self):
        model = tf.layers.max_pooling2d(
            inputs=self.tf_ph_x,
            pool_size=[10, 10],
            strides=10
        )

        model = tf.layers.conv2d(
            inputs=model,
            filters=16,
            kernel_size=[4, 4],
            padding="same",
            activation=tf.nn.relu
        )

        model = tf.layers.max_pooling2d(
            inputs=model,
            pool_size=[2, 2],
            strides=2
        )

        model = tf.layers.conv2d(
            inputs=self.tf_ph_x,
            filters=32,
            kernel_size=[4, 4],
            padding="same",
            activation=tf.nn.relu
        )

        model = tf.layers.max_pooling2d(
            inputs=model,
            pool_size=[2, 2],
            strides=2
        )

        model = tf.reshape(
            tensor=model,
            shape=[-1, 7 * 7 * 128]
        )
        model = tf.layers.dense(
            inputs=model,
            units=2048
        )

        model = tf.layers.dense(
            inputs=model,
            units=256
        )

        # Logits layer
        model = tf.layers.dense(
            inputs=model,
            units=self.hyper_param_label_size,
            name='model_logits_output'
        )

        self.tf_tensor_model = model

    def _initialize_tensor_cost(self):
        self.tf_tensor_cost = tf.losses.mean_squared_error(labels=self.tf_ph_label_one_hot, predictions=self.tf_tensor_model)

    def _initialize_train_optimizer(self):
        self.tf_tensor_train = tf.train.GradientDescentOptimizer(self.hyper_param_learn_rate).minimize(self.tf_tensor_cost)

    def _initialize_predictor(self):
        self.tf_tensor_predictor = tf.argmax(input=self.tf_tensor_model, axis=1, output_type=tf.int32)
        self.tf_tensor_correctness = tf.count_nonzero(tf.equal(self.tf_tensor_predictor, self.tf_ph_labels)) / tf.cast(tf.shape(self.tf_ph_labels)[0], dtype=tf.int64)





    def cnn_model_fn(self, features, labels, mode):
        tf.logging.set_verbosity(tf.logging.INFO)

        # Input layer
        input_layer = tf.reshape(tensor=features["x"], shape=[-1, self.hyper_param_picture_width, self.hyper_param_picture_height, 1])

        # Convolutional layer 1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",         # Keep the same width/height dimensions of the output tensor as for the input tensor
            activation=tf.nn.relu
        )
        # Shape of conv1: [batch_size, 28, 28, 32]

        # Pooling layer 1
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2
        )
        # Shape of pool1: [batch_size, 14, 14, 32]

        # Convolutional layer 2 and pooling layer 2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )

        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2
        )

        # Dense layer
        pool2_flat = tf.reshape(
            tensor=pool2,
            shape=[-1, 7 * 7 * 64]
        )
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        # Logits layer
        logits = tf.layers.dense(
            inputs=dropout,
            units=10
        )

        predictions = {
            # Generate predictions for PREDICT and EVAL mode
            "classes" : tf.argmax(input=logits, axis=1),
            # Add softmax_tensor to the graph. It is used for PREDICT and by the logging_hook
            "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate loos for both TRAIN and EVAL modes
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op for TRAIN mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        #Add evaluation metrics for EVAL mode
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    def run(self):
        # Load training and eval data
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
            model_fn=self.cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

        # Set up logging for predictions
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)