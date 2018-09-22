import numpy as np
import tensorflow as tf
from tensorflow import train

from ConvNets.ConvNetMNistSolver.CNNData import CNNData


class ConvNetMNistSolver:
    data_train = None
    data_validate = None

    tf_ph_x = None
    tf_ph_label = None

    tf_tensor_model = None
    tf_tensor_cost = None
    tf_tensor_train = None

    hyper_param_label_size = None
    hyper_param_picture_width = None
    hyper_param_picture_height = None


    def train_own_model(self):
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            for i in range(100):
                #####Train#####
                result = session.run(self.tf_tensor_train, feed_dict={
                    self.tf_ph_x: self.data_train.data_x,
                    self.tf_ph_label: self.data_train.labels
                })

                training_cost = session.run(self.tf_tensor_cost, feed_dict={
                    self.tf_ph_x: self.data_train.data_x,
                    self.tf_ph_label: self.data_train.labels
                })
                print("Train Cost = {}".format(training_cost))


    def initialize_own_model(self,
                  data_train: CNNData):
        #self.data_train = data_train

        self.hyper_param_label_size = 10
        self.hyper_param_picture_height = 28
        self.hyper_param_picture_width = 28

        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        train_data = mnist.train.images  # Returns np.array
        train_data = train_data.reshape((-1, self.hyper_param_picture_width, self.hyper_param_picture_height, 1))

        train_labels_one_hot = np.eye(self.hyper_param_label_size)[train_labels]

        self.data_train = CNNData(train_data, train_labels_one_hot)

        eval_data = mnist.test.images  # Returns np.array
        eval_data = eval_data.reshape((-1, self.hyper_param_picture_width, self.hyper_param_picture_height, 1))
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        # Black and white picture
        self.tf_ph_x = tf.placeholder(tf.float32, [None, self.hyper_param_picture_width, self.hyper_param_picture_height, 1])
        self.tf_ph_label = tf.placeholder(tf.int32, [None, self.hyper_param_label_size])
        self._initialize_model()
        self._initialize_tensor_cost()
        self._initialize_train_optimizer()


    def _initialize_model(self):
        model = tf.layers.conv2d(
            inputs=self.tf_ph_x,
            filters=4,
            kernel_size=[3, 3],
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
            shape=[-1, 14 * 14 * 4]
        )
        model = tf.layers.dense(
            inputs=model,
            units=1024
        )

        # Logits layer
        model = tf.layers.dense(
            inputs=model,
            units=self.hyper_param_label_size
        )

        self.tf_tensor_model = model

    def _initialize_tensor_cost(self):
        self.tf_tensor_cost = tf.losses.mean_squared_error(labels=self.tf_ph_label, predictions=self.tf_tensor_model)

    def _initialize_train_optimizer(self):
        hyper_param_learn_rate = 0.01
        self.tf_tensor_train = tf.train.GradientDescentOptimizer(hyper_param_learn_rate).minimize(self.tf_tensor_cost)





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