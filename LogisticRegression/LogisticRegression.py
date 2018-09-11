import uuid
from datetime import datetime

import tensorflow as tf
import numpy as np
from six.moves import xrange

from LogisticRegression.LogisticRegressionData import LogisticRegressionData


class LogisticRegression:
    data_train = None
    data_validation = None

    hyper_param_polynomialDegree = None
    hyper_param_iterations = None
    hyper_param_learn_rate = None
    data_train_m = None
    data_params = None

    tf_hold_x = None
    tf_hold_y = None

    tf_var_W = [None]
    tf_var_B = None

    tf_tensor_model = None
    tf_tensor_cost = None
    tf_tensor_validation_cost = None
    tf_tensor_train = None
    tf_tensor_classifier = None
    tf_tensor_correctness_percentage = None

    tf_scalar_summary_cost = None
    tf_scalar_summary_correctness_percentage = None

    def initialize(
            self,
            data_train: LogisticRegressionData,
            data_validation: LogisticRegressionData,
            hyper_param_polynomialDegree = 3,
            hyper_param_iterations = 10000,
            hyper_param_learn_rate = 0.01,
            feature_scale = True,
            label_0_cost_modification = 1.0,
            label_1_cost_modification = 1.0):

        self.data_train = data_train
        self.data_validation = data_validation

        print("First x input data_params: ")
        print(self.data_train.data_x[0])
        # print("Max value of x: ", np.matrix.max(self.data_train.data_x))
        # print("Min value of x: ", np.matrix.min(self.data_train.data_x))

        self.hyper_param_polynomialDegree = hyper_param_polynomialDegree
        self.hyper_param_iterations = hyper_param_iterations
        self.hyper_param_learn_rate = hyper_param_learn_rate
        self.data_train_m = self.data_train.data_x.shape[0]
        self.data_params = self.data_train.data_x.shape[1]
        self.tf_var_W = [None] * hyper_param_polynomialDegree

        self.tf_hold_x = tf.placeholder(tf.float32, [None, self.data_params], name="x")
        self.tf_hold_y = tf.placeholder(tf.float32, [None, 1])

        self.tf_var_W[0] = tf.Variable(((np.random.rand(self.data_params, 1) - 0.5) * 0.0001).astype(np.float32), name="W" + str(0))
        self.tf_tensor_model = tf.matmul(self.tf_hold_x, self.tf_var_W[0])
        for i in range(1, self.hyper_param_polynomialDegree):
            self.tf_var_W[i] = tf.Variable(((np.random.rand(self.data_params, 1) - 0.5) * 0.0001).astype(np.float32), name="W" + str(i))
            self.tf_tensor_model = tf.add(tf.matmul(tf.pow(self.tf_hold_x, i + 1), self.tf_var_W[i]), self.tf_tensor_model)

        self.tf_var_B = tf.Variable(((np.random.rand(1) - 0.5) * 0.0001).astype(np.float32), name="b")
        tf.summary.histogram("b", self.tf_var_B)
        self.tf_tensor_model = tf.add(self.tf_tensor_model, self.tf_var_B)

        self.tf_tensor_cost = tf.reduce_sum(
            tf.multiply(tf.multiply(-self.tf_hold_y, tf.log(tf.sigmoid(self.tf_tensor_model))), label_1_cost_modification)
            -tf.multiply(tf.multiply((1-self.tf_hold_y), tf.log(1 - tf.sigmoid(self.tf_tensor_model))), label_0_cost_modification)) / tf.to_float(tf.shape(self.tf_hold_x)[0])
        self.tf_scalar_summary_cost = tf.summary.scalar("Cost", self.tf_tensor_cost)

        #self.tf_tensor_validation_cost = tf.reduce_sum(
        #    tf.multiply(tf.multiply(-self.tf_hold_y, tf.log(tf.sigmoid(self.tf_tensor_model))), label_1_cost_modification)
        #    -tf.multiply(tf.multiply((1-self.tf_hold_y), tf.log(1 - tf.sigmoid(self.tf_tensor_model))), label_0_cost_modification)) / tf.to_float(tf.shape(self.tf_hold_x)[0])
        #tf.summary.scalar("Validation Cost", self.tf_tensor_validation_cost)

        self.tf_tensor_classifier = tf.round(tf.sigmoid(self.tf_tensor_model))
        self.tf_tensor_correctness_percentage = tf.reduce_sum(tf.cast(tf.equal(self.tf_tensor_classifier, self.tf_hold_y), tf.float32)) / tf.to_float(tf.shape(self.tf_hold_x)[0])
        self.tf_scalar_summary_correctness_percentage = tf.summary.scalar("Correctness percentage", self.tf_tensor_correctness_percentage)

        self.tf_tensor_train = tf.train.GradientDescentOptimizer(hyper_param_learn_rate).minimize(self.tf_tensor_cost)

        print("Number of label 0 in set: ", self.data_train_m - np.sum(self.data_train.data_y))
        print("Number of label 1 in set: ", np.sum(self.data_train.data_y))


    def minimize(self):
        '''
        Performs the Linear Regression according to the input provided in the initialize method
        :return: Nothing
        '''
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            summary_merged = tf.summary.merge_all()
            # writer = tf.summary.FileWriter("logs", session.graph)
            sessionFolder = str(uuid.uuid4())
            writer_train = tf.summary.FileWriter("./logs/" + sessionFolder + "/train", session.graph)
            writer_validation = tf.summary.FileWriter("./logs/" + sessionFolder + "/validation")

            for i in xrange(self.hyper_param_iterations):
                #####Train#####
                summary, result = session.run([summary_merged, self.tf_tensor_train], feed_dict={
                    self.tf_hold_x: self.data_train.data_x,
                    self.tf_hold_y: self.data_train.data_y
                })

                training_cost = session.run(self.tf_tensor_cost, feed_dict={
                    self.tf_hold_x: self.data_train.data_x,
                    self.tf_hold_y: self.data_train.data_y
                })
                print("Train Cost = {}".format(training_cost))

                print("Train correctness percentage: ", session.run(self.tf_tensor_correctness_percentage, feed_dict={
                    self.tf_hold_x: self.data_train.data_x,
                    self.tf_hold_y: self.data_train.data_y
                }))

                writer_train.add_summary(summary, i)

                #####Validation#####
                summary, validation_cost = session.run([self.tf_scalar_summary_cost, self.tf_tensor_cost], feed_dict={
                    self.tf_hold_x: self.data_validation.data_x,
                    self.tf_hold_y: self.data_validation.data_y
                })
                writer_validation.add_summary(summary, i)
                print("Validation Cost = {}".format(validation_cost))

                summary, validation_correctness_percentage = session.run([self.tf_scalar_summary_correctness_percentage, self.tf_tensor_correctness_percentage], feed_dict={
                    self.tf_hold_x: self.data_train.data_x,
                    self.tf_hold_y: self.data_train.data_y
                })
                print("Validation correctness percentage: ", validation_correctness_percentage);
                writer_validation.add_summary(summary, i)


#               print("Incorrect label guesses: ", np.sum(np.abs(np.subtract(self.data_train.data_y, guess_labels))))

                #predicted_labels = session.run(self.tf_tensor_classifier, feed_dict={
                #    self.tf_hold_x: self.data_train.data_x,
                #    self.tf_hold_y: self.data_train.data_y
                #})
                #correct_guesses_count = session.run(tf.reduce_sum(tf.cast(tf.equal(predicted_labels, self.data_train.data_y), tf.float32)))

                #print("Correct guesses: ", correct_guesses_count, ". Incorrect guesses: ", self.data_train_m - correct_guesses_count)
                #print("Incorrect label 0 guesses: ", np.sum(np.multiply(1 - self.data_train.data_y, predicted_labels)))
                #print("Incorrect label 1 guesses: ", np.sum(np.multiply(self.data_train.data_y, 1 - predicted_labels)))


