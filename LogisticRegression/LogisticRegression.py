import tensorflow as tf
import numpy as np
from six.moves import xrange

class LogisticRegression:
    x_input = 0
    y_input = 0

    polynomialDegree = 0
    iterations = 0
    learn_rate = 0
    m = 0
    params = 0
    x = 0
    W = [None]
    b = 0
    y = 0
    model = 0
    cost = 0
    train = 0
    prediction = 0
    classifier = 0
    correctness_percentage = 0


    def initialize(self, x_input, y_input, polynomialDegree = 3, iterations = 10000, learn_rate = 0.01, feature_scale = True, label_0_cost_modification = 1.0, label_1_cost_modification = 1.0):
        '''
        Initializes the linear regression algorithm
        :param x_input: The training input
        :param y_input: The training labels
        :param iterations: The number of minimize iterations
        :param learn_rate: The learning rate of the algorithm
        :return: Nothing
        '''
        x_max = np.amax(x_input, axis=0)
        x_min = np.amin(x_input, axis=0)

        if feature_scale:
            x_input = (x_input-x_min)/(x_max-x_min)

        print("First x input params: ")
        print(x_input[0])
        # print("Max value of x: ", np.matrix.max(x_input))
        # print("Min value of x: ", np.matrix.min(x_input))

        self.x_input = x_input
        self.y_input = np.float32(y_input)
        self.polynomialDegree = polynomialDegree
        self.iterations = iterations
        self.learn_rate = learn_rate
        self.m = x_input.shape[0]
        self.params = x_input.shape[1]
        self.W = [None] * polynomialDegree

        self.x = tf.placeholder(tf.float32, [self.m, self.params], name="x")
        self.y = tf.placeholder(tf.float32, [self.m, 1])

        self.W[0] = tf.Variable(((np.random.rand(self.params, 1)-0.5)*0.0001).astype(np.float32), name="W"+str(0))
        self.model = tf.matmul(self.x, self.W[0])
        for i in range(1, self.polynomialDegree):
            self.W[i] = tf.Variable(((np.random.rand(self.params, 1)-0.5)*0.0001).astype(np.float32), name="W"+str(i))
            self.model = tf.add(tf.matmul(tf.pow(self.x, i+1), self.W[i]), self.model)

        self.b = tf.Variable(((np.random.rand(1)-0.5)*0.0001).astype(np.float32), name="b")
        tf.summary.histogram("b", self.b)
        self.model = tf.add(self.model, self.b)

        self.cost = tf.reduce_sum(
            tf.multiply(tf.multiply(-self.y_input, tf.log(tf.sigmoid(self.model))), label_1_cost_modification)
            -tf.multiply(tf.multiply((1-self.y_input), tf.log(1-tf.sigmoid(self.model))), label_0_cost_modification)) / self.m
        tf.summary.scalar("Training cost", self.cost)

        self.classifier = tf.round(tf.sigmoid(self.model))
        self.correctness_percentage = tf.reduce_sum(tf.cast(tf.equal(self.classifier, self.y), tf.float32)) / self.m
        tf.summary.scalar("Correctness percentage", self.correctness_percentage)

        self.train = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.cost)

        print("Number of label 0 in set: ", self.m - np.sum(self.y_input))
        print("Number of label 1 in set: ", np.sum(self.y_input))


    def minimize(self):
        '''
        Performs the Linear Regression according to the input provided in the initialize method
        :return: Nothing
        '''
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("logs", session.graph)

            for i in xrange(self.iterations):
                summary, result = session.run([merged, self.train], feed_dict={
                    self.x: self.x_input,
                    self.y: self.y_input
                })

                print("Cost = {}".format(session.run(self.cost, feed_dict={
                    self.x: self.x_input,
                    self.y: self.y_input
                })))

                print("Correctness percentage: ", session.run(self.correctness_percentage, feed_dict={
                    self.x: self.x_input,
                    self.y: self.y_input
                }))

                writer.add_summary(summary, i)

#               print("Incorrect label guesses: ", np.sum(np.abs(np.subtract(self.y_input, guess_labels))))

                predicted_labels = session.run(self.classifier, feed_dict={
                    self.x: self.x_input,
                    self.y: self.y_input
                })
                correct_guesses_count = session.run(tf.reduce_sum(tf.cast(tf.equal(predicted_labels, self.y_input), tf.float32)))

                print("Correct guesses: ", correct_guesses_count, ". Incorrect guesses: ", self.m - correct_guesses_count)
                print("Incorrect label 0 guesses: ", np.sum(np.multiply(1 - self.y_input, predicted_labels)))
                print("Incorrect label 1 guesses: ", np.sum(np.multiply(self.y_input, 1 - predicted_labels)))


