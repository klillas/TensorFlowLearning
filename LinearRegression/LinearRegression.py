import tensorflow as tf
import numpy as np
from six.moves import xrange
from sklearn.preprocessing import StandardScaler

# Change to test commit and push to GitHub
class LinearRegression:
    x_input = 0
    y_input = 0

    iterations = 0
    learn_rate = 0
    m = 0
    params = 0
    x = 0
    W = 0
    b = 0
    y = 0
    model = 0
    cost = 0
    train = 0

    def initialize(self, x_input, y_input, iterations = 100000, learn_rate = 0.01, feature_scale = True):
        '''
        Initializes the linear regression algorithm
        :param x_input: The training input
        :param y_input: The training labels
        :param iterations: The number of minimize iterations
        :param learn_rate: The learning rate of the algorithm
        :return: Nothing
        '''

        if feature_scale:
            scaler = StandardScaler()
            scaler.fit(x_input)
            x_input = scaler.transform(x_input)

        self.x_input = x_input
        self.y_input = y_input
        self.iterations = iterations
        self.learn_rate = learn_rate
        m = x_input.shape[0]
        params = x_input.shape[1]

        self.x = tf.placeholder(tf.float32, [m, params], name="x")
        self.W = tf.Variable(tf.zeros([params, 1]), name="W")
        self.b = tf.Variable(tf.zeros([1]), name="b")
        self.y = tf.placeholder(tf.float32, [m, 1])
        self.model = tf.add(tf.matmul(self.x, self.W), self.b)

        self.cost = tf.reduce_mean(tf.square(self.y - self.model))
        self.train = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.cost)

    def minimize(self):
        '''
        Performs the Linear Regression according to the input provided in the initialize method
        :return: Nothing
        '''
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            for _ in xrange(self.iterations):
                session.run(self.train, feed_dict={
                    self.x: self.x_input,
                    self.y: self.y_input
                })

                print("Cost = {}".format(session.run(self.cost, feed_dict={
                    self.x: self.x_input,
                    self.y: self.y_input
                })))

                guess_labels = session.run(tf.round(self.model), feed_dict={
                    self.x: self.x_input,
                    self.W: session.run(self.W),
                    self.b: session.run(self.b)
                })

                print("Incorrect label guesses: ", np.sum(np.abs(np.subtract(self.y_input, guess_labels))))
