import os

from ConvNets.ConvNetMNistSolver.ConvNetMNistSolver import ConvNetMNistSolver
from LinearRegression.LinearRegression import LinearRegression
from LogisticRegression.LogisticRegression import LogisticRegression
from LogisticRegression.LogisticRegressionData import LogisticRegressionData
from TensorFlowExamples.BasicOperations.BasicMatrixOperations import BasicMatrixOperations
from TensorFlowExamples.BasicOperations.ConstantExample import ConstantExample
from TensorFlowExamples.BasicOperations.MatrixReduction import MatrixReduction
from TensorFlowExamples.BasicOperations.NumpyToTensorConversion import NumpyToTensorConversion
from TensorFlowExamples.BasicOperations.PlaceholderExample import PlaceholderExample
from TensorFlowExamples.BasicOperations.SegmentationExample import SegmentationExample
from TensorFlowExamples.BasicOperations.SequenceUtilitiesExamples import SequenceUtilitiesExamples
from TensorFlowExamples.BasicOperations.TensorboardExample import TensorboardExample
from TensorFlowExamples.KNearestNeighbors.KNearestNeighbors import KNearestNeighbors
from datasets.IOHelper import IOHelper

#constantExample = ConstantExample()
#constantExample.run()


#print("")
#placeholderExample = PlaceholderExample()
#placeholderExample.run()

#print("")
#tensorboardExample = TensorboardExample()
#tensorboardExample.run()

#print("")
#numpyToTensorConversion = NumpyToTensorConversion()
#numpyToTensorConversion.run()

#print("")
#basicMatrixOperations = BasicMatrixOperations()
#basicMatrixOperations.run()

#print("")
#matrixReduction = MatrixReduction()
#matrixReduction.run()

#print("")
#segmentationExample = SegmentationExample()
#segmentationExample.run()

#print("")
#sequenceUtilitiesExamples = SequenceUtilitiesExamples()
#sequenceUtilitiesExamples.run()

#print("")
#io_helper = IOHelper()
#knearestNeighbors = KNearestNeighbors()
#knearestNeighbors.run_learning(os.path.abspath(os.path.join(os.path.dirname(__file__),'datasets','creditcard.csv')), k=3)

#print("")
#linear_regression = LinearRegression()
#x_train, y_train, x_validate, y_validate = io_helper.loadCsvData(os.path.abspath(os.path.join(os.path.dirname(__file__),'datasets','creditcard.csv')))
#linear_regression.initialize(x_train, y_train)
#linear_regression.minimize()

#print("")
#io_helper = IOHelper()
#logistic_regression = LogisticRegression()
#x_train, y_train, x_validate, y_validate = io_helper.loadCsvData(
#    os.path.abspath(os.path.join(os.path.dirname(__file__),'datasets','creditcard.csv')),
#    trainPart=0.8,
#    validationPart=0.2)
#logisticRegressionDataTrain = LogisticRegressionData(x_train, y_train, feature_scale=True)
#logisticRegressionDataValidate = LogisticRegressionData(x_validate, y_validate)
#logisticRegressionDataValidate.OverrideMinMax(logisticRegressionDataTrain.x_min, logisticRegressionDataTrain.x_max)

#logistic_regression.initialize(
#    logisticRegressionDataTrain,
#    logisticRegressionDataValidate,
#    hyper_param_polynomialDegree=10,
#    hyper_param_iterations=100000,
#    hyper_param_learn_rate=0.1,
#    hyper_param_lambda=0.1,
#    feature_scale=True,
#    label_0_cost_modification=1.0,
#    label_1_cost_modification=750)
#logistic_regression.minimize()




print("")
# Load training and eval data
mnistSolver = ConvNetMNistSolver()
mnistSolver.initialize_own_model(None)
mnistSolver.train_own_model()
#mnistSolver.test()