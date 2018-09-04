import os

from LinearRegression.LinearRegression import LinearRegression
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

io_helper = IOHelper()

#print("")
#knearestNeighbors = KNearestNeighbors()
#knearestNeighbors.run_learning(os.path.abspath(os.path.join(os.path.dirname(__file__),'datasets','creditcard.csv')), k=3)

print("")
linear_regression = LinearRegression()
x_input, y_input = io_helper.loadCsvData(os.path.abspath(os.path.join(os.path.dirname(__file__),'datasets','creditcard.csv')))
linear_regression.initialize(x_input, y_input)
linear_regression.minimize()