import numpy as np
import cProfile

from ConvNets.ConvNetMNistSolver.CNNData import CNNData
from ConvNets.ConvNetMNistSolver.ConvNetMNistSolver import ConvNetMNistSolver
from ConvNets.SemanticSegmentation.SemanticSegmentation import SemanticSegmentation
from ConvNets.SemanticSegmentation.SemanticSegmentationTrainingDataLoader import SemanticSegmentationTrainingDataLoader

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




#print("")
# Load training and eval data
#hyper_param_label_size = 10
#hyper_param_picture_height = 28
#hyper_param_picture_width = 28

#mnist = tf.contrib.learn.datasets.load_dataset("mnist")

#train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#train_data = mnist.train.images  # Returns np.array
#train_data = train_data.reshape((-1, hyper_param_picture_width, hyper_param_picture_height, 1))
#train_labels_one_hot = np.eye(hyper_param_label_size)[train_labels]
#data_train = CNNData(train_data, train_labels, train_labels_one_hot)

#validation_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#validation_data = mnist.test.images  # Returns np.array
#validation_data = validation_data.reshape((-1, hyper_param_picture_width, hyper_param_picture_height, 1))
#validation_labels_one_hot = np.eye(hyper_param_label_size)[validation_labels]
#data_validate = CNNData(validation_data[0:7000], validation_labels[0:7000], validation_labels_one_hot[0:7000])

#mnistSolver = ConvNetMNistSolver()
#mnistSolver.initialize_own_model(
#    hyper_param_model_name="MyModel30",
#    data_train=data_train,
#    data_validate=data_validate,
#    hyper_param_label_size=hyper_param_label_size,
#    hyper_param_picture_height=hyper_param_picture_height,
#    hyper_param_picture_width=hyper_param_picture_width,
#    hyper_param_load_existing_model=True,
#    hyper_param_save_model_interval_seconds=300
#)
#mnistSolver.train_own_model()






print("")
# Load training and eval data
print("Loading training data")
training_data_generator = SemanticSegmentationTrainingDataLoader()
training_data_generator.initialize(
    batch_size=5,
    probability_delete_example=0.0,
    minimum_available_training_set_size=1)

semantic_segmentation = SemanticSegmentation()
semantic_segmentation.initialize(
    training_data_generator,
    training_data_generator.image_height,
    training_data_generator.image_width,
    training_data_generator.image_channels,
    #0.0001, ==> Slowly decreasing
    #0.03, ==> Slowly increasing
    #0.001, ==> Decreasing
    0.00003,
    batch_size=training_data_generator.batch_size,
    hyper_param_model_name="BallFinder_Boundary_01",
    load_existing_model=False,
    save_model_interval_seconds=300,
    dropout_keep_prob=0.98,
    validation_batch_size=50,
    validation_every_n_steps=1000,
    adaptive_learning_rate_active=False,
    adaptive_learning_rate=0.05,
    max_epochs=1000000000)

#for i in range(5000, 5100):
    #picture_data = training_data_generator.load_picture_data("c:/temp/training/" + str(i) +"_CameraLeftEye.jpg")
    #semantic_segmentation.predict_and_create_image("c:/temp/pic" + str(i) + ".jpg", picture_data)

#picture_data = training_data_generator.load_picture_data("c:/temp/RealWorldBalls/20181026_193618 - 256x192.jpg")
#semantic_segmentation.predict_and_create_image("c:/temp/prediction.jpg", picture_data)

#picture_data = training_data_generator.load_picture_data("c:/temp/RealWorldBalls/20181026_193611 - 256x192.jpg")
#semantic_segmentation.predict_and_create_image("c:/temp/prediction.jpg", picture_data)

semantic_segmentation.train_own_model()
#cProfile.run('semantic_segmentation.train_own_model()')
