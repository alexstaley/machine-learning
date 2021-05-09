"""
    Alex Staley -- Student ID: 919519311
        Assignment 2 -- February 2020

    ### HERE BEGINS THE Experiment.py FILE ###

    This code defines the high-level methods used to run the
    experiment described in assignment 2. It contains methods to
        # Import and properly parse the training and test data, ensuring even classification for Experiment 2
        # Run all three experiments as a set of learning epochs with input data and parameters
        # Process the output of the neural network
        # Generate a plot of accuracy on training and validation data over a set of learning epochs
        # Generate a confusion matrix of the final validation test of the experiment

    Parameters are set via global constants declared in the Perceptron.py file.
"""

from neuralNetwork.Network import *
import itertools as it
import matplotlib.pyplot as plt


# Import and interpret training and test data
def parseData(path, partial):
    """
    :param path: filename
    :param partial: (reciprocal of) fraction of data to import

    :return: pixels: training/test data matrix
    :return: targetVector: for computing accuracy
    :return: targetValues: corresponding values of targetVector
    """
    # Read target values:
    print("\tReading target values...")
    with open(path) as targetRaw:
        targetValues = np.genfromtxt(it.islice(targetRaw, 0, None, partial), usecols=(0,), delimiter=',', dtype=int)

    # Show data is classified evenly:
    if partial != 1:
        plt.hist(targetValues)
        plt.title("Input classes (" + str(PCT_DATA) + "% training data)")
        plt.ylabel("Number of classifications")
        plt.xlabel("Class")
        plt.show()
        plt.close()

    # Create a target vector:
    numRows = targetValues.size
    targetVector = np.full((numRows, NUM_OUTPUT_UNITS), 0.1)
    for t in range(numRows):
        targetVector[t, targetValues[t]] = 0.9

    # Read input data:
    numCols = range(1, NUM_HIDDEN_INPUTS)
    print("\tReading images...")
    with open(path) as pixelsRaw:
        pixels = np.genfromtxt(it.islice(pixelsRaw, 0, None, partial), usecols=numCols, delimiter=',')

    # Scale input values by 1/255
    numRows = len(pixels)
    for i in range(numRows):
        pixels[i] = pixels[i] / RESOLUTION

    # Append bias value 1 to each row
    biases = np.ones((numRows, 1))
    pixels = np.append(pixels, biases, axis=1)

    return pixels, targetVector, targetValues


# Execute epoch 0 and wrap the learningEpoch() function
def runExperiment(
        network, trainingSet, trainingVector, validationSet, validationVector, validationValues):
    """
    :param network: NeuralNetwork object being trained
    :param trainingSet: set of training examples
    :param trainingVector: element at index of corresponding actual value is 0.9
    :param validationSet: set of validation examples
    :param validationVector: element at index of corresponding actual value is 0.9
    :param validationValues: array of actual values

    :return: trainingAccuracy: accuracy over training set per epoch
    :return: validationAccuracy: accuracy over validation set per epoch
    :return: confusionMatrix: table of predicted vs actual values
    """
    hits = 0    # accuracy counter
    numTrainingRows = len(trainingSet)

    # Prepare accuracy tables:
    trainingAccuracy = np.array([])
    validationAccuracy = np.array([])

    # Create learning rate arrays:
    hiddenEta = np.full(NUM_HIDDEN_UNITS, ETA)
    outputEta = np.full(NUM_OUTPUT_UNITS, ETA)

    # Epoch 0: run training set with random weights
    print("\tRunning training set with random weights...")
    for k in range(numTrainingRows):
        # Forward-propagate each example:
        outputActivation, hiddenActivation, hiddenResponse = network.forwardPropagate(trainingSet[k, :])

        # Track accuracy:
        if checkAccuracy(outputActivation, trainingVector[k, :]):
            hits += 1

    # Record accuracy
    trainingAccuracy = np.append(trainingAccuracy, hits)
    hits = 0

    # Epoch 0: run validation set with random weights
    print("\tRunning validation set with random weights...")
    for k in range(NUM_VALIDATION_ROWS):
        # Forward-propagate each example:
        outputActivation, hiddenActivation, hiddenResponse = network.forwardPropagate(validationSet[k, :])

        # Track accuracy:
        if checkAccuracy(outputActivation, validationVector[k, :]):
            hits += 1

    # Record accuracy and prepare confusion matrix:
    validationAccuracy = np.append(validationAccuracy, hits)
    confusionMatrix = np.zeros((NUM_OUTPUT_UNITS, NUM_OUTPUT_UNITS))

    # Display status report:
    print("\tEpoch 0 of", NUM_EPOCHS, "complete.", hits, "hits on validation data.")

    # Execute learning experiment:
    for i in range(NUM_EPOCHS):
        predictedValues, trainAc, validAc = learningEpoch(
            network, trainingSet, trainingVector, validationSet, validationVector, hiddenEta, outputEta)

        # Record accuracy:
        trainingAccuracy = np.append(trainingAccuracy, trainAc)
        validationAccuracy = np.append(validationAccuracy, validAc)

        # After the final epoch, compile the confusion matrix:
        if i == NUM_EPOCHS - 1:
            confusionMatrix = getConfused(validationValues, predictedValues)

        # Report progress:
        print("\tEpoch", i+1, "of", NUM_EPOCHS, "complete.", end="\n\t")
        print(validAc, "hits on validation data,", trainAc, "on training data.")
    return trainingAccuracy, validationAccuracy, confusionMatrix


# Execute a complete learning epoch
def learningEpoch(
        network, trainingSet, trainingVector, validationSet, validationVector, hiddenEta, outputEta):
    """
    :param network: NeuralNetwork object being trained
    :param trainingSet: set of training examples
    :param trainingVector: element at index of corresponding actual value is 0.9
    :param validationSet: set of validation examples
    :param validationVector: element at index of corresponding actual value is 0.9
    :param hiddenEta: array of NUM_HIDDEN_UNITS full of ETA
    :param outputEta: array of NUM_OUTPUT_UNITS full of ETA

    :return: predictedValues: array of values predicted by the network
    :return: trainingAccuracy: array of accuracy scores over trainingSet per epoch
    :return: validationAccuracy: array of accuracy scores over validationSet per epoch
    """
    predictedValues = np.empty(NUM_VALIDATION_ROWS, dtype=int)
    trainingAccuracy = 0
    validationAccuracy = 0
    numTrainingRows = len(trainingSet)

    # Track each iteration's weight change for momentum factor:
    momentO = np.zeros((NUM_OUTPUT_UNITS, NUM_OUTPUT_INPUTS))
    momentH = np.zeros((NUM_HIDDEN_UNITS, NUM_HIDDEN_INPUTS))

    # Train network:
    print("\t\tTraining network...")
    for k in range(numTrainingRows):
        # Forward-propagate each example:
        outputActivation, hiddenActivation, hiddenResponse = network.forwardPropagate(trainingSet[k, :])

        # Calculate error:
        hiddenError, outputError = network.calculateError(
            trainingVector[k, :], outputActivation, hiddenActivation)

        # Back-propagate each example:
        momentO, momentH = network.backPropagate(
            trainingSet[k, :], outputError, hiddenError, hiddenResponse, hiddenEta, outputEta, momentO, momentH)

    # Test on training set:
    print("\t\tTesting on training set...")
    for k in range(numTrainingRows):
        # Forward-propagate each example:
        outputActivation, hiddenActivation, hiddenResponse = network.forwardPropagate(trainingSet[k, :])

        # Count correct classifications:
        if checkAccuracy(outputActivation, trainingVector[k, :]):
            trainingAccuracy += 1

    # Test on validation set:
    print("\t\tTesting on validation set...")
    for k in range(NUM_VALIDATION_ROWS):
        # Forward-propagate each example:
        outputActivation, hiddenActivation, hiddenResponse = network.forwardPropagate(validationSet[k, :])

        # Count correct classifications and record predicted values:
        classification, didHit = classify(outputActivation, validationVector[k, :])
        if didHit:
            validationAccuracy += 1
        predictedValues[k] = classification

    return predictedValues, trainingAccuracy, validationAccuracy


# Classify a prediction
def classify(outputActivation, targetVector):
    """
    :param outputActivation: array of NUM_OUTPUT_UNITS output values
    :param targetVector: element at index of corresponding actual value is 0.9

    :return: classification: actual value
    :return: didHit: True if accurate, False otherwise
    """
    # Determine classification:
    greatestOutput = -50.
    classification = -1
    for i in range(NUM_OUTPUT_UNITS):
        if greatestOutput < outputActivation[i]:
            greatestOutput = outputActivation[i]
            classification = i

    # Check classification:
    if targetVector[classification] == 0.9:
        didHit = True
    else:
        didHit = False

    return classification, didHit


# Check accuracy of a prediction
def checkAccuracy(outputActivation, targetVector):
    """
    :param outputActivation: array of NUM_OUTPUT_UNITS output values
    :param targetVector: element at index of corresponding actual value is 0.9

    :return: True if accurate, False otherwise
    """
    # Determine classification:
    greatestOutput = -50.
    classification = -1
    for i in range(NUM_OUTPUT_UNITS):
        if greatestOutput < outputActivation[i]:
            greatestOutput = outputActivation[i]
            classification = i

    # Check classification:
    if targetVector[classification] == 0.9:
        didHit = True
    else:
        didHit = False

    return didHit


# Generate an accuracy plot
def produceGraph(trainingAccuracy, validationAccuracy):
    """
    :param trainingAccuracy: accuracy over training set per epoch
    :param validationAccuracy: accuracy over validation set per epoch
    """
    # Display accuracies as a percentage:
    trainingAccuracy = np.multiply(trainingAccuracy, TRAINING_SCALAR)
    validationAccuracy = np.multiply(validationAccuracy, TESTING_SCALAR)

    # Display appropriate header for experiment:
    if PARTIAL != 1:
        header = "Accuracy (" + str(PCT_DATA) + "% training data)"
    elif MOMENTUM != 0:
        header = "Accuracy (momentum = " + str(MOMENTUM) + ")"
    else:
        header = "Accuracy (" + str(NUM_HIDDEN_UNITS) + " hidden units)"

    # Generate plot:
    plt.plot(trainingAccuracy)
    plt.plot(validationAccuracy)

    # Label plot:
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy over training data", "Accuracy over validation data"])
    plt.title(header)

    plt.show()
    plt.savefig("accuracy_plot.png")


# Generate the confusion matrix
def getConfused(actualValues, predictedValues):
    """
    :param actualValues: array of actual values in testing set
    :param predictedValues: array of classifications made by perceptron group

    :return: confusionMatrix: updated confusion matrix
    """
    confusionMatrix = np.zeros((NUM_OUTPUT_UNITS, NUM_OUTPUT_UNITS), dtype=int)

    # Increment the corresponding entry in each (actual, predicted) pair
    for i in range(NUM_VALIDATION_ROWS):
        confusionMatrix[actualValues[i], predictedValues[i]] += 1
    return confusionMatrix


# Process and print the confusion matrix
def produceMatrix(confusionMatrix):
    """
    :param confusionMatrix: table of actual vs predicted classes
    """
    # Display appropriate header for experiment:
    if PARTIAL != 1:
        header = "Confusion Matrix (" + str(PCT_DATA) + "% training data)"
    elif MOMENTUM != 0:
        header = "Confusion Matrix (momentum = " + str(MOMENTUM) + ")"
    else:
        header = "Confusion Matrix (" + str(NUM_HIDDEN_UNITS) + " hidden units)"

    label = np.arange(10)   # tick labels for matrix axes
    upperBound = 0          # upper bound of confused data

    # Determine range of confused data:
    for i in range(NUM_OUTPUT_UNITS):
        for j in range(NUM_OUTPUT_UNITS):
            if i != j and upperBound < confusionMatrix[i, j]:
                upperBound = confusionMatrix[i, j]
    upperBound += 10  # for color coherence

    # Generate a heat map:
    fig, ax = plt.subplots()
    ax.imshow(confusionMatrix, vmin=0, vmax=upperBound)

    # Label axes:
    ax.set_xlabel("Predicted classes")
    ax.set_ylabel("Actual classes")
    ax.set_xticks(label)
    ax.set_yticks(label)

    # Write values:
    for i in range(NUM_OUTPUT_UNITS):
        for j in range(NUM_OUTPUT_UNITS):
            if i != j:  # write confused data in white
                ax.text(j, i, confusionMatrix[i, j], ha="center", va="center", color="w")
            else:       # write accurate data in black
                ax.text(j, i, confusionMatrix[i, j], ha="center", va="center", color="k")
    ax.set_title(header)

    plt.show()
    plt.savefig("confusion_matrix.png")


