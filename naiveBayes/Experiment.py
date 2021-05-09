"""
    Alex Staley -- Student ID: 919519311
        Assignment 3 -- February 2020

    Implemented here is the high-level functionality of the
    naive Bayes classifier. Defined below are four functions:

        readFile()  imports training and validation data,
                    sorting training data by class and
                    separating ground truth labels.

        learnData() obtains the mean and standard deviation
                    values associated with the data.

        testBayes() classifies the data in the validation set,
                    acting as a wrapper for the classify() function.

        classify()  classifies each object in the validation set.

    Functions defined here are called in the naive_bayes.py file.
    Functions called here are implemented in the Utilities.py file.
"""

from naiveBayes.Utilities import *


def readFile(path, training):
    """
    :param path: local file path to read
    :param training: false if reading validation data

    :return: rawFeatures: array of attributes
    :return: labels: ground truth (class) labels
    """
    # Read entire file:
    with open(path) as targetRaw:
        rawFeatures = np.genfromtxt(targetRaw, delimiter='', dtype=float)

    # Copy and cut ground truth column:
    labels = np.array(rawFeatures[:, -1], dtype=int)
    rawFeatures = np.delete(rawFeatures, -1, axis=1)

    # Return validation data:
    if not training:
        return rawFeatures, labels

    # Sort training data by class label:
    classIndices = np.argsort(labels, kind="stable")
    labels = np.sort(labels, kind="stable")
    features = np.zeros(rawFeatures.shape)
    for i in range(np.size(classIndices)):
        np.copyto(features[i], rawFeatures[classIndices[i]])

    return features, labels


def learnData(features, labels):
    """
    :param features: array of training data, ordered by class
    :param labels: array of ground truth labels, ordered by class

    :return: totalMean: array of mean values
    :return: totalStdDev: array of standard deviations
    """
    classes, numClasses = getNumClasses(labels)
    numAttributes = np.size(features, axis=1)
    totalMean = np.array([])
    totalStdDev = np.array([])
    j = 0

    for i in range(numClasses):
        subFeatures = np.array([])
        beginRange = j  # Start at the next class
        for j in range(beginRange, np.size(labels)):
            if labels[j] != classes[i]:
                break  # Stop between classes
            # Assemble features for a class
            subFeatures = np.append(subFeatures, features[j, :])
        subFeatures = np.resize(subFeatures, (int(np.size(subFeatures)/numAttributes), numAttributes))

        # Get mu and sigma:
        mean, stdDev = learnGreek(subFeatures)

        # Print results:
        for k in range(numAttributes):
            print("Class %d" % classes[i], end=", ")
            print("attribute %d" % (k+1), end=", ")
            print("mean = %.2f" % mean[k], end=", ")
            print("std = %.2f" % stdDev[k], end="\n")

        # Update total arrays:
        totalMean = np.append(totalMean, mean)
        totalStdDev = np.append(totalStdDev, stdDev)

    # Resize and return final total arrays:
    totalMean = np.resize(totalMean, (numClasses, numAttributes))
    totalStdDev = np.resize(totalStdDev, (numClasses, numAttributes))
    return totalMean, totalStdDev


def testBayes(features, labels, mean, stdDev):
    """
    :param features: validation data set
    :param labels: validation ground truth labels
    :param mean: array of mean values
    :param stdDev: array of standard deviations

    :return: classification accuracy
    """
    numRows = np.size(features, axis=0)
    classes, numClasses = getNumClasses(labels)
    classificationAccuracy = np.array([])
    priors = estimatePrior(labels)

    for i in range(numRows):
        # Classify each row:
        predictedClass, probPredicted, accuracy = classify(
                features[i, :], int(labels[i]), mean, stdDev, numClasses, priors[int(labels[i])])

        # Print results:
        print("ID=%5d" % (i+1), end=", ")
        print("predicted=%3d" % predictedClass, end=", ")
        print("probability = %.4f" % probPredicted, end=", ")
        print("true=%3d" % labels[i], end=", ")
        print("accuracy=%4.2f" % accuracy, end="\n")

        classificationAccuracy = np.append(classificationAccuracy, accuracy)

    return np.mean(classificationAccuracy)


def classify(features, label, mean, stdDev, numClasses, prior):
    """
    :param features: all attributes of one validation object
    :param label: label of validation object
    :param mean: array of mean values
    :param stdDev: array of standard deviations
    :param numClasses: number of classes
    :param prior: appropriate class probability

    :return: winner: predicted class
    :return: currentMax: "probability" of predicted class
    :return: accuracy: 1 for correct prediction, 0 for incorrect
    """
    numAttributes = np.size(features)
    tied = np.array([numClasses])
    numTied = 0
    currentMax = -100
    winner = -1

    # Loop over classes
    for i in range(numClasses):
        prob = np.log10(prior)

        # Apply Gaussian for each attribute:
        for j in range(numAttributes):
            prob += np.log10(max(gaussian(features[j], mean[i, j], stdDev[i, j]), .001))

        # Record max probability and corresponding class:
        if prob > currentMax:
            currentMax = prob
            winner = i+1

            # Erase record of overtaken tie:
            if numTied > 0:
                tied = np.array([numClasses])
                numTied = 0

        # Deal with ties:
        elif prob == currentMax:
            tied = np.append(tied, i+1)
            numTied += 1
            # First tie:
            if numTied == 1:
                tied = np.append(tied, winner)
                numTied += 1
            winner = i+1

    # Resolve accuracy:
    if numTied > 0:
        accuracy = 1/numTied
        winner = tied[np.random.randint(0, numTied)]
    else:
        if winner == label:
            accuracy = 1
        else:
            accuracy = 0

    return winner, currentMax, accuracy
