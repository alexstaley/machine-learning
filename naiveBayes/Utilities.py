"""
    Alex Staley -- Student ID: 919519311
        Assignment 3 -- February 2020

    Implemented here is the low-level functionality of the
    naive Bayes classifier. Defined below are four functions:

        estimatePrior() computes the probability of each class
                        occurring in the validation set.

        getNumClasses() gets the number of classes defined in
                        the data set, along with an array
                        of the class values themselves.

        gaussian()      returns the value of the Gaussian
                        probability density function, given
                        mean, standard deviation and an input value.

        learnGreek()    computes the mean and standard deviation
                        associated with a given training object.

    Functions defined here are called in the Experiment.py file.
"""

import numpy as np
from collections import Counter


def estimatePrior(labels):
    """
    :param labels: ground truth labels for validation data
    :return: priors: dictionary of class instances
    """
    classes, numClasses = getNumClasses(labels)
    numObjects = np.size(labels)

    # Count class instances:
    priors = Counter(labels)

    # Compute prior probabilities per class:
    for i in range(numClasses):
        priors[classes[i]] = priors[classes[i]] / numObjects

    return priors


def getNumClasses(labels):
    """
    :param labels: ground truth labels for a data set

    :return: classes: array of class labels used
    :return: numClasses: number of classes in the data set
    """
    classes = np.array([], dtype=int)
    numClasses = 0
    found = False

    for i in range(np.size(labels)):
        for j in range(np.size(classes)):
            if classes[j] == labels[i]:
                found = True
                break
        if not found:
            classes = np.append(classes, labels[i])
            numClasses += 1
        else:
            found = False

    return classes, numClasses


def gaussian(feature, mean, stdDev):
    """
    :param feature: single value taken by an attribute
    :param mean: mean value associated with the corresponding class
    :param stdDev: standard deviation associated with the corresponding class

    :return: output of Gaussian for naive Bayes classifier over continuous data
    """
    exTerm = -1 * ((np.square(feature - mean)) / (2*np.square(stdDev)))
    piTerm = np.reciprocal(stdDev * np.sqrt(2*np.pi))

    return piTerm * np.exp(exTerm)


def learnGreek(features):
    """
    :param features: sub-array of all features associated with a given class label

    :return: mean: array of mean values for all attributes
    :return: stdDev: array of standard deviation values for all attributes
    """
    numAttributes = np.size(features, axis=1)

    mean = np.zeros(numAttributes)
    stdDev = np.zeros(numAttributes)

    for i in range(numAttributes):
        mean[i] = np.mean(features[:, i])
        stdDev[i] = np.std(features[:, i])

        # Ensure variance stays above 0.0001:
        if stdDev[i] < 0.01:
            stdDev[i] = 0.01

    return mean, stdDev
