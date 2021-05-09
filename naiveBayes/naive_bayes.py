"""
    Alex Staley -- Student ID: 919519311
        Assignment 3 -- February 2020

    This program utilizes a naive Bayes classifier
    with a Gaussian distribution to classify a data set
    conforming to the format of the UCI data sets.

    The user should run the program from the command line,
    with the file path of the training set as the first argument
    and the file path of the validation set as the second, e.g.
python naive_bayes.py /file/path/to/training_set.txt /file/path/to/validation_set.txt

    Functions called here are implemented in the Experiment.py file.
"""

import sys
from naiveBayes.Experiment import *

trainingFeatures, trainingLabels = readFile(sys.argv[1], training=True)
validationFeatures, validationLabels = readFile(sys.argv[2], training=False)
print("\n")

mean, stdDev = learnData(trainingFeatures, trainingLabels)
print("\n\n\n")

classificationAccuracy = testBayes(validationFeatures, validationLabels, mean, stdDev)
print("\nClassification accuracy=%6.4f" % classificationAccuracy)
