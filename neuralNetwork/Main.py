"""
    Alex Staley -- Student ID: 919519311
        Assignment 2 -- February 2020

    ### HERE BEGINS THE Main.py FILE ###

    This code creates a neural network of perceptron objects, runs the experiments
    described in assignment 2, and displays the results in the prescribed format.

    Parameters are set via global constants declared in the Perceptron.py file.
    This includes the primary variables of each experiment in the assignment:
                # NUM_HIDDEN_UNITS  # Experiment 1 variable--default 100
                # PARTIAL           # Experiment 2 variable--default 1
                # MOMENTUM          # Experiment 3 variable--default 0

    File paths are defined in this file, lines 22/23. Change as needed.
"""

from neuralNetwork.Experiment import *

# DEFINE FILE PATHS FOR TRAINING AND VALIDATION DATA HERE:
trainingFilepath = "./venv/mnist_train.csv"
validationFilepath = "./venv/mnist_test.csv"

print("\nImporting training file...")
trainingSet, trainingVector, trainingValues = parseData(trainingFilepath, PARTIAL)
print("Importing validation file...")
validationSet, validationVector, validationValues = parseData(validationFilepath, partial=1)

print("\nInitializing neural network...\n")
network = NeuralNetwork()

print("########  RUNNING EXPERIMENT:  ########", end="\n\t\t")
print(NUM_HIDDEN_UNITS, "hidden units", end="\n\t\t")
print(PCT_DATA, "of training data", sep="% ")
print("\t\tMomentum =", MOMENTUM, '\n')
trainingAccuracy, validationAccuracy, confusionMatrix = runExperiment(
    network, trainingSet, trainingVector, validationSet, validationVector, validationValues)

print("\nExperiment complete! Displaying results")
produceGraph(trainingAccuracy, validationAccuracy)
produceMatrix(confusionMatrix)
