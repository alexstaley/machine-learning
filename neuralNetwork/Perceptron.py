"""
    Alex Staley -- Student ID: 919519311
        Assignment 2 -- February 2020

    ### HERE BEGINS THE Perceptron.py FILE ###

    This code defines the Perceptron class. It contains an array
    of randomly generated floating point values for weights, plus a
    dedicated target value and associated target vector. Its methods
    compute the dot product of inputs and weights, the sigmoid
    activation function, and the adjustment of weights (including momentum).

    Also declared here is a set of global constants serving as parameters.
    This includes the primary variables of all three experiments, which are
    defined and documented on lines 21-23 of this file.
"""

import numpy as np

# Experiment variables:
NUM_HIDDEN_UNITS = 20  # Number of perceptrons in the hidden layer     # EXPERIMENT 1: Default = 100 for exp 2, 3
PARTIAL = 1             # (Reciprocal of) Fraction of data to import    # EXPERIMENT 2: Default = 1 for exp 1, 3
MOMENTUM = 0            # Oscillation mitigator                         # EXPERIMENT 3: Default = 0 for exp 1, 2

# Experiment parameters:
ETA = 0.1                       # Learning rate of the network
NUM_EPOCHS = 25                 # Number of epochs in the learning cycle
PCT_DATA = (1 / PARTIAL) * 100  # Percent of training data in experiment 2
NUM_TRAINING_ROWS = 60000       # Number of rows in the training set
NUM_VALIDATION_ROWS = 10000     # Number of rows in the testing set
RESOLUTION = 255                # Maximum value for individual raw input elements
TRAINING_SCALAR = PARTIAL * (100 / NUM_TRAINING_ROWS)   # Scaling factor for computing accuracy
TESTING_SCALAR = 100 / NUM_VALIDATION_ROWS              # Scaling factor for computing accuracy

# Network parameters:
WEIGHT_RANGE = 0.05             # Maximum initial weight value
WEIGHT_SCALE = 10               # Reciprocal of scaling factor for random initial weight distribution
NUM_OUTPUT_UNITS = 10           # Number of perceptrons in the output layer
NUM_HIDDEN_INPUTS = 785         # Number of inputs for each perceptron in the hidden layer
NUM_OUTPUT_INPUTS = NUM_HIDDEN_UNITS+1      # Number of inputs for each perceptron in the output layer


class pTron(object):
    weights = np.array([])

    def __init__(self, target, num_inputs):
        """
        Initialize weights as an array of num_inputs random
        values between (-)WEIGHT_RANGE and (+)WEIGHT_RANGE
        """

        # Create target vector:
        self.target = target
        self.targetVector = np.full(NUM_OUTPUT_UNITS, 0.1)
        if target < NUM_OUTPUT_UNITS:
            self.targetVector[target] = 0.9

        # Initialize weights to random values between +/-WEIGHT_RANGE
        self.weights = np.random.rand(num_inputs)
        self.weights = np.multiply(self.weights, (1 / WEIGHT_SCALE))
        self.weights = np.add(self.weights, -WEIGHT_RANGE)

    # Take the inner product of the inputs and their corresponding weights.
    # Return the value of the sigmoid activation function, with inner product as input.
    def activate(self, features):
        """
        :param features: 1-d array of input values from one example
        :return: sigmoid activation function output
        """
        # Compute the inner product:
        dotProduct = np.dot(features, self.weights)

        # Invoke the sigmoid function:
        return 1 / (1 + np.exp(-dotProduct))

    # Update the weights, incorporating momentum
    def updateWeights(self, response, adjustment, updateMoment):
        """
        :param response: array of preceding layer's output values
        :param adjustment: factor of successive layer's error * learning rate
        :param updateMoment: weight change calculated at last update

        :return: bigDelta: magnitude of weights change
        """
        # Calculate momentum:
        momentum = np.multiply(updateMoment, MOMENTUM)

        # Calculate Delta_w:
        bigDelta = np.add(momentum, np.multiply(response, adjustment))

        # Update the weights:
        self.weights = np.add(self.weights, bigDelta)

        return bigDelta
