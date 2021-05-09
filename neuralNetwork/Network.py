"""
    Alex Staley -- Student ID: 919519311
        Assignment 2 -- February 2020

    ### HERE BEGINS THE Network.py FILE ###

    This code defines the NeuralNetwork class. It contains two arrays
    of Perceptron objects, representing a hidden layer and an output layer
    of neurons. Methods implemented here execute forward propagation
    through both layers, calculation of the error associated with each
    training example, and back propagation through both layers.

    Parameters are set via global constants declared in the Perceptron.py file.
"""

from neuralNetwork.Perceptron import *


class NeuralNetwork(object):
    # Two connected layers of perceptron-like objects
    hiddenLayer = np.array([])
    outputLayer = np.array([])

    def __init__(self):
        for i in range(NUM_HIDDEN_UNITS):
            self.hiddenLayer = np.append(self.hiddenLayer, pTron(i, NUM_HIDDEN_INPUTS))
        for i in range(NUM_OUTPUT_UNITS):
            self.outputLayer = np.append(self.outputLayer, pTron(i, NUM_OUTPUT_INPUTS))

    def forwardPropagate(self, features):
        """
        :param features: 1-d array of input fodder

        :return: outputActivation: for error calculation
        :return: hiddenActivation: for error calculation
        :return: hiddenActivationWithBias: for back propagation
        """

        bias = 1
        hiddenActivation = np.array([])
        outputActivation = np.array([])

        # Propagate inputs thru hidden layer:
        for i in range(NUM_HIDDEN_UNITS):
            hiddenActivation = np.append(hiddenActivation, self.hiddenLayer[i].activate(features))

        # Append bias value:
        hiddenActivationWithBias = np.append(hiddenActivation, bias)

        # Propagate hidden activations thru output layer:
        for i in range(NUM_OUTPUT_UNITS):
            outputActivation = np.append(outputActivation, self.outputLayer[i].activate(hiddenActivationWithBias))

        return outputActivation, hiddenActivation, hiddenActivationWithBias

    def calculateError(self, targetVector, outputActivation, hiddenActivation):
        """
        :param targetVector: one training example, 0.9 for the actual value index, 0.1 elsewhere
        :param outputActivation: array of activation values from the output layer
        :param hiddenActivation: array of activation values from the hidden layer

        :return: hiddenError: for back propagation hidden -> input layer
        :return: outputError: for back propagation output -> hidden layer
        """
        # Calculate output error:
        outputError = np.multiply(
            outputActivation, np.multiply(
                np.subtract(np.ones(NUM_OUTPUT_UNITS), outputActivation), np.subtract(
                    targetVector, outputActivation)))

        # Get hidden-output weights array:
        feedbackWeights = np.empty(NUM_OUTPUT_UNITS)
        errorPropagation = np.empty(NUM_HIDDEN_UNITS)
        for i in range(NUM_HIDDEN_UNITS):
            for j in range(NUM_OUTPUT_UNITS):
                feedbackWeights[j] = self.outputLayer[j].weights[i]
            errorPropagation[i] = np.dot(feedbackWeights, outputError)

        # Calculate hidden error:
        hiddenError = np.multiply(
            hiddenActivation, np.multiply(
                np.subtract(np.ones(NUM_HIDDEN_UNITS), hiddenActivation), errorPropagation))

        return hiddenError, outputError

    def backPropagate(self, features, outputError, hiddenError, hiddenResponse, hiddenEta, outputEta, lastOut, lastHid):
        """
        :param features: one training example
        :param outputError: output -> hidden weight update factor
        :param hiddenError: hidden -> input weight update factor
        :param hiddenResponse: activation values for hidden->input layer including bias
        :param hiddenEta: array full of learning rate values for hidden layer
        :param outputEta: array full of learning rate values for output layer
        :param lastOut: bigDelta value for last update's momentum (output layer)
        :param lastHid: bigDelta value for last update's momentum (hidden layer)

        :return: lastOut: bigDelta value for next update's momentum (output layer)
        :return: lastHid: bigDelta value for next update's momentum (hidden layer)
        """
        # Update weights in output layer:
        outputAdjustment = np.multiply(outputEta, outputError)
        for i in range(NUM_OUTPUT_UNITS):
            lastOut[i] = self.outputLayer[i].updateWeights(
                hiddenResponse, outputAdjustment[i], lastOut[i])

        # Update weights in hidden layer:
        hiddenAdjustment = np.multiply(hiddenEta, hiddenError)
        for i in range(NUM_HIDDEN_UNITS):
            lastHid[i] = self.hiddenLayer[i].updateWeights(
                features, hiddenAdjustment[i], lastHid[i])

        return lastOut, lastHid
