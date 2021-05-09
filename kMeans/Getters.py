"""
    Alex Staley -- Student ID: 919519311
        Assignment 4 -- March 2020

    This file defines the low-level functionality of the K-clusters
    algorithm. Implemented below are the following functions:

        # getRandomCenters() picks NUM_CLUSTERS random training
            objects to use as initial cluster centers.
        # getCenter() calculates the mean (centroid) of a given cluster.
        # getClosestIndex() returns the index of the cluster center
            closest to an object.
        # getAvgMSE() computes the average mean square error over all clusters.
        # getMeanSqSep() computes the mean square separation of the clusters.
        # getMeanEntropy() computes the mean entropy of the clusters.
        # getEntropy() computes the entropy of a given cluster.
        # getConvergence() determines if the K-means algorithm has converged.
        # getConfused() processes prediction accuracy data into a
            confusion matrix.
"""

import numpy as np
import random as rand
from kMeans.Specs import *


# Pick NUM_CLUSTERS objects at random to act as initial centers
def getRandomCenters(features):
    """
    :param features: training data set

    :return: centers: mean values per cluster per attribute
    """
    unique = False
    centers = []
    centerIDs = np.array([])

    # Get NUM_CLUSTERS unique random indices:
    while not unique:
        centerIDs = np.random.randint(low=0, high=NUM_TRAINING_ROWS, size=NUM_CLUSTERS)
        unique = True
        for i in range(NUM_CLUSTERS):
            for j in range(NUM_CLUSTERS):
                if centerIDs[i] == centerIDs[j] and not i == j:
                    unique = False

    # Make an array of the corresponding features:
    for i in range(NUM_CLUSTERS):
        centers.append(features[centerIDs[i], 0:-1])

    return centers


# Compute the mean values of a single cluster:
def getCenter(cluster):
    """
    :param cluster: training objects assigned to a cluster

    :return: array of mean values of each attribute for a cluster
    """
    numObjects = np.size(cluster, axis=0)
    means = np.zeros(NUM_ATTRIBUTES)
    if numObjects == 0:
        return means

    # Get mean of each attribute over the cluster:
    for i in range(NUM_ATTRIBUTES):
        for j in range(numObjects):
            means[i] += (cluster[j][i])
        means[i] = means[i] / numObjects

    return means


# Get the index of the closest center to an object
def getClosestIndex(distances):
    """
    :param distances: array of distances from an object to all cluster centers
    :return: closestIndex: index of the closest cluster center to the object
    """
    size = np.size(distances)
    closest = LARGE_NUMBER
    closestIndex = 0
    tied = [0]

    # Find the minimum distance:
    for i in range(size):
        if distances[i] <= closest:
            closest = distances[i]
            closestIndex = i

    # Check for ties:
    tied[0] = closestIndex
    for i in range(size):
        if not i == closestIndex:
            if distances[i] == closest:
                tied.append(i)

    # If there is a tie, pick a winner at random:
    if not len(tied) == 1:
        winner = rand.randint(0, len(tied) - 1)
        return tied[winner]

    return closestIndex


# Calculate the average mean square error over all clusters
def getAvgMSE(centers, clusters):
    """
    :param centers: mean values per cluster per attribute
    :param clusters: nested list of clustered training objects

    :return: avgMSE: average mean square error over all clusters
    """
    mse = 0.
    avgMSE = [x for x in range(NUM_CLUSTERS)]

    for i in range(NUM_CLUSTERS):
        clusterSize = len(clusters[i])
        # Get the mean square error of each cluster:
        for j in range(clusterSize):
            # Omit ground truth labels:
            blindCluster = np.copy(clusters[i][j])
            blindCluster = np.delete(blindCluster, -1)
            mse += np.sum(np.square(np.subtract(blindCluster, centers[i])))
        mse = mse / clusterSize
        avgMSE[i] = mse
    avgMSE = np.mean(avgMSE)
    return avgMSE


# Calculate the mean square separation for the set of clusters
def getMeanSqSep(centers):
    """
    :param centers: mean values per cluster per attribute

    :return: meanSqSep: mean square separation
    """
    mss = 0.
    for i in range(NUM_CLUSTERS-1):
        for j in range(i+1, NUM_CLUSTERS):
            mss += np.sum(np.square(np.subtract(centers[i], centers[j])))
    mss = mss / (0.5*NUM_CLUSTERS*(NUM_CLUSTERS-1))
    return mss


# Calculate the mean entropy of the set of clusters
def getMeanEntropy(clusters):
    """
    :param clusters: nested list of clustered training data

    :return: meanEntropy: measure of disorder of cluster distribution
    """
    meanEntropy = 0.

    # Sum the entropy of each cluster:
    for i in range(NUM_CLUSTERS):
        clusterSize = len(clusters[i])
        meanEntropy += clusterSize * getEntropy(clusters[i])

    return meanEntropy / NUM_TRAINING_ROWS


# Calculate the entropy of a cluster
def getEntropy(cluster):
    """
    :param cluster: a cluster of training data (including ground truth label)

    :return: entropy: measure of disorder within the cluster
    """
    clusterSize = len(cluster)
    classCount = np.zeros(NUM_CLASSES, dtype=int)
    entropy = 0.

    # Count the instances of each class:
    for i in range(clusterSize):
        classCount[cluster[i][-1]] += 1

    # Calculate entropy:
    for i in range(NUM_CLASSES):
        if not classCount[i] == 0:
            prob = classCount[i]/NUM_TRAINING_ROWS
            entropy += (prob * np.log2(prob))
    return -entropy


# Check if the clusters have converged
def getConvergence(nClust, oClust):
    """
    :param nClust: new set of clustered training data
    :param oClust: old set of clustered training data

    :return: True if the clusters match. False otherwise
    """
    # Compare sizes of corresponding clusters:
    for i in range(NUM_CLUSTERS):
        if not len(oClust[i]) == len(nClust[i]):
            return False
    return True


# Generate the confusion matrix
def getConfused(labels, predictions):
    """
    :param labels: ground truth labels
    :param predictions: predicted class assignments

    :return: confusionMatrix: actual vs predicted classes
    """
    confusionMatrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    # Increment the corresponding entry in each (actual, predicted) pair
    for i in range(np.size(labels)):
        confusionMatrix[labels[i], predictions[i]] += 1
    return confusionMatrix
