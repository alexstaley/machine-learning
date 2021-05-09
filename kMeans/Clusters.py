"""
    Alex Staley -- Student ID: 919519311
        Assignment 4 -- March 2020

    This file defines operations on the clusters of the K-clusters
    algorithm. Implemented below are the following functions:

        # assignClusters() assigns each training object to
            one of NUM_CLUSTERS clusters.
        # classifyClusters() associates each cluster with its
            best fit class by wrapping the classifyCluster() function.
        # classifyCluster() returns the class label that appears
            most often in a given cluster.
        # validateClusters() measures the accuracy of the trained
            cluster centers over the validation data.
        # visualizeClusters() produces a grayscale image of the
            trained cluster centers.
"""

import matplotlib.pyplot as plt
from kMeans.Getters import *


# Make cluster assignments based on centers
def assignClusters(features, centers):
    """
    :param features: training data set
    :param centers: mean values per cluster per attribute

    :return: clusters: nested list of clustered training data
    """
    clusters = [[] for x in range(NUM_CLUSTERS)]
    distances = np.array([])

    # Assign each object to its nearest center's cluster:
    for i in range(NUM_TRAINING_ROWS):
        for j in range(NUM_CLUSTERS):
            # Get Euclidean distance:
            distances = np.append(distances, np.linalg.norm(features[i, 0:-1] - centers[j]))
        # Assign the closest center:
        closestIndex = getClosestIndex(distances)
        clusters[closestIndex].append(features[i])
        # Clean up:
        distances = np.array([])

    return clusters


# Associate each cluster with its best fit class
def classifyClusters(clusters):
    """
    :param clusters: nested list of clustered training data

    :return: classyCenters: dictionary of centers with appended class labels
    """
    # Classify and center each cluster:
    labels = [classifyCluster(clusters[x]) for x in range(NUM_CLUSTERS)]
    centers = [getCenter(clusters[x]) for x in range(NUM_CLUSTERS)]

    # Append class labels:
    for i in range(NUM_CLUSTERS):
        centers[i] = np.append(centers[i], labels[i])

    return centers


# Assign a class label to a given cluster
def classifyCluster(cluster):
    """
    :param cluster: labeled training data assigned to a common cluster

    :return: most common label in cluster
    """
    clusterSize = len(cluster)
    labels = [0 for x in range(NUM_CLASSES)]
    currentMax = 0
    maxLabel = 0
    tied = [0]

    # Count the number of each class label in the cluster:
    for i in range(clusterSize):
        labels[cluster[i][-1]] += 1

    # Get the class with the most objects assigned:
    for i in range(NUM_CLASSES):
        if labels[i] > currentMax:
            currentMax = labels[i]
            maxLabel = i

    # Check for ties:
    tied[0] = maxLabel
    for i in range(NUM_CLASSES):
        if not i == maxLabel:
            if labels[i] == currentMax:
                tied.append(i)

    # If there is a tie, pick a winner at random:
    if not len(tied) == 1:
        winner = rand.randint(0, len(tied) - 1)
        return tied[winner]

    return maxLabel


# Classify validation data based on classified cluster centers:
def validateClusters(features, centers):
    """
    :param features: validation data set
    :param centers: mean values per cluster per attribute, labeled

    :return: predictedValues: array of classification predictions
    """
    predictedValues = np.zeros(NUM_VALIDATION_ROWS, dtype=int)
    distances = np.array([])

    # Determine the prediction of each validation object:
    for i in range(NUM_VALIDATION_ROWS):
        for j in range(NUM_CLUSTERS):
            # Get Euclidean distances to each center:
            distances = np.append(distances, np.linalg.norm(features[i, 0:-1] - centers[j][0:-1]))
        # Find the closest center:
        closestIndex = getClosestIndex(distances)
        predictedValues[i] = int(centers[closestIndex][-1])
        # Clean up:
        distances = np.array([])

    return predictedValues


# Represent cluster centers as a grayscale image
def visualizeClusters(centers):
    """
    :param: centers: array of labeled cluster centers
    """
    for i in range(NUM_CLUSTERS):
        # Track class label and square image:
        label = int(centers[i][-1])
        image = np.copy(centers[i][0:-1])
        image = np.resize(image, (8, 8))

        # Create image:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.set_title(
            "Center Visualization: " + str(NUM_CLUSTERS) + " clusters\nclass " + str(label))
        plt.show()


