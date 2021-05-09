"""
    Alex Staley -- Student ID: 919519311
        Assignment 4 -- March 2020

    This file defines the high-level functionality of the K-clusters
    algorithm. Implemented below are the following functions:

        # readfile() imports raw data into a numpy array. Class
            labels are included but not used in training.
        # trainKclusters() oversees NUM_RUNS iterations of
            training, reports the results of the best
            iteration, and classifies the resulting clusters.
        # executeTrainingRun() executes the main loop of the K-means
            clustering algorithm and computes the required data.
        # testKclusters() tests the accuracy of the clusters
            resulting from the training algorithm.
        # produceMatrix() processes the results of the
            validation test into a confusion matrix.
"""

from kMeans.Clusters import *


# Import and process data sets
def readFile(path):
    """
    :param path: local file path to read

    :return: features: array of attributes
    """
    # Interpret file:
    with open(path) as targetRaw:
        features = np.genfromtxt(targetRaw, delimiter=',', dtype=int)

    return features


# Define a set of clusters on training data, over NUM_RUNS trials
def trainKclusters(features):
    """
    :param features: training data set

    :return: classyCenters: mean values per cluster per attribute, with appended class label
    """
    bestClusters = []
    bestTrial = -1
    currentMaxAMSE = LARGE_NUMBER
    meanSS = 0.
    meanEnt = 0.

    # Execute NUM_RUNS runs:
    for i in range(NUM_RUNS):
        print("\tTraining clusters: iteration", i+1)
        clusters, trialAMSE, trialMSS, trialEnt = executeTrainingRun(getRandomCenters(features), features)

        # Track minimum average mean square error, record leader's data:
        if currentMaxAMSE > trialAMSE:
            bestTrial = i+1
            bestClusters = clusters.copy()
            currentMaxAMSE = trialAMSE
            meanSS = trialMSS
            meanEnt = trialEnt

    # Assign a class value to each cluster:
    classyCenters = classifyClusters(bestClusters)

    # Display results of best run, return its centers:
    print("\nFor the best trial (trial", bestTrial, end="):\n")
    print("\tAverage mean square error = %.5f" % currentMaxAMSE)
    print("\tMean square separation = %.5f" % meanSS)
    print("\tMean entropy = %.5f" % meanEnt)
    return classyCenters


# Main body of learning algorithm
def executeTrainingRun(centers, features):
    """
    :param centers: initial (random) mean values per cluster per attribute
    :param features: training data set

    :return: clusters: nested list of clustered training data
    :return: avgMSE: average mean square error
    :return: meanSqSep: mean square separation
    :return: meanEnt: mean entropy
    """
    iteration = 1
    converged = False

    # First iteration:
    oldClusters = assignClusters(features, centers)
    centers = [getCenter(oldClusters[x]) for x in range(NUM_CLUSTERS)]
    avgMSE = getAvgMSE(centers, oldClusters)
    print("Main loop: lap", iteration, end=". ")
    print("Average mean square error = %.5f" % avgMSE)
    clusters = assignClusters(features, centers)

    # Main loop:
    while not converged:
        converged = True
        iteration += 1
        centers = [getCenter(clusters[x]) for x in range(NUM_CLUSTERS)]
        avgMSE = getAvgMSE(centers, clusters)
        print("Main loop: lap", iteration, end=". ")
        print("Average mean square error = %.5f" % avgMSE)

        oldClusters = np.copy(clusters)
        clusters = assignClusters(features, centers)

        # Check for convergence:
        converged = getConvergence(clusters, oldClusters)

    # Calculate average MSE, MSS and entropy:
    avgMSE = getAvgMSE(centers, clusters)
    meanSqSep = getMeanSqSep(centers)
    meanEnt = getMeanEntropy(clusters)

    return clusters, avgMSE, meanSqSep, meanEnt


# Test the defined clusters on validation data
def testKclusters(centers, features):
    """
    :param centers: mean values per class per attribute, with class labels
    :param features: validation data set

    :return: confusionMatrix: accurate vs predicted values
    """

    # Test clusters on validation data:
    predictedValues = validateClusters(features, centers)

    # Generate and return raw confusion matrix:
    labels = np.array(features[:, -1], dtype=int)
    return getConfused(labels, predictedValues)


# Process and print the confusion matrix
def produceMatrix(confusionMatrix):
    """
    :param confusionMatrix: table of actual vs predicted classes
    """
    label = np.arange(10)   # tick labels for matrix axes
    upperBound = 0          # upper bound of confused data
    confusedTotal = 0
    accurateTotal = 0

    # Determine accuracy and range of confused data:
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i == j:
                accurateTotal += confusionMatrix[i, j]
            else:
                confusedTotal += confusionMatrix[i, j]
                if upperBound < confusionMatrix[i, j]:
                    upperBound = confusionMatrix[i, j]
    upperBound += 10  # for color coherence
    accuracy = round(accurateTotal / (accurateTotal+confusedTotal), 2)

    # Display appropriate header for experiment:
    header = "Confusion Matrix (" + str(NUM_CLUSTERS) + " clusters)\naccuracy = " + str(accuracy)

    # Generate a heat map:
    fig, ax = plt.subplots()
    ax.imshow(confusionMatrix, vmin=0, vmax=upperBound)

    # Label axes:
    ax.set_xlabel("Predicted classes")
    ax.set_ylabel("Actual classes")
    ax.set_xticks(label)
    ax.set_yticks(label)

    # Write values:
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j:  # write confused data in white
                ax.text(j, i, confusionMatrix[i, j], ha="center", va="center", color="w")
            else:       # write accurate data in black
                ax.text(j, i, confusionMatrix[i, j], ha="center", va="center", color="k")
    ax.set_title(header)

    plt.show()
