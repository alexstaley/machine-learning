"""
    Alex Staley -- Student ID: 919519311
        Assignment 4 -- March 2020

    This program classifies handwritten digits
    from the UCI ML directory using an unsupervised
    K-means clustering algorithm.

    Before running the program the user should
    specify the local file paths of the
    training and validation data sets in
    lines 26 and 27 of the Specs.py file.

    The primary variable of the experiment
    described in assignment 4, the number of
    clusters to learn, is assigned the value
    NUM_CLUSTERS in line 35 of the Specs.py file.
"""

from kMeans.Experiment import *

print("\nImporting training file...")
trainingFeatures = readFile(TRAINING_FILEPATH)
print("Importing validation file...")
validationFeatures = readFile(VALIDATION_FILEPATH)

print("\nSeparating data into", NUM_CLUSTERS, end=' ')
print("clusters over", NUM_RUNS, "trials...")
centers = trainKclusters(trainingFeatures)

print("\nTesting clusters on validation data...")
confusionMatrix = testKclusters(
    centers, validationFeatures)

print("Testing complete. Displaying results")
produceMatrix(confusionMatrix)
visualizeClusters(centers)
