"""
    Alex Staley -- Student ID: 919519311
        Assignment 4 -- March 2020

    This file contains definitions for the local
    training and validation file paths, the number
    of clusters to learn, and various other
    experimental parameters.

    The user should specify the local file paths of
    training and validation data in lines 26 and 27
    below before running the program.

    Both training and validation data sets are expected
    to contain NUM_ATTRIBUTES attributes for each object,
    along with class labels in the greatest sub-index.
    That is, the shape of the data sets are expected to be
    (NUM_TRAINING_ROWS x (NUM_ATTRIBUTES +1)) and
    (NUM_VALIDATION_ROWS x (NUM_ATTRIBUTES +1)), with the "+1"
    representing the class labels.

    These parameters are hard-coded in lines 30-32 below.
"""

# **** DEFINE FILE PATHS FOR TRAINING AND TEST DATA HERE: **** #
TRAINING_FILEPATH = "../venv/optdigits.train"
VALIDATION_FILEPATH = "../venv/optdigits.test"

# **** DEFINE SHAPE OF TRAINING AND TEST DATA HERE: **** #
NUM_ATTRIBUTES = 64             # Number of attributes for each object
NUM_TRAINING_ROWS = 3823        # Number of objects in training set
NUM_VALIDATION_ROWS = 1797      # Number of objects in validation set

# **** DEFINE EXPERIMENTAL VARIABLE VALUE HERE: **** #
NUM_CLUSTERS = 30               # Number of clusters

# Other parameters:
NUM_RUNS = 5                    # Number of training runs
NUM_CLASSES = 10                # Number of classes
LARGE_NUMBER = 1000000.         # Ceiling for computing minimum MSE
