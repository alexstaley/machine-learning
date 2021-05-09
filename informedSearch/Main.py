"""
Alex Staley -- 919519311
CS441 -- Program 1
January 31, 2021

This program implements an experiment in informed search for the 8-Puzzle solution.
Each of two algorithms and three heuristics are executed on five different starting states,
and the solution paths and average lengths are recorded and displayed.

The informed search algorithms tested in this experiment are:
    1 == Best-First
    2 == A*
For each algorithm, three different heuristics are used to inform the search function:
    1 == Manhattan distance
    2 == Number of misplaced tiles
    3 == Number of inversions

The user should run this program with a Python 3 interpreter.
"""
from informedSearch.Search import *

REPETITIONS = 5
HEURISTICS = 3
ALGORITHMS = 2


averages = [0, 0, 0, 0, 0, 0]

# Run experiment REPETITIONS times
for ex in range(REPETITIONS):
    stepsArray = []
    initNode = setupNPuzzle(PUZZNUM)

    # Execute search for each of HEURISTICS and ALGORITHMS
    for heu in range(HEURISTICS):
        for alg in range(ALGORITHMS):
            steps = searchFunction(initNode, alg+1, heu+1)
            stepsArray.append(steps)

    # After each experiment, total steps taken for each method
    for i in range(HEURISTICS * ALGORITHMS):
        averages[i] += stepsArray[i]

# Divide to calculate average number of steps
for i in range(HEURISTICS * ALGORITHMS):
    averages[i] = averages[i] / (HEURISTICS * ALGORITHMS)

# Record results
print("Avgs:", averages)
with open("../informedSearch.txt", 'a') as file:
    file.write("AVERAGES:\n")
    for i in range(HEURISTICS * ALGORITHMS):
        file.write(str(averages[i]))
        file.write('\n')
