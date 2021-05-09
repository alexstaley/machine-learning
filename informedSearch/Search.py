"""
Alex Staley -- 919519311
CS441 -- Program 1
January 31, 2021

This file implements a function to set up a randomly generated
initial state from which a solution is reachable, as well as
the general search function for a solution to the 8-puzzle.

Algorithms supported via argument to searchFunction are:
    1 == Best-First
    2 == A*
Heuristics supported via argument to searchFunction are:
    1 == Manhattan distance
    2 == Number of misplaced tiles
    3 == Number of inversions

The user can define the maximum number of iterations in the search function in
line 22: MOVELIMIT = {max iterations}
"""
from informedSearch.Queue import *

import random as rd

# Define move limit for search function
# and max length of explored list
MOVELIMIT = 100000
MAXEXPLORED = 5000


def setupNPuzzle(n):
    """
    Generate the (random) initial state for the n-puzzle
    :param n: number of non-blank squares
    :return: List of strings ['5', '2', 'b', ..., '8', '1'] where b = blank
    """
    # Ensure solvable starting state
    matchingParity = False
    while not matchingParity:
        state = rd.sample(range(1, n + 1), n)
        matchingParity = parityCheck(state)

    # Convert ints to strings and insert blank
    blank = rd.randint(0, n)
    strState = list(map(str, state))
    strState.insert(blank, 'b')

    # Get heuristic metrics and return starting node
    root = Node(strState, None, None, 0)
    root.mDist = root.getManhattan()
    root.misplaced = root.countMisplaced()
    root.inversions = countInversions(root.state)
    return root


def searchFunction(root, alg, heuristic):
    """
    Search for a solution according to the given algorithm and heuristic
    :param root: Node corresponding to starting state
    :param alg: 1 == Best-first; 2 == A*
    :param heuristic: 1 == Manhattan Distance; 2 == Misplaced Tiles; 3 == Inversions
    :return: Number of steps in solution path
    """
    path = np.array([root])
    children = np.array(root.expand(heuristic))
    pQ = Queue()

    # Expand root node (starting state)
    # and insert into priority queue according to heuristic
    for child in children:
        if alg == 1:
            pQ.insertBestFirst(child, heuristic)
        if alg == 2:
            pQ.insertAStar(child, heuristic)

    # Initialize explored list with starting state
    explored = np.array([root])

    reachedGoal = False
    moves = 0
    # Search for solution
    while not reachedGoal:
        if moves == MOVELIMIT:
            break

        # Select node from the front of the priority queue
        current = pQ.getBest(explored)

        # Check if goal state has been reached
        if current.inGoalState(heuristic):
            path = np.append(path, [current], axis=0)
            reachedGoal = True
            break

        # Add current node to solution path immediately following its parent
        path = path[:current.depth, ]
        path = np.append(path, [current], axis=0)

        # Expand current node
        children = np.array(current.expand(heuristic))

        # Add children to priority queue according to heuristic
        for child in children:
            if not child.isIn(path):
                if alg == 1:
                    pQ.insertBestFirst(child, heuristic)
                if alg == 2:
                    pQ.insertAStar(child, heuristic)

        # Add current node to explored list, starting it over
        # from scratch once it contains MAXEXPLORED nodes
        explored = np.append(explored, [current], axis=0)
        if len(explored) > MAXEXPLORED:
            explored = np.array([current])

        moves += 1

    # Record the number of steps in the solution, if found
    if reachedGoal:
        numSteps = len(path)
        for step in path:
            print(step.state)
        print("Solution has", numSteps, "steps.")
    else:
        numSteps = MOVELIMIT
        print(MOVELIMIT, "move limit reached; no solution found.")

    # Record the solution path and corresponding algorithm and heuristic
    with open("../informedSearch.txt", 'a') as file:
        file.write("Algorithm: ")
        file.write(str(alg))
        file.write("\nHeuristic: ")
        file.write(str(heuristic))
        if reachedGoal:
            file.write("\nSolution path:\n")
            for node in path:
                for item in node.state:
                    file.write(str(item))
                    file.write(", ")
                file.write('\n')
            file.write("-----")
            file.write(str(numSteps))
            file.write(" steps in solution\n\n\n")
        else:
            file.write("\nNo solution found\n\n\n")

    return numSteps
