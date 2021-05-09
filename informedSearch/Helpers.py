"""
Alex Staley -- 919519311
CS441 -- Program 1
January 31, 2021

This file implements three general helper functions:
    # parityCheck: Ensures the goal state can be reached from the given state
    # countInversions: Counts the metric used for heuristic #3
    # getManhattanDistance: Gets the Manhattan distance of a single tile
"""
import math

PUZZNUM = 8
DIM = math.isqrt(PUZZNUM+1)


def parityCheck(state):
    """
    Check if there are an even number of inversions in the state
    :param state: Disordered list of ints 1 - n
    :return: True if solution state is reachable from given state
    """
    if countInversions(state) % 2 == 0:
        return True
    return False


def countInversions(state):
    """
    Count the number of inversions in the given state
    :param state: Iterable (list of strings or ints expected, 'b' element accounted for)
    :return: number of inversions in the given state
    """
    n = len(state)
    invs = 0

    # Count inversions
    for i in range(n - 1):
        for j in range(i + 1, n):

            # Ignore blank tile 'b'
            if 'b' not in (state[i], state[j]):
                if state[i] > state[j]:
                    invs += 1
    return invs


def getManhattanDistance(tile, row, col):
    """
    Get the Manhattan distance of an individual tile in a given location.
    :param row: Current row of tile
    :param col: Current column of tile
    :param tile: {string} identity of tile
    :return: Number of squares the tile is from its goal position
    """
    # Calculate where the tile should be
    if tile == 'b':
        return 0
    else:
        goal = int(tile)
        if goal <= DIM:
            # Top row
            rowGoal = 1
        elif goal > PUZZNUM + 1 - DIM:
            # Bottom row
            rowGoal = DIM
        else:
            # Middle row
            rowGoal = 2
        colGoal = goal % DIM
        if colGoal == 0:
            colGoal = DIM

    # Calculate the Manhattan distance
    return abs(row - rowGoal) + abs(col - colGoal)
