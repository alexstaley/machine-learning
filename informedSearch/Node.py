"""
Alex Staley -- 919519311
CS441 -- Program 1
January 31, 2021

This file implements the Node class used to define each node in the search tree.

Initially defined are the parameters:
    # state: A list of strings representing the state, e.g. ['1', '5', '3', '6', 'b', 4', '8', '2', '7']
    # parent: A Node object representing the parent node in the search tree
    # action: The action taken to obtain the current node from its parent: 'L', 'R', 'U', or 'D'
    # depth: The depth of the node within the search tree

Additionally defined according to the specified heuristic are the instance variables:
    # mDist: The Manhattan distance of the node
    # misplaced: The number of misplaced tiles in the node
    # inversions: The number of inversions in the node
"""
from informedSearch.Helpers import *

import numpy as np


class Node:
    def __init__(self, state, parent, action, depth):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth

    def isEqual(self, friend):
        for i in range(PUZZNUM+1):
            if not friend.state[i] == self.state[i]:
                return False
        return True

    def isIn(self, path):
        for node in path:
            if self.isEqual(node):
                return True
        return False

    def moveLeft(self, b):
        ns = self.state.copy()
        ns[b], ns[b-1] = ns[b-1], ns[b]
        return Node(ns, self, 'L', self.depth+1)

    def moveRight(self, b):
        ns = self.state.copy()
        ns[b], ns[b+1] = ns[b+1], ns[b]
        return Node(ns, self, 'R', self.depth+1)

    def moveUp(self, b):
        ns = self.state.copy()
        ns[b], ns[b-3] = ns[b-3], ns[b]
        return Node(ns, self, 'U', self.depth+1)

    def moveDown(self, b):
        ns = self.state.copy()
        ns[b], ns[b+3] = ns[b+3], ns[b]
        return Node(ns, self, 'D', self.depth+1)

    def inGoalState(self, heuristic):
        """
        Check if the current node is in the goal state, as measured by the given heuristic
        :param heuristic: 1 == Manhattan Distance; 2 == Misplaced Tiles; 3 == Inversions
        :return: Boolean if the node is in its goal state
        """
        if heuristic == 1:
            return self.getManhattan() == 0
        if heuristic == 2:
            return self.countMisplaced() == 0
        if heuristic == 3:
            return countInversions(self.state) == 0

    def isOverBFS(self, friend, heuristic):
        """
        Measure the node against another node using the given heuristic,
        according to the best-first search algorithm
        :param friend: Node to be compared against
        :param heuristic: 1 == Manhattan Distance; 2 == Misplaced Tiles; 3 == Inversions
        :return: Boolean if current node's heuristic value is >= that of its friend
        """
        if heuristic == 1:
            return self.getManhattan() >= friend.mDist
        if heuristic == 2:
            return self.countMisplaced() >= friend.misplaced
        if heuristic == 3:
            return countInversions(self.state) >= friend.inversions

    def isOverAS(self, friend, heuristic):
        """
        Measure the node against another node using the given heuristic,
        according to the A* search algorithm
        :param friend: Node to be compared against
        :param heuristic: 1 == Manhattan Distance; 2 == Misplaced Tiles; 3 == Inversions
        :return: Boolean if current node's heuristic value is >= that of its friend
        """
        if heuristic == 1:
            return self.getManhattan() + self.depth >= friend.mDist + friend.depth
        if heuristic == 2:
            return self.countMisplaced() + self.depth >= friend.misplaced + friend.depth
        if heuristic == 3:
            return countInversions(self.state) + self.depth >= friend.inversions + friend.depth

    def expand(self, heuristic):
        """
        Expand the node, generating a list of 2-4 child nodes.
        Calculate the heuristic metric of each child.
        :param heuristic: 1 == Manhattan Distance; 2 == Misplaced Tiles; 3 == Inversions
        :return: List of child nodes
        """
        blank = self.state.index('b')
        children = np.array([])

        # Generate a child for each possible move
        if blank not in (0, 3, 6):  # Move left
            children = np.append(children, [self.moveLeft(blank)], axis=0)
        if blank not in (2, 5, 8):  # Move right
            children = np.append(children, [self.moveRight(blank)], axis=0)
        if blank not in (0, 1, 2):  # Move up
            children = np.append(children, [self.moveUp(blank)], axis=0)
        if blank not in (6, 7, 8):  # Move down
            children = np.append(children, [self.moveDown(blank)], axis=0)

        # Calculate appropriate heuristic metric of each child
        for child in children:
            if heuristic == 1:
                child.mDist = child.getManhattan()
            if heuristic == 2:
                child.misplaced = child.countMisplaced()
            if heuristic == 3:
                child.inversions = countInversions(child.state)
        return children

    def getManhattan(self):
        """
        Get the Manhattan distance of the node
        :return: Manhattan distance (int)
        """
        distance = 0
        row = 0
        col = 0
        for tile, i in zip(self.state, range(PUZZNUM+1)):

            # Determine current location of tile
            if i % DIM == 0:
                row += 1
                col = 1
            if i in (1, 2, 4, 5, 7, 8):
                col += 1

            # Calculate Manhattan distance of tile
            distance += getManhattanDistance(tile, row, col)
        return distance

    def countMisplaced(self):
        """
        Count the number of misplaced tiles in the Node's state
        :return: Number of misplaced tiles (int)
        """
        numMisplaced = 0
        for i in range(PUZZNUM):

            # Guard against int('b')
            if self.state[i] == 'b':
                numMisplaced += 1

            # Check if tile is misplaced
            elif not i+1 == int(self.state[i]):
                numMisplaced += 1

        # Check the final square for a 'b'
        if not self.state[PUZZNUM] == 'b':
            numMisplaced += 1
        return numMisplaced
