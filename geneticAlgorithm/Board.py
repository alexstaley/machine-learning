"""
Alex Staley -- 919519311
CS441, Program 2 -- February 2021

This file implements the Board class used to represent a valid state of the 8-Queens puzzle.
The Board class contains two attributes:
    * state: a 2-D numpy array of zeros and ones representing a chess board with zero or eight queens
    * fitness: a measure of the number of pairs of mutually attacking queens on the board

Implemented below are methods to:
    * Generate a valid state, either randomly or by splicing the states of two existing Board objects
    * Calculate the positions (i.e. 2-D indices) of the eight queens on the Board
    * Calculate, return, and manually set the fitness score of the Board
    * Induce a random mutation in a Board's state
    * Print the Board's state, queen indices, and fitness
"""
import numpy as np
import random as rd

MAX_ATTACKS = 28  # Maximum number of mutually attacking queen pairs (== 28 for 8-Queens)


class Board:
    def __init__(self):
        """
        Create an empty board; initialize fitness to 0
        """
        self.state = np.zeros((8, 8))
        self.fitness = 0

    def randomQueens(self):
        """
        Initialize random configuration of 8-queens puzzle.
        :return: this Board object (with fitness calculated)
        """
        qRow = rd.randint(0, 7)
        qCol = rd.randint(0, 7)
        for cell in range(8):
            while self.state[qRow, qCol]:
                qRow = rd.randint(0, 7)
                qCol = rd.randint(0, 7)
            self.state[qRow, qCol] = 1

        self.calculateFitness()
        return self

    def spliceQueens(self, parentA, parentB, xOver):
        """
        Set the state as a mix of the states of the two given Board objects
        :param parentA: Board object providing the front portion of the state
        :param parentB: Board object providing the back portion of the state
        :param xOver: integer representing the parentA/parentB crossover point
        :return: this Board object (with fitness calculated)
        """
        qA = parentA.getQueens()
        qB = parentB.getQueens()

        # Assign parent queens to child board
        for i in range(xOver):
            self.state[qA[i]] = 1
        for j in range(xOver, 8):
            self.state[qB[j]] = 1

        # Ensure child board still contains 8 queens
        while self.state.sum() < 8:
            choice = rd.randint(0, 7)
            self.state[qA[choice]] = 1
            if self.state.sum() == 8:
                break
            self.state[qB[choice]] = 1

        # Calculate child's fitness
        self.calculateFitness()
        return self

    def getQueens(self):
        """
        Get a list of the coordinates of the queens on this Board object
        :return: list of eight 2-tuples
        """
        queens = []
        for index, cell in np.ndenumerate(self.state):
            if cell:
                queens.append(index)
        return queens

    def getFitness(self):
        """
        Get this Board object's fitness
        :return: integer fitness value between 0 and MAX_ATTACKS
        """
        return self.fitness

    def setFitness(self, fakeFitness):
        """
        For testing edge cases
        :param fakeFitness: fitness score to set
        """
        self.fitness = fakeFitness

    def calculateFitness(self):
        """
        Counts the number of mutual attacks between queens on the board, without repeats
        """
        attacks = 0
        for r in range(8):
            for c in range(8):
                # For each queen, check for attacks
                if self.state[r, c]:
                    for col in range(8):
                        majDg = r - c + col  # upper left - lower right
                        minDg = r + c - col  # upper right - lower left
                        if not col == c:
                            # Check row
                            if self.state[r, col]:
                                attacks += 1
                            # Check major diagonal
                            if -1 < majDg < 8:
                                if self.state[majDg, col]:
                                    attacks += 1
                            # Check minor diagonal
                            if -1 < minDg < 8:
                                if self.state[minDg, col]:
                                    attacks += 1
                        else:
                            # Check column
                            for row in range(8):
                                if self.state[row, col] and not row == r:
                                    attacks += 1
        self.fitness = int(attacks / 2)

    def mutate(self):
        """
        Move one of the queens on the board to a different, randomly selected, empty cell
        """
        # Pick a random empty cell
        gA = rd.randint(0, 7)
        gB = rd.randint(0, 7)
        while self.state[gA, gB]:
            gA = rd.randint(0, 7)
            gB = rd.randint(0, 7)

        # Remove a random queen and place it in the empty cell
        queens = self.getQueens()
        flip = rd.randint(0, 7)
        self.state[queens[flip]] = 0
        self.state[gA, gB] = 1

    def print(self):
        """
        Print a visual representation of the state of the board
        """
        for r in range(8):
            for c in range(8):
                if self.state[r, c] == 0:
                    if r % 2 == c % 2:
                        print('-', end='  ')
                    else:
                        print('+', end='  ')
                elif self.state[r, c] == 1:
                    print('Q', end='  ')
            print('\n', end='')
        print("Queens:", self.getQueens())

