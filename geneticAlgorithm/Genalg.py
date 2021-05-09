"""
Alex Staley -- 919519311
CS441, Program 2 -- February 2021

This file implements the bulk of the Genetic Algorithm.
Defined below are functions to:
    * Select the fittest parents from a population
    * Repopulate the universe from two parent Board objects

Also defined below are constant terms representing:
    * The number of individuals in each generation
    * The inverse proportion of mutations generated per generation
        (The actual mutation rate is calculated as POP_SIZE / MUTANT_RATE.
        In other words, one out of every MUTANT_RATE children suffers a mutation.)
"""
from geneticAlgorithm.Board import *

POP_SIZE = 100    # Default 100
MUTANT_RATE = 5   # Default 5 *** This is an inverse: greater value = fewer mutations ***


def selectParents(population):
    """
    Select the fittest two individuals in the given population for breeding
    :param population: a list of Board objects
    :return: tuple of two Board objects
    """
    minFit = [MAX_ATTACKS, MAX_ATTACKS]  # Fitness of two current fittest boards
    fittest = [-1, -1]                   # Indices of two current fittest boards
    for index, board in enumerate(population):
        fitness = board.getFitness()
        # Compare with current fittest
        if fitness < minFit[0]:
            minFit[1] = minFit[0]
            minFit[0] = fitness
            fittest[1] = fittest[0]
            fittest[0] = index
        # Compare with current second-fittest
        elif fitness < minFit[1]:
            minFit[1] = fitness
            fittest[1] = index
    return population[fittest[0]], population[fittest[1]]


def populate(parentA, parentB):
    """
    Generate a population of POP_SIZE individuals given two parent Board objects
    :param parentA: Fittest Board object of its generation
    :param parentB: Second-fittest Board object of its generation
    :return: list of POP_SIZE Board objects
    """
    population = []
    mutator = rd.randint(1, MUTANT_RATE)

    for pair in range(POP_SIZE // 2):
        # Produce two children
        xOver = rd.randint(1, 7)
        kidA = Board().spliceQueens(parentA, parentB, xOver)
        kidB = Board().spliceQueens(parentB, parentA, xOver)

        # Mutate one in every MUTANT_RATE children
        if rd.randint(1, MUTANT_RATE) == mutator:
            kidA.mutate()
        if rd.randint(1, MUTANT_RATE) == mutator:
            kidB.mutate()

        # Add children to population
        population.append(kidA)
        population.append(kidB)
    return population
