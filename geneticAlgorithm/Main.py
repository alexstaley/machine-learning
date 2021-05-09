"""
Alex Staley -- 919519311
CS441, Program 2 -- February 2021

This program finds a solution for the 8-Queens puzzle using a Genetic Algorithm.
Iterating over a maximum number of generations, the program will search the solution
space. The program will print the generation number and average fitness of each generation.
The program will also print a depiction of the board for the fittest breeding parents
of each generation (if the parent's fitness is less than 4).

When a solution is found or the maximum generation number reached, the program will
plot the average fitness as a function of the generation number, and display the board
configuration of a selection of fittest individuals throughout the life of the experiment.
"""
from geneticAlgorithm.Experiment import *


gens = NUM_ITERS                    # Counter for the number of generations created
fitnessAvgs = []                    # List of average fitnesses of each generation
exampleBoards = []                  # List of selected Board objects to display
exBoardIndices = defineSamples()    # List of generation numbers of selected Board objects
population = setupPopulation()      # Population of randomly generated starting states

for generation in range(NUM_ITERS):
    print("\n***** Generation", generation, "*****")

    # Determine average fitness of the generation
    averageFitness = getAvgFitness(population)
    print("Average fitness:", averageFitness)
    fitnessAvgs.append(averageFitness)

    # Select parents for breeding
    parentA, parentB = selectParents(population)
    bestFit = parentA.getFitness()

    # Save selected board states for the results
    if generation in exBoardIndices:
        exampleBoards.append(parentA)

    # Check if the goal has been reached
    if bestFit == 0:
        print("Solution found!")
        parentA.print()
        gens = generation
        exampleBoards.append(parentA)
        break

    # Print board states with fitness < 4
    if bestFit < 4:
        print("Fitness level", bestFit, "reached:")
        parentA.print()

    # Repopulate the next generation
    population = populate(parentA, parentB)

# Display the results
displayGraph(fitnessAvgs, gens)
displayBoards(exampleBoards, exBoardIndices, gens)

