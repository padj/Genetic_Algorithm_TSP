## 
# Bespoke module created to solve the 2D generalised travelling salesman 
# problem via a genetic algorithm, generalised to allow for comparisons of 
# GA parameters.

# Future additions to add multiple selection and crossover processes. 

#%% PREAMBLE
import numpy as np
import random
import matplotlib.pyplot as plt

#%% FUNCTION DEFINITIONS

def makeTowns(nTowns, xMin, xMax, yMin, yMax, RNG):
    towns = []
    if RNG == False:
        random.seed(a=1)
    for i in range(nTowns):
        towns.append([random.uniform(xMin, xMax), random.uniform(yMin, yMax)])
    random.seed()
    return towns  
    
def createPopulation(numberOfTowns, popSize):
    population = []
    sampleList = range(numberOfTowns)
    for i in range(popSize):
        population.append(random.sample(sampleList, numberOfTowns))
    return population


def calcFitness(idv, towns):
    # fitness is the cumulative Euclidean distance between each city in the 
    # route order, including the distance between the last and first cities.
    fitness = 0 #initialise fitness
    for i in range(len(idv)-1):
        xDiff = towns[idv[i]][0] - towns[idv[i+1]][0]
        yDiff = towns[idv[i]][1] - towns[idv[i+1]][1]
        fitness += np.sqrt(xDiff**2 + yDiff**2)
    # Add length from last city, back to first city
    xDiff = towns[idv[0]][0] - towns[idv[-1]][0]
    yDiff = towns[idv[0]][1] - towns[idv[-1]][1]
    fitness += np.sqrt(xDiff**2 + yDiff**2)
    return fitness


def determineFitnesses(population, towns):
    fitnesses = []
    for i in range(len(population)):
        fitnesses.append(calcFitness(population[i], towns))
    return fitnesses

        
def initialise(nTowns, nPop, xLowerBound, xUpperBound, yLowerBound, 
                  yUpperBound, RNG):
    towns = makeTowns(nTowns, xLowerBound, xUpperBound, yLowerBound, 
                      yUpperBound, RNG) # create towns
    
    population = createPopulation(nTowns, nPop) # create initial population
    
    fitnesses = determineFitnesses(population, towns) # determine fitnesses 
    # of initial population
    
    return [towns, population, fitnesses]


def elitism(population, fitnessList, eliteRate):
    A = list(range(len(population))) # create range of population
    zipped = zip(fitnessList, A) # zip together pop range and fitnesses
    B = sorted(zipped) # sort by fitnesses
    topNum = int(np.round(eliteRate*len(population)))
    elites = []
    others = []
    for i in range(topNum):
        elites.append(population[B[i][1]])
    for j in range(len(population)-topNum):
        others.append(population[B[j][1]])
    return [elites, others]
 

def selection(others):
    # 
    selected = others
    return selected


def crossover(matingPool):
    # single point crossover
    newPop = []
    for i in range(int(len(matingPool)/2)):
        [parentA, parentB] = random.sample(matingPool,2) # pick parents
        matingPool.remove(parentA) # remove parents from mating pool
        matingPool.remove(parentB)
        crossPoint = random.randint(0, len(parentA)-1) # pick crossover point.
        
        # Crossover for child A
        childA = [len(parentA)+1]*len(parentA) # make child
        childA[crossPoint:] = parentB[crossPoint:] # add in section from parent
        childA[:crossPoint] = [gene for gene in parentA if gene not in childA]
        # list comprehension, find genes in parentA no already in child, and 
        # keep in original order, add them to the start of the child.
        
        # Repeat for child B
        childB = [len(parentB)+1]*len(parentB) # make child
        childB[crossPoint:] = parentA[crossPoint:] # add in section from parent
        childB[:crossPoint] = [gene for gene in parentB if gene not in childB]
        # list comprehension, find genes in parentA no already in child, and 
        # keep in original order, add them to the start of the child.        
                
        newPop.append(childA) # Add children to new population
        newPop.append(childB)
    
    return newPop

def mutation(population, mutationRate):
    newPop = population[:]
    for i in range(len(population)): # For each individual in pop
        if random.random() <= mutationRate: # decide if mutation occurs
            a = random.randint(0, len(population[i])-1) # pick where mutation occurs 1
            b = random.randint(0, len(population[i])-1) # pick second
            
            gene1 = population[i][a] # first city to swap
            gene2 = population[i][b] # second city to swap
            
            newPop[i][a] = gene2 # swap cities
            newPop[i][b] = gene1
        
    return newPop

def runGA(population, towns, eliteRate, mutationRate, iterationMax, 
          convergenceCriteria):
    # Run the algorithm
    bestIndividuals = []
    bestFitnesses = []
    globalBestFitness = [1e12]
    residual, iteration = 0,0
    while (residual > convergenceCriteria) + (iteration > iterationMax) == 0:
        fitnesses = determineFitnesses(population=population[:], towns=towns) # determine fitnesses
        
        [elites, others] = elitism(population=population[:], 
                                fitnessList=fitnesses, eliteRate=eliteRate) 
        # determine elites & others where others = population - elites
        
        bestIndividuals.append(elites[0]) # Add best current individual to list
        # Calculate and store the current best fitness for convergence.
        currentBestFitness = determineFitnesses(
                            population=[elites[0]], towns=towns)
        bestFitnesses.append(currentBestFitness)
        
        selected = selection(others=others[:]) # carry out selection on others

        crossed = crossover(matingPool=selected[:]) # carry out crossover on 
        # selected individuals.
        
        newPop = elites + crossed # newPopulation is the elite individuals 
        # plus the result of the crossover
        
        newPop = mutation(population=newPop[:], mutationRate=mutationRate) 
        # carry out mutation on the new population
        
        # compare the last X values of the best individuals, where X is 
        # controlled by convergenceCriteria. If the best fitnesses hasn't 
        # changed assume converged.
        if len(bestIndividuals)>convergenceCriteria:
            lastX = bestIndividuals[-convergenceCriteria:]
            lastXF = determineFitnesses(population=lastX[:], towns=towns)
            if np.sum(lastXF/lastXF[-1]) == convergenceCriteria:
                residual = 1
        if currentBestFitness < globalBestFitness:
            residual = 0
            globalBestFitness = currentBestFitness
        else:
            residual += 1
        
        population = newPop[:]
        iteration += 1
    
    finalFitnesses = determineFitnesses(population=population[:], towns=towns)
    return population, finalFitnesses, bestIndividuals, bestFitnesses

def plotCitiesRoute(towns, individual, colour, figNum, route):
    plt.figure(figNum)
    i = 0
    for town in towns:
        plt.scatter(town[0],town[1],s=50, c='r', marker='H')
        cityName = 'city%s'%i
        plt.annotate(cityName, town)
        i += 1
    plt.grid('on')
    plt.axis([0,100,0,100])
    if route == True:
        for i in range(len(individual)):
            fromTown = individual[i]
            if i == len(individual)-1:
                toTown = individual[0]
            else:
                toTown = individual[i+1]
            dx = towns[toTown][0]-towns[fromTown][0]
            dy = towns[toTown][1]-towns[fromTown][1]
            plt.arrow(towns[fromTown][0], towns[fromTown][1], dx, dy, color=colour)
    
    
    