#%% Preamble

import numpy as np
import matplotlib.pyplot as plt
import TSP 

#%% Initialise
# This section initialises the problem including number of towns in the wolrd, 
# total GA population size, limits for location of the towns, and whether or
# not to randomly determine the town locations (i.e RNG=False). Setting RNG to 
# False allows for comparisons of the efficiency of GA parameters. This section
# also plots an initial example route.

total_number_towns = 10
total_GA_population_size = 100
town_location_limit_lowerX = 0
town_location_limit_upperX = 100
town_location_limit_lowerY = 0
town_location_limit_upperY = 100

[towns, pop, initialFitnesses] = TSP.initialise(nTowns = total_number_towns, 
                                                nPop = total_GA_population_size, 
                                                xLowerBound = town_location_limit_lowerX,
                                                xUpperBound = town_location_limit_upperX, 
                                                yLowerBound = town_location_limit_lowerY, 
                                                yUpperBound = town_location_limit_upperY, 
                                                RNG = False)

TSP.plotCitiesRoute(towns = towns, 
                    individual = pop[0], 
                    colour = 'b', 
                    figNum = 0,
                    route = False)

#%% RUN
# This section runs the GA based on the towns, population, and fitnesses of 
# the intialisation function and based on the input elitism rate (%), 
# mutation rate (%), maximum number of generations, and the stagnation 
# criteria. 

elitism_rate = 0.1
mutation_rate = 0.01
maximum_number_generations = 10000
stagnation_criteria = 500 # This represents the number of generations where
# the best fitness of the population did not change in order to assume the 
# algorithm has converged. 

[finalPop, finalFitnesses, bestIdvs, bestFits] = TSP.runGA(population = pop[:], 
                                                           towns = towns, 
                                                           eliteRate = elitism_rate, 
                                                           mutationRate = mutation_rate, 
                                                           iterationMax = maximum_number_generations, 
                                                           convergenceCriteria = stagnation_criteria)

#%% OUTPUTS
# This section prints to console the total number of 
# generations simulated, the initial best fitness, the best fitness of the 
# final population, and the best global fitness found, and then plots the best
# individual of the final population. This section also plots 
# the convergence of the algorithm based on the best fitness of each 
# population against the generation. 

print('\n- Total number of generations = %s' %len(bestIdvs))
print('- Best fitness of initial population = %s' %np.min(initialFitnesses))
print('- Best fitness of final population = %s' %np.min(finalFitnesses))
print('- Best fitness found during algorithm = %s' %np.min(bestFits))


TSP.plotCitiesRoute(towns = towns, 
                    individual = bestIdvs[-1], 
                    colour = 'r', 
                    figNum = 1,
                    route = True)

plt.figure(2)
plt.plot(bestFits)
plt.grid('on')
plt.xlabel('Generation number')
plt.ylabel('Best fitness per generation')
