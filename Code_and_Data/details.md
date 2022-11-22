Initial implementation
----
1/0 will represent an item picked or not picked out of a total of n items, represented as a string of 1's and 0's

. The most used selection methods, are
roulette-wheel, rank selection, steady-state selection

1. Start: Randomly generate a population of N chromosomes. 
2. Fitness: Calculate the fitness of all chromosomes.
3 Create a new population:
a. Selection: According to the selection method select 2 chromosomes
from the population. 
b. Crossover: Perform crossover on the 2 chromosomes selected. 
c. Mutation: Perform mutation on the chromosomes obtained.
4. Replace: Replace the current population with the new population. 
5. Test: Test whether the end condition is satisfied. If so, stop. If not, return the
best solution in current population and go to Step 2. 
Each iteration of this process is called generation. 

Roulette wheel where each individual is represented by a space proportional to their fitness (evaluation/avg. eval)
^ then start picking based on stochastic sampling with replacement

A selection process that will more closely match the expected fittness values is "remainder
stochastic sampling" For each string i where fi/f 1.0, the int value of this fraction is the number of times it is selected. So 1.36, is 1 copy + a 0.36 change of placing a second copy

Can be done doing *remainder stochastic sampling* -> *stochastic universal sampling*


