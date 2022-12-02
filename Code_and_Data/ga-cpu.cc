/*
    SIMPLE MODEL WITH MORE CUDA FRIENDLY FUNCTIONS
    all function returns are void
*/
#include<vector>
#include<iostream>
#include<random>
#include<fstream>
#include<numeric>
#include<algorithm>
#include <ctime>
#include<math.h>

const int GENERATIONS = 1000;
const float ELITEISM = 0.1; 

struct chromosome {
    int score;
    int* genes;
};


// TODO: Consider initializing in a better than random way. Especially helpful for low capacity, high number of items
void initializeChromosomes(chromosome* chromosomes, int* weights, int numberOfItems, int capacity, int numberOfChromosomes) {

    std::random_device device;
    std::mt19937 generator{device()};
    std::bernoulli_distribution bd{0.5};

    for (int i = 0; i < numberOfChromosomes; i++) {
        std::size_t size_of_items = numberOfItems * sizeof(int);
        chromosomes[i].score = 0;
        chromosomes[i].genes = (int*)malloc(size_of_items);
        
        for (int j = 0, weight = 0; j < numberOfItems; j++) {
            int bit = bd(generator);
            if (bit == 1 && weight + weights[j] > capacity) {
                chromosomes[i].genes[j] = 0;
            } else {
                weight += bit * weights[j];
                chromosomes[i].genes[j] = bit;
            }
        }
    }
}

void evaluateChromosomes(chromosome* chromosomes, int* bestScore, float* avg, int* totalScore, int* weights, int* values, int numberOfItems, int capacity, int numberOfChromosomes) {
    float average = 0.0;
    int best = 0;
    int total = 0;

    for (int i = 0; i < numberOfChromosomes; i++) {
        int weight = 0;
        int score = 0;
        for (int j = 0; j < numberOfItems; j++) {
            if(chromosomes[i].genes[j] == 1) {
                weight += weights[j];
                score += values[j];
            }
        }
        // score penalty for too much weight
        if (weight > capacity) {
           // going to set it to just 1 to ensure that roulette selection works further downstream.
           score = 1;
        }
        chromosomes[i].score = score;
        average += (score / (float)numberOfChromosomes);
        total += score;
        if (score > best) {
            best = score;
        }
    }

    *bestScore = best;
    *avg = average;
    *totalScore = total;
}

// TODO : Make sure IDs not the same
void rouletteSelection(chromosome* chromosomes, int* parent1, int* parent2, int totalScore, int numberOfChromsomes) {
    if (totalScore == 0) {
        *parent1 = rand() % numberOfChromsomes;
        *parent2 = rand() % numberOfChromsomes;
    }

    int partialScore = totalScore;

    for (int i = 0, r = rand() % totalScore, sum= 0; i < numberOfChromsomes; i++) {
        sum += chromosomes[i].score;
        if (sum >= r) {
            *parent1 = i;
            partialScore -= chromosomes[i].score;
            break;
        }
    }

    for (int i = 0, r = rand() % partialScore, sum = 0; i < numberOfChromsomes; i++) {
        if (i != *parent1) {
            sum += chromosomes[i].score;
            if (sum >= r) {
                *parent2 = i;
                break;
            }
        }                
    }
}


void mutateChromosomes(chromosome* chromosomes, int numberOfItems, int elites, int numberOfChromsomes) {
    for (int i = elites; i < numberOfChromsomes; i++) {
        for (int j = 0; j < numberOfItems; j++) {
            int r = rand() % 1000;
            if (r < 1) {
                chromosomes[i].genes[j] = !chromosomes[i].genes[j];
            }
        }
    }
}

void crossover(chromosome* parent1, chromosome* parent2, chromosome* child1, chromosome* child2, int numberOfItems) {
    int cutPoint = rand() % numberOfItems;
    for (int i = 0; i < numberOfItems; i++) {
        if (i < cutPoint) {
            child1->genes[i] = parent1->genes[i];
            child2->genes[i] = parent2->genes[i];            
        } else {
            child1->genes[i] = parent2->genes[i];
            child2->genes[i] = parent1->genes[i];   
        }
    }
}

int cmpfunc(const void* a, const void* b) {
    chromosome* ch1 = (chromosome*)a;
    chromosome* ch2 = (chromosome*)b;

    return (ch2->score - ch1->score);
}

void reproduceChromosomes(chromosome* chromosomes, int totalScore, int numberOfItems, int numberOfChromosomes) {

    // sort parent chromosomes based on rank 
    qsort(chromosomes, numberOfChromosomes, sizeof(chromosome), cmpfunc);

    // offsprings
    chromosome* offspring = (chromosome*)malloc(numberOfChromosomes * sizeof(chromosome));
    
    for (int i = 0; i < numberOfChromosomes; i++) {
        std::size_t size_of_items = numberOfItems * sizeof(int);
        offspring[i].score = 0;
        offspring[i].genes = (int*)malloc(size_of_items);
    }
    
    // setup offsprings w/ eliteism
    int eliteChromosomeCount = ceil(ELITEISM*numberOfChromosomes);
    for (int i = 0; i < eliteChromosomeCount; i++) {
        for (int j = 0; j < numberOfItems; j++) {
            offspring[i].genes[j] = chromosomes[i].genes[j];
        }
    }

    // go through the remaining genes, select parents, perform crossover, and store parents
    int remainingOffspring = numberOfChromosomes - eliteChromosomeCount;
    int offspring1IDx = eliteChromosomeCount;
    int offspring2IDx = eliteChromosomeCount + 1;

    while (remainingOffspring > 0) {
         int  parent1Idx;
         int  parent2Idx;
         rouletteSelection(chromosomes, &parent1Idx, &parent2Idx, totalScore, numberOfChromosomes);

         if (offspring2IDx < numberOfChromosomes) {
            crossover(&chromosomes[parent1Idx], &chromosomes[parent2Idx], &offspring[offspring1IDx], &offspring[offspring2IDx], numberOfItems);
         } else {
            // only keep the first child to finish off the array size
            int cutPoint = rand() % numberOfItems;
            for (int i = 0; i < numberOfItems; i++) {
                if (i < cutPoint) {
                    offspring[offspring1IDx].genes[i] = chromosomes[parent1Idx].genes[i];
                } else {
                    offspring[offspring1IDx].genes[i] = chromosomes[parent2Idx].genes[i];
                }
            }
         }
         remainingOffspring -=2;
         offspring1IDx += 2;
         offspring2IDx += 2;
    }

    // possibly mutate
    mutateChromosomes(offspring, numberOfItems, eliteChromosomeCount, numberOfChromosomes);

    // make chromosomes into offspring
    for (int i = 0; i < numberOfChromosomes; i++) {
        for(int j = 0; j < numberOfItems; j++) {
            chromosomes[i].genes[j] = offspring[i].genes[j];
        }
    }

    // free memory of the offspring
    for (int i = 0; i < numberOfChromosomes; i++) {
        free(offspring[i].genes);
    }
    free(offspring);
}

int main(int argc, char* argv[]) {

    std::vector<int> weights_tmp, values_tmp;
    int numberOfItems, capacityOfKnapsack;


    // read or get the weights, values and max_weight for the problem somehow
    std::fstream input_data;
    input_data.open("data/test_1000.txt", std::ios_base::in);
    if (!input_data) {
        std::cout << " File does not exist! " << std::endl;
    } else {
        int weight, value;
        input_data >> numberOfItems;
        input_data >> capacityOfKnapsack;

        for(int i = 0; i < numberOfItems; i++) {
            input_data >> weight >> value;
            weights_tmp.push_back(weight);
            values_tmp.push_back(value);
        }
        input_data.close();
    }


    // setting up weights and values
    std::size_t size_of_items = numberOfItems * sizeof(int);
    int* weights;
    int* values;
    weights = (int*)malloc(size_of_items); //TODO: free this
    values = (int*)malloc(size_of_items); //TODO: free this

    for(int i = 0; i < numberOfItems; i ++) {
        weights[i] = weights_tmp[i];
        values[i] = values_tmp[i];
    }

    std::vector<double> timings;

    for(int num_chromosomes = 50; num_chromosomes <= 1000; num_chromosomes += 50) {

        // we have a NUM_OF_CHROMOSOMES amount of chromosome structs, that each will have genes encoded
        // in binary, with 1 representing the presence of an item, 0 the lack of item.
        std::size_t size_of_chromosomes = num_chromosomes * sizeof(chromosome);
        chromosome* chromosomes = (chromosome*)malloc(size_of_chromosomes);

        // starting TIMER
        std::clock_t start = std::clock();

        int best = 0, total = 0;
        float average = 0.0;

        for (int i = 1; i <= GENERATIONS; i++) {
            if (i == 1)
                initializeChromosomes(chromosomes, weights, numberOfItems, capacityOfKnapsack, num_chromosomes);
            else {
                reproduceChromosomes(chromosomes, total, numberOfItems, num_chromosomes);
            }
            evaluateChromosomes(chromosomes, &best, &average, &total, weights, values, numberOfItems, capacityOfKnapsack, num_chromosomes);

            // if (i % 5 == 0)
            //     std::cout << "Generation " << i << " || Top : " << best << " || Avg : " << average << std::endl;
        }
        
        // std::cout << " Top Value after " << GENERATIONS << " generations : " << best << "\n";

        // Freeing some set values
        for (int i = 0; i < num_chromosomes; i++) {
            free(chromosomes[i].genes);
        }
        free(chromosomes);


        // stopping our timer
        double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        std::cout<< "Duration for " << GENERATIONS << " : " << duration <<'\n';
        timings.push_back(duration);
    }

    free(weights);
    free(values);
    // Write the durations to file
    std::ofstream fout("reg_results.txt");
    for(auto const& x : timings)
        fout << x << '\n';

    return 0;
}