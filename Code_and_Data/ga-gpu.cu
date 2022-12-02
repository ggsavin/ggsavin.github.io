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
#include<curand_kernel.h>
#include<cublas_v2.h>

const int GENERATIONS = 1000;
// const int NUM_OF_CHROMOSOMES = 1024;
// const float ELITEISM = 0.1; 

__device__ int CHROMOSOME_COUNT;
__device__ curandState_t* states[1024];

struct chromosome {
    int score;
    int weight;
    int* genes;
};

// REDUCER CODE

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(EXIT_FAILURE); 
    }
}

__global__ 
void initKernel(int seed) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < CHROMOSOME_COUNT; i += stride) {
        curandState_t* s = new curandState_t;
        if (s != 0) {
            curand_init(seed, i, 0, s);
        }
        states[i] = s;
    }
}

__global__
void initializeChromosomes(chromosome* chromosomes, int* weights, int numberOfItems, int capacity) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < CHROMOSOME_COUNT; i += stride) {
        chromosomes[i].score = 0;
        chromosomes[i].weight = 0;
        for (int j = 0, weight = 0; j < numberOfItems; j++) {
            int bit = curand(states[i]) % 2;
            if (bit == 1 && weight + weights[j] > capacity) {
                chromosomes[i].genes[j] = 0;
            } else {
                weight += bit * weights[j];
                chromosomes[i].genes[j] = bit;
            }
        }
    }
}

__global__
void evaluteChromosomesGeneLevel(chromosome* chromosomes, int* scores, int* weights, int* values, int* gScores, int* gWeights, int numberOfItems, int capacity, int population) {
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int chromosomeIdx = index / numberOfItems;
    int innerIndex = index % numberOfItems;

    // lets just do an atomic add for each gene level
    if (index < numberOfItems * population) {
        gWeights[index] = chromosomes[chromosomeIdx].genes[innerIndex] * weights[innerIndex];
        gScores[index] = chromosomes[chromosomeIdx].genes[innerIndex] * values[innerIndex];
    }
    __syncthreads();

    // now we should go ahead and add each one to 
    if (index < numberOfItems * population) {
        atomicAdd(&chromosomes[chromosomeIdx].score, gScores[index]);
        atomicAdd(&chromosomes[chromosomeIdx].weight, gWeights[index]);
    }
    __syncthreads();

    // now we have a score and weight. Go through each chromosome, and change up score based on weight. We waste all the threads in dead blocks now
    if (index < population) {
        if (chromosomes[index].weight > capacity)
            chromosomes[index].score= 1;
    }
}

__global__
void evaluateChromosomes(chromosome* chromosomes, int* weights, int* values, int numberOfItems, int capacity) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < CHROMOSOME_COUNT; i += stride) {
        int weight=0;
        int score=0;

        // cublasSdot();

        for (int j = 0; j < numberOfItems; j++) {
            if(chromosomes[i].genes[j] == 1) {
                weight += weights[j];
                score += values[j];
            }
        }
        if (weight > capacity) {
           score = 1;
        } else if (score == 0) {
            // if all genes are 0 somehow, still add 1 to it;
            score = 1;
        }
        chromosomes[i].score = score;
    }
}

__global__
void pullScores(chromosome* chromosomes, int* scores) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < CHROMOSOME_COUNT; i += stride) {
        scores[i] = chromosomes[i].score;
    }
}

__device__
void rouletteSelection(int total, int* scores, int* id1, int* id2, curandState_t* state) {
    int partialScore = total;
    for (int i = 0, r = curand(state) % partialScore, sum= 0; i < CHROMOSOME_COUNT; i++) {
        sum += scores[i];
        if (sum >= r) {
            *id1 = i;
            partialScore -= scores[i];
            break;
        }
    }

    for (int i = 0, r = curand(state) % partialScore, sum = 0; i < CHROMOSOME_COUNT; i++) {
        if (i != *id1) {
            sum += scores[i];
            if (sum >= r) {
                *id2 = i;
                break;
            }
        }                
    }
};

__device__
void crossover(int* parent1genes, int* parent2genes, int* offspringGenes, curandState_t* state, int numberOfItems) {
    int cutPoint = curand(state) % numberOfItems;
    int whichOffspring = curand(state) % 2;

    // add crossover chance all together. might just keep higher score parent

    for (int i = 0; i < numberOfItems; i++) {
        if (i < cutPoint) {
            offspringGenes[i] = whichOffspring ? parent2genes[i] : parent1genes[i];          
        } else {
            offspringGenes[i] = whichOffspring ? parent1genes[i] : parent2genes[i];  
        }
    }
}

__global__ 
void mutateChromosomes(chromosome* chromosomes, int numberOfItems, int population) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int chromosomeIdx = index / numberOfItems;
    int innerIndex = index % numberOfItems;

    // we are actually going to run the thread on genes of the chromosomes directly, all at once
    if (index < numberOfItems * population) {
        int r = curand(states[chromosomeIdx]) % 1000;
        if (r < 1)
            chromosomes[chromosomeIdx].genes[innerIndex] = !chromosomes[chromosomeIdx].genes[innerIndex];
    }
}


__global__
void GPUreproduceChromosomes(chromosome* chromosomes, chromosome* offspring, int* scores, int* total, int numberOfItems) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // each thread will grab 2 parents randomly out of chromosomes, based on scores and total, then cross them over
    for (int i = index; i < CHROMOSOME_COUNT; i+= stride) {
        
        int p1dx;
        int p2dx;

         // 2 expensive modulo operations in the roulette selection
        rouletteSelection(*total, scores, &p1dx, &p2dx, states[i]);

        // probably going to have terrible memory bank hits here as threads may be reading from the most high scoring
        // parents often TODO: maybe do this a percentage of the time, if not just pick the highest scoring parent
        crossover(chromosomes[p1dx].genes, chromosomes[p2dx].genes, offspring[i].genes, states[i], numberOfItems);

        // mutation happens afterwards
    }


    __syncthreads();
    for (int i = index; i < CHROMOSOME_COUNT; i+= stride) {
        chromosomes[i].score = offspring[i].score;
    }
}

__global__
void copyOffspringIntoChromosomes(chromosome* chromosomes, chromosome* offspring,  int numberOfItems, int population) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int chromosomeIdx = index / numberOfItems;
    int innerIndex = index % numberOfItems;

    // threads are running on the genes themselves
    if (index < numberOfItems * population) {
            chromosomes[chromosomeIdx].genes[innerIndex] = offspring[chromosomeIdx].genes[innerIndex];
    }
}

__device__ void warpReduce(volatile int* idata, int tid) {
    idata[tid] += idata[tid + 32];
    idata[tid] += idata[tid + 16];
    idata[tid] += idata[tid + 8];
    idata[tid] += idata[tid + 4];
    idata[tid] += idata[tid + 2];
    idata[tid] += idata[tid + 1];
}

// Use this for reducer, maybe avoid needing a total
__global__ void sumReducer(int* out, chromosome* chromosomes, int N) {
    __shared__ int partial_sum[1024];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (tid == 0) {
        *out = 0;
    }
    __syncthreads();

    int initSum = 0;
    for (int i = tid; i < N; i += stride) {
        initSum += chromosomes[i].score;
    }
    // also probably need to ensure these threads fit 
    partial_sum[threadIdx.x] = initSum;
    __syncthreads();

    // Doing a reduction over
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) warpReduce(partial_sum, threadIdx.x);
    if (threadIdx.x == 0) atomicAdd(out, partial_sum[0]);
}

int main() {

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

    // weights = (int*)malloc(size_of_items); 
    // values = (int*)malloc(size_of_items); 
    cudaMallocManaged(&weights, size_of_items);
    cudaMallocManaged(&values, size_of_items);

    for(int i = 0; i < numberOfItems; i ++) {
        weights[i] = weights_tmp[i];
        values[i] = values_tmp[i];
    }

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    size_t threadsPerBlock = 1024;
    size_t numberOfBlocks = 32 * numberOfSMs;

    std::vector<double> timings;

    for(int num_chromosomes = 32; num_chromosomes <= 1024; num_chromosomes *= 2) {

        int* scores;
        cudaMallocManaged(&scores, num_chromosomes * sizeof(int));

        int* gWeights;
        int* gScores;
        cudaMallocManaged(&gWeights, numberOfItems * num_chromosomes * sizeof(int));
        cudaMallocManaged(&gScores, numberOfItems* num_chromosomes * sizeof(int));

        // we have a NUM_OF_CHROMOSOMES amount of chromosome structs, that each will have genes encoded
        // in binary, with 1 representing the presence of an item, 0 the lack of item.
        std::size_t size_of_chromosomes = num_chromosomes * sizeof(chromosome);
        // chromosome* chromosomes;= (chromosome*)malloc(size_of_chromosomes);
        chromosome* chromosomes;
        chromosome* offspring;
        cudaMallocManaged(&chromosomes, size_of_chromosomes);
        cudaMallocManaged(&offspring, size_of_chromosomes);

        // setting up memory for the genes themselves
        for (int i = 0; i < num_chromosomes; i++) {
            cudaMallocManaged(&chromosomes[i].genes, numberOfItems * sizeof(int));
            cudaMallocManaged(&offspring[i].genes, numberOfItems * sizeof(int));
        }

        cudaMemcpyToSymbol(CHROMOSOME_COUNT, &num_chromosomes, sizeof(int), 0, cudaMemcpyHostToDevice);

        // starting TIMER
        // std::clock_t start = std::clock();

        // int* best; 
        // int* total;
        // float* average;

        // int best = 0, total = 0;
        // float average = 0.0;
        int* total;
        cudaMallocManaged(&total, sizeof(int));

        // int minGrid, minBlockSize;
        // cudaOccupancyMaxPotentialBlockSize(&minGrid, &minBlockSize, initKernel);
        // std::cout << "InitKernel : MinGrid - " << minGrid << " | minBlockSize - " << minBlockSize << "\n";
        // cudaOccupancyMaxPotentialBlockSize(&minGrid, &minBlockSize, initializeChromosomes);
        // std::cout << "Initializer Chromosomes : MinGrid - " << minGrid << " | minBlockSize - " << minBlockSize << "\n";
        // cudaOccupancyMaxPotentialBlockSize(&minGrid, &minBlockSize, evaluateChromosomes);
        // std::cout << "EvaluateChromosomes : MinGrid - " << minGrid << " | minBlockSize - " << minBlockSize << "\n";
        // cudaOccupancyMaxPotentialBlockSize(&minGrid, &minBlockSize, pullScores);
        // std::cout << "PullScores : MinGrid - " << minGrid << " | minBlockSize - " << minBlockSize << "\n";
        // cudaOccupancyMaxPotentialBlockSize(&minGrid, &minBlockSize, GPUreproduceChromosomes);
        // std::cout << "PullScores : MinGrid - " << minGrid << " | minBlockSize - " << minBlockSize << "\n";

        // How many blocks, and how many threads per block?
        int BlocksGeneSize = ((num_chromosomes * numberOfItems) + threadsPerBlock) / threadsPerBlock;
        std::clock_t start = std::clock();

        for (int i = 1; i <= GENERATIONS; i++) {
            if (i == 1) {
                initKernel<<<1, threadsPerBlock>>>(time(NULL));
                initializeChromosomes<<<1, threadsPerBlock>>>(chromosomes, weights, numberOfItems, capacityOfKnapsack);
            } else {
                GPUreproduceChromosomes<<<1, threadsPerBlock>>>(chromosomes, offspring, scores, total, numberOfItems);
                copyOffspringIntoChromosomes<<<BlocksGeneSize,threadsPerBlock>>>(chromosomes, offspring, numberOfItems, num_chromosomes);
                mutateChromosomes<<<BlocksGeneSize,threadsPerBlock>>>(chromosomes, numberOfItems, num_chromosomes);
            }
            //evaluateChromosomes<<<1, threadsPerBlock>>>(chromosomes, weights, values, numberOfItems, capacityOfKnapsack);
            evaluteChromosomesGeneLevel<<<BlocksGeneSize, threadsPerBlock>>>(chromosomes, scores, weights, values, gScores, gWeights, numberOfItems, capacityOfKnapsack, num_chromosomes);
            pullScores<<<1, threadsPerBlock>>>(chromosomes, scores);
            sumReducer<<<1, threadsPerBlock>>>(total, chromosomes, num_chromosomes);
            // cudaDeviceSynchronize();
            // total  = GPUReduction<1024>(scores, num_chromosomes);

            // for (int i = 0; i < NUM_OF_CHROMOSOMES; i++) {
            //     // std::cout << "Chromosome " << i << " score - " << chromosomes[i].score << " : ";
            //     std::cout << "Chromosome " << i << " score - " << scores[i] << " : ";
            //     for (int j = 0; j < numberOfItems; j++) {
            //         std::cout << chromosomes[i].genes[j];
            //     }
            //     std::cout << "\n";
            // }
            // if (i % 10 == 0)
            //     std::cout << "Generation " << i << " || Total : " << total <<  "\n";
            //     //std::cout << "Generation " << i << " || Top : " << best << " || Avg : " << average << "\n";
        }
        cudaDeviceSynchronize();

        // std::cout << " Top Value after " << GENERATIONS << " generations : " << scores[0] << "\n";

        // Freeing some set values
        double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        timings.push_back(duration);
        std::cout<< "Duration for " << GENERATIONS << " : " << duration <<'\n';

        cudaFree(scores);
        cudaFree(gWeights);
        cudaFree(gScores);
        for (int i = 0; i < num_chromosomes; i++) {
            cudaFree(chromosomes[i].genes);
            cudaFree(offspring[i].genes);
        }
        cudaFree(chromosomes);
        cudaFree(offspring);
        cudaFree(total);
        // // stopping our timer
        // double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        // timings.push_back(duration);
        // std::cout<< "Duration for " << GENERATIONS << " : " << duration <<'\n';
    }

    cudaFree(weights);
    cudaFree(values);

    // std::ofstream fout("gpu_results.txt");
    // for(auto const& x : timings)
    //     fout << x << '\n';

    return 0;
}