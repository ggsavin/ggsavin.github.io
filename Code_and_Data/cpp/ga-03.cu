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

const int GENERATIONS = 2;
const int NUM_OF_CHROMOSOMES = 5;
const float ELITEISM = 0.1; 

__device__ const int CHROMOSOME_COUNT = 5;
__device__ curandState_t* states[NUM_OF_CHROMOSOMES];

struct chromosome {
    int score;
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

template <size_t blockSize, typename T>
__device__ void warpReduce(volatile T *sdata, size_t tid)
{
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

template <size_t blockSize, typename T>
__global__ void reduceCUDA(T* g_idata, T* g_odata, size_t n)
{
    __shared__ T sdata[blockSize];

    size_t tid = threadIdx.x;
    //size_t i = blockIdx.x*(blockSize*2) + tid;
    //size_t gridSize = blockSize*2*gridDim.x;
    size_t i = blockIdx.x*(blockSize) + tid;
    size_t gridSize = blockSize*gridDim.x;
    sdata[tid] = 0;

    while (i < n) { sdata[tid] += g_idata[i]; i += gridSize; }
    //while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >=  512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >=  256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >=  128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
template<size_t blockSize, typename T>
T GPUReduction(T* dA, size_t N)
{
    T tot = 0.;
    size_t n = N;
    size_t blocksPerGrid = std::ceil((1.*n) / blockSize);

    T* tmp;
    cudaMalloc(&tmp, sizeof(T) * blocksPerGrid); checkCUDAError("Error allocating tmp [GPUReduction]");

    T* from = dA;

    do
    {
        blocksPerGrid   = std::ceil((1.*n) / blockSize);
        reduceCUDA<blockSize><<<blocksPerGrid, blockSize>>>(from, tmp, n);
        from = tmp;
        n = blocksPerGrid;
    } while (n > blockSize);

    if (n > 1)
        reduceCUDA<blockSize><<<1, blockSize>>>(tmp, tmp, n);

    cudaDeviceSynchronize();
    checkCUDAError("Error launching kernel [GPUReduction]");

    cudaMemcpy(&tot, tmp, sizeof(T), cudaMemcpyDeviceToHost); checkCUDAError("Error copying result [GPUReduction]");
    cudaFree(tmp);
    return tot;
}
// END OF  REDUCER CODE

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
void evaluateChromosomes(chromosome* chromosomes, int* weights, int* values, int numberOfItems, int capacity) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < CHROMOSOME_COUNT; i += stride) {
        int weight=0;
        int score=0;
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

__device__
void mutateChromosome(int* genes, curandState_t* state, int numberOfItems) {
    for (int j = 0; j < numberOfItems; j++) {
        int r = curand(state) % 1000;
        if (r < 1) {
            genes[j] = !genes[j];
        }
    }
}

__global__
void GPUreproduceChromosomes(chromosome* chromosomes, chromosome* offspring, int* scores, int total, int numberOfItems) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // each thread will grab 2 parents randomly out of chromosomes, based on scores and total, then cross them over
    for (int i = index; i < CHROMOSOME_COUNT; i+= stride) {
        
        int p1dx;
        int p2dx;

         // 2 expensive modulo operations in the roulette selection
        rouletteSelection(total, scores, &p1dx, &p2dx, states[i]);

        // probably going to have terrible memory bank hits here as threads may be reading from the most high scoring
        // parents often TODO: maybe do this a percentage of the time, if not just pick the highest scoring parent
        crossover(chromosomes[p1dx].genes, chromosomes[p2dx].genes, offspring[i].genes, states[i], numberOfItems);

        // possibly mutate
        // also multiple expensive hits with % operators
        mutateChromosome(offspring[i].genes, states[i], numberOfItems);
    }

    __syncthreads();
    for (int i = index; i < CHROMOSOME_COUNT; i+= stride) {
        chromosomes[i].score = offspring[i].score;

        for (int j = 0; j < numberOfItems; j++) {
            chromosomes[i].genes[j] = offspring[i].genes[j];
        }
    }
}


void mutateChromosomes(chromosome* chromosomes, int numberOfItems, int elites) {
    for (int i = elites; i < NUM_OF_CHROMOSOMES; i++) {
        for (int j = 0; j < numberOfItems; j++) {
            int r = rand() % 1000;
            if (r < 1) {
                chromosomes[i].genes[j] = !chromosomes[i].genes[j];
            }
        }
    }
}
int cmpfunc(const void* a, const void* b) {
    chromosome* ch1 = (chromosome*)a;
    chromosome* ch2 = (chromosome*)b;

    return (ch2->score - ch1->score);
}

int main() {

    std::vector<int> weights_tmp, values_tmp;
    int numberOfItems, capacityOfKnapsack;

    // read or get the weights, values and max_weight for the problem somehow
    std::fstream input_data;
    input_data.open("data/ga-input-test.txt", std::ios_base::in);
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

    int* scores;
    cudaMallocManaged(&scores, NUM_OF_CHROMOSOMES * sizeof(int));

    // we have a NUM_OF_CHROMOSOMES amount of chromosome structs, that each will have genes encoded
    // in binary, with 1 representing the presence of an item, 0 the lack of item.
    std::size_t size_of_chromosomes = NUM_OF_CHROMOSOMES * sizeof(chromosome);
    // chromosome* chromosomes;= (chromosome*)malloc(size_of_chromosomes);
    chromosome* chromosomes;
    chromosome* offspring;
    cudaMallocManaged(&chromosomes, size_of_chromosomes);
    cudaMallocManaged(&offspring, size_of_chromosomes);

    // setting up memory for the genes themselves
    for (int i = 0; i < NUM_OF_CHROMOSOMES; i++) {
        cudaMallocManaged(&chromosomes[i].genes, numberOfItems * sizeof(int));
        cudaMallocManaged(&offspring[i].genes, numberOfItems * sizeof(int));
    }

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    size_t threadsPerBlock = 256;
    size_t numberOfBlocks = 32 * numberOfSMs;

    // starting TIMER
    std::clock_t start = std::clock();

    // int* best; 
    // int* total;
    // float* average;

    int best = 0, total = 0;
    float average = 0.0;

    int minGrid, minBlockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &minBlockSize, initKernel);
    std::cout << "InitKernel : MinGrid - " << minGrid << " | minBlockSize - " << minBlockSize << "\n";
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &minBlockSize, initializeChromosomes);
    std::cout << "Initializer Chromosomes : MinGrid - " << minGrid << " | minBlockSize - " << minBlockSize << "\n";
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &minBlockSize, evaluateChromosomes);
    std::cout << "EvaluateChromosomes : MinGrid - " << minGrid << " | minBlockSize - " << minBlockSize << "\n";
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &minBlockSize, pullScores);
    std::cout << "PullScores : MinGrid - " << minGrid << " | minBlockSize - " << minBlockSize << "\n";
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &minBlockSize, GPUreproduceChromosomes);
    std::cout << "PullScores : MinGrid - " << minGrid << " | minBlockSize - " << minBlockSize << "\n";

    for (int i = 1; i <= GENERATIONS; i++) {
        if (i == 1) {
            initKernel<<<numberOfBlocks, threadsPerBlock>>>(time(NULL));
            initializeChromosomes<<<numberOfBlocks, threadsPerBlock>>>(chromosomes, weights, numberOfItems, capacityOfKnapsack);
            cudaDeviceSynchronize();
        } else {
            GPUreproduceChromosomes<<<numberOfBlocks, threadsPerBlock>>>(chromosomes, offspring, scores, total, numberOfItems);
            cudaDeviceSynchronize();
        }
        evaluateChromosomes<<<numberOfBlocks, threadsPerBlock>>>(chromosomes, weights, values, numberOfItems, capacityOfKnapsack);
        pullScores<<<numberOfBlocks, threadsPerBlock>>>(chromosomes, scores);
        cudaDeviceSynchronize();
        total  = GPUReduction<1024>(scores, NUM_OF_CHROMOSOMES);
        // reduce<<<numberOfBlocks, threadsPerBlock>>>(chromosomes, sumTotal);
         // int tot  = GPUReduction<1024>(chromosomes, NUM_OF_CHROMOSOMES);
        cudaDeviceSynchronize();

        for (int i = 0; i < NUM_OF_CHROMOSOMES; i++) {
            // std::cout << "Chromosome " << i << " score - " << chromosomes[i].score << " : ";
            std::cout << "Chromosome " << i << " score - " << scores[i] << " : ";
            for (int j = 0; j < numberOfItems; j++) {
                std::cout << chromosomes[i].genes[j];
            }
            std::cout << "\n";
        }
        if (i % 10 == 0)
            std::cout << "Generation " << i << " || Total : " << total <<  "\n";
            //std::cout << "Generation " << i << " || Top : " << best << " || Avg : " << average << "\n";
    }
    

    // Freeing some set values
    cudaFree(weights);
    cudaFree(values);
    cudaFree(scores);
    for (int i = 0; i < NUM_OF_CHROMOSOMES; i++) {
        cudaFree(chromosomes[i].genes);
        cudaFree(offspring[i].genes);
    }
    cudaFree(chromosomes);
    cudaFree(offspring);


    // stopping our timer
    double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<< "Duration for " << GENERATIONS << " : " << duration <<'\n';

    return 0;
}