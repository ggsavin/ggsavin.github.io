#include<vector>
#include<iostream>
#include<random>
#include<fstream>
#include<numeric>
#include<algorithm>
#include <ctime>
#include<math.h>
#include<curand_kernel.h>


__device__ const int CHROMOSOME_COUNT = 5;

__global__
void initChromosomes(int* chromosomes, int* weights, int numberOfItems, int capacity) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < CHROMOSOME_COUNT * numberOfItems; i += stride) {
        curandState_t* s = new curandState_t;
        curand_init(threadIdx.x, i, 0, s);
        chromosomes[i] = curand(s) % 2;
    }
    // maybe we do some reductions to see the sum
}

__global__
void evaluateChromosomes(int* chromosomes, int* scores, int* weights, int* values, int numberOfItems, int capacity) {
    __shared__ int w[5];
    __shared__ int v[5];

    // partial reduction to get score for each chromosome. POPULATION * NUMBER OF ITEMS

}

int main(int argc, char* argv[]) {
    
    std::vector<int> weights_tmp, values_tmp;
    int numberOfItems, capacityOfKnapsack;

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

    // Values 
    int population = 5;
    int generations = 2;


    int* scores; // population size
    int* chromosomes; // these are the genes in int population * numberOfItems

    cudaMallocManaged(&scores, population * sizeof(int));
    cudaMallocManaged(&chromosomes, population * numberOfItems * sizeof(int));

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    size_t threadsPerBlock = 1024;
    size_t numberOfBlocks = 32 * numberOfSMs;

    std::clock_t start = std::clock();
    // initChromosomes<<<numberOfBlocks, threadsPerBlock>>>(chromosomes, weights, numberOfItems, capacityOfKnapsack);
    for (int i = 1; i <= generations; i++) {

        if (i == 1) {
            // initKernel<<<numberOfBlocks, threadsPerBlock>>>(time(NULL));
            initChromosomes<<<numberOfBlocks, threadsPerBlock>>>(chromosomes, weights, numberOfItems, capacityOfKnapsack);
        }
        cudaDeviceSynchronize();

        std::cout << "Generation : " << i << "\n";

        for (int i = 0; i < population; i++) {
            std::cout << "Chromosome " << i << " : ";
            // std::cout << "Chromosome " << i << " score - " << scores[i] << " : ";
            for (int j = 0; j < numberOfItems; j++) {
                std::cout << chromosomes[i*numberOfItems + j];
            }
            std::cout << "\n";
        }

        std::cout << "\n";
    }

    double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<< "Duration for " << generations << " : " << duration <<'\n';

    cudaFree(chromosomes);
    cudaFree(scores);
}