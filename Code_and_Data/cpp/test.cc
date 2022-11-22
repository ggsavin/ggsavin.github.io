#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

struct Solution {
    double rank, x,y,z;
    void fitness() {
        double ans = (6 * x + -y + std::pow(z, 200)) - 25;
        rank = (ans == 0) ? 9999 : std::abs(1/ans);
    }
};


int main() {
    // Setup
    std::random_device device;
    std::uniform_real_distribution<double> unif(-100,100);
    std::vector<Solution> solutions;

    const int NUM = 100000;
    for (int i = 0; i < NUM; i++)
        solutions.push_back(Solution{
            0,
            unif(device),
            unif(device),
            unif(device)
        });
    while(true) {
        // Run fitness function
        for(auto& s: solutions) {s.fitness(); }

        // Sort solutions based on rank
        std::sort(solutions.begin(), solutions.end(), [](const auto& lhs, const auto& rhs){
            return lhs.rank > rhs.rank;
        });

        // Print top solutions
        //std::for_each(solutions.begin(), solutions.begin() + 10, [](const auto& s){
        //    std::cout << std::fixed << "Rank " << static_cast<int>(s.rank) << "\n x:" << s.x << " y:" << s.y << " z:" << s.z << " \n";
       // });

        // Selecting best candidates
        const int SAMPLE_SIZE = 1000;
        std::vector<Solution> sample;
        std::copy(solutions.begin(), solutions.begin() + SAMPLE_SIZE, std::back_inserter(sample));
        solutions.clear();

        // Mutate the top solutions by a %
        // Uusally people will mutate a small percentage of them
        std::uniform_real_distribution<double> m(0.99, 1.01);
        std::for_each(sample.begin(), sample.end(), [&](auto& s){
            s.x *= m(device);
            s.y *= m(device);
            s.z *= m(device);
        });

        // Crossover
        std::uniform_int_distribution<int> cross(0, SAMPLE_SIZE-1);
        for(int i = 0; i < NUM; i++) {
            solutions.push_back(Solution{
                0, sample[cross(device)].x, sample[cross(device)].y, sample[cross(device)].z
            });
        }
    };
}