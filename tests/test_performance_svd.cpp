#include <gtest/gtest.h>

#include "../src/algorithms/svd.h"
#include "helpers.h"
#include <chrono>

namespace {
using namespace LinearKit::Algorithm;
using Type = std::complex<long double>;
using LinearKit::Tests::RandomMatrixGenerator;

RandomMatrixGenerator<Type> gen(91348);

int64_t GetMeanTime(int32_t size, int32_t iterations) {
    int64_t total_ms = 0;

    for (auto i = 0; i < iterations; ++i) {
        auto matrix = gen.GetRandomMatrix(size, size);

        auto start = std::chrono::high_resolution_clock::now();
        auto [U, S, VT] = SVD(matrix);
        auto end = std::chrono::high_resolution_clock::now();

        total_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
    }
    return total_ms / iterations;
}

TEST(TEST_PERFORMANCE_SVD, Performance) {
    const int32_t iterations = 10;
    const int32_t min_size = 1;
    const int32_t max_size = 100;

    for (int32_t size = min_size; size <= max_size; ++size) {
        auto mean_time = GetMeanTime(size, iterations);
        std::cout << "[SVD Performance test] Matrix size: " << size
                  << ", mean time to compute: " << mean_time << " ms.\n";
    }
}
} // namespace
