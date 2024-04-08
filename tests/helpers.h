#pragma once

#include "../types/matrix.h"

#include <random>

namespace matrix_lib::tests {
template <typename T>
class RandomMatrixGenerator {
    using IntDistribution = std::uniform_int_distribution<int32_t>;

    struct MatrixPair {
        Matrix<T> left;
        Matrix<T> right;
    };

public:
    explicit RandomMatrixGenerator(int32_t seed, int32_t matrix_min_size = 1,
                                   int32_t matrix_max_size = 100,
                                   int32_t value_from = -100,
                                   int32_t value_to = 100)
        : rng_(seed), rd_number_(value_from, value_to),
          rd_matrix_size_(matrix_min_size, matrix_max_size) {
    }

    int32_t GetRandomValue() {
        return rd_number_(rng_);
    }

    int32_t GetRandomMatrixSize() {
        return rd_matrix_size_(rng_);
    }

    Matrix<T> GetRandomMatrix(int32_t row, int32_t col) {
        Matrix<T> result(row, col);
        result.ApplyToEach([&](T &val) { val = GetRandomValue(); });
        return result;
    }

    MatrixPair GetSquareMatrices() {
        auto size = GetRandomMatrixSize();
        return {GetRandomMatrix(size, size), GetRandomMatrix(size, size)};
    }

    MatrixPair GetRectangleMatrices() {
        auto col_row = GetRandomMatrixSize();
        auto m1_row = GetRandomMatrixSize();
        auto m2_col = GetRandomMatrixSize();
        return {GetRandomMatrix(m1_row, col_row),
                GetRandomMatrix(col_row, m2_col)};
    }

private:
    std::mt19937 rng_;
    IntDistribution rd_number_;
    IntDistribution rd_matrix_size_;
};
} // namespace matrix_lib::tests
