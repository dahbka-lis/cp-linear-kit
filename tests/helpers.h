#pragma once

#include <gtest/gtest.h>
#include <random>

#include "../types/matrix.h"

namespace matrix_lib::tests {
template <typename T>
void CompareMatrices(const Matrix<T> &m1,
                     const std::vector<std::vector<T>> &m2) {
    ASSERT_EQ(m1.Rows(), m2.size())
        << "Matrix height size mismatch for compare.";

    for (size_t i = 0; i < m2.size(); ++i) {
        ASSERT_EQ(m1.Columns(), m2[i].size())
            << "Matrix width size mismatch for compare.";

        for (size_t j = 0; j < m2[i].size(); ++j) {
            ASSERT_TRUE(matrix_lib::utils::IsEqualFloating(m1(i, j), m2[i][j]))
                << "Matrices are not equal.";
        }
    }
}

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
          rd_matrix_size_(matrix_min_size, matrix_max_size) {}

    int32_t GetRandomValue() { return rd_number_(rng_); }

    int32_t GetRandomMatrixSize() { return rd_matrix_size_(rng_); }

    Matrix<T> GetRandomMatrix(int32_t row, int32_t col) {
        Matrix<T> result(row, col);
        result.ApplyToEach([&](T &val) { val = GetRandomValue(); });
        return result;
    }

    MatrixPair GetSquareMatrices() {
        auto size = GetRandomMatrixSize();
        auto m1 = GetRandomMatrix(size, size);
        auto m2 = GetRandomMatrix(size, size);

        return {m1, m2};
    }

    MatrixPair GetRectangleMatrices() {
        auto col_row = GetRandomMatrixSize();
        auto m1_row = GetRandomMatrixSize();
        auto m2_col = GetRandomMatrixSize();

        auto m1 = GetRandomMatrix(m1_row, col_row);
        auto m2 = GetRandomMatrix(col_row, m2_col);

        return {m1, m2};
    }

private:
    std::mt19937 rng_;
    IntDistribution rd_number_;
    IntDistribution rd_matrix_size_;
};
} // namespace matrix_lib::tests
