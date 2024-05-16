#pragma once

#include "../src/types/matrix.h"

#include <random>

namespace LinearKit::Tests {
template <typename T>
class RandomGenerator {
    using IntDistribution = std::uniform_int_distribution<int32_t>;

public:
    explicit RandomGenerator(int32_t seed)
        : rng_(seed), rd_number_(kNumberFrom, kNumberTo),
          rd_matrix_size_(kMatrixMinSize, kMatrixMaxSize) {
    }

    int32_t GetInt() {
        return rd_number_(rng_);
    }

    T GetRandomTypeNumber() {
        if constexpr (Utils::Details::IsFloatComplexT<T>::value) {
            using F = T::value_type;
            auto real = static_cast<F>(GetInt());
            auto imag = static_cast<F>(GetInt());
            return {real, imag};
        }

        return static_cast<T>(GetInt());
    }

    int32_t GetMatrixSize() {
        return rd_matrix_size_(rng_);
    }

    Matrix<T> GetMatrix(int32_t row, int32_t col) {
        Matrix<T> result(row, col);
        result.ApplyForEach([&](T &val) { val = GetRandomTypeNumber(); });
        return result;
    }

    Matrix<T> GetSymmetricMatrix(int32_t size) {
        Matrix<T> result(size);
        for (IndexType i = 0; i < size; ++i) {
            for (IndexType j = i; j < size; ++j) {
                result(i, j) = GetRandomTypeNumber();
                result(j, i) = result(i, j);
            }
        }
        return result;
    }

private:
    static constexpr int32_t kMatrixMinSize = 0;
    static constexpr int32_t kMatrixMaxSize = 100;
    static constexpr int32_t kNumberFrom = -100;
    static constexpr int32_t kNumberTo = 100;

    std::mt19937 rng_;
    IntDistribution rd_number_;
    IntDistribution rd_matrix_size_;
};
} // namespace LinearKit::Tests
