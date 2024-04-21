#pragma once

#include "../types/matrix.h"
#include "../utils/is_float_complex.h"

#include <random>

namespace matrix_lib::tests {
template <typename T>
class RandomMatrixGenerator {
    using IntDistribution = std::uniform_int_distribution<int32_t>;

public:
    explicit RandomMatrixGenerator(int32_t seed)
        : rng_(seed), rd_number_(kNumberFrom, kNumberTo),
          rd_matrix_size_(kMatrixMinSize, kMatrixMaxSize) {
    }

    int32_t GetRandomInt() {
        return rd_number_(rng_);
    }

    T GetRandomTypeNumber() {
        if constexpr (utils::details::IsFloatComplexT<T>::value) {
            using F = T::value_type;
            auto real = static_cast<F>(GetRandomInt());
            auto imag = static_cast<F>(GetRandomInt());
            return {real, imag};
        }

        return static_cast<T>(GetRandomInt());
    }

    int32_t GetRandomMatrixSize() {
        return rd_matrix_size_(rng_);
    }

    Matrix<T> GetRandomMatrix(int32_t row, int32_t col) {
        Matrix<T> result(row, col);
        result.ApplyForEach([&](T &val) { val = GetRandomTypeNumber(); });
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
} // namespace matrix_lib::tests
