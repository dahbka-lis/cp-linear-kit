#pragma once

#include "../types/matrix.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T>
struct GivensPair {
    T cos = T{0};
    T sin = T{0};
};

template <utils::FloatOrComplex T>
GivensPair<T> GetGivensCoefficients(T first_elem, T second_elem) {
    auto abs_first = std::abs(first_elem);
    auto abs_second = std::abs(second_elem);
    auto sqrt_abs = std::sqrt(abs_first * abs_first + abs_second * abs_second);

    return {first_elem / sqrt_abs, -second_elem / sqrt_abs};
}

template <utils::FloatOrComplex T>
Matrix<T> GetGivensMatrix(IndexType size, IndexType from, IndexType to,
                          T first_elem, T second_elem) {
    auto res = Matrix<T>::Identity(size);
    auto [cos, sin] = GetGivensCoefficients(first_elem, second_elem);

    res(from, from) = cos;
    res(from, to) = -sin;

    if constexpr (utils::IsFloatComplexValue<T>()) {
        res(to, from) = std::conj(sin);
        res(to, to) = std::conj(cos);
    } else {
        res(to, from) = sin;
        res(to, to) = cos;
    }

    return res;
}
} // namespace matrix_lib::algorithms
