#pragma once

#include "../types/matrix.h"

namespace matrix_lib::algorithms {
using IndexType = details::Types::IndexType;

template <utils::FloatOrComplex T = long double>
struct GivensPair {
    T cos = T{0};
    T sin = T{0};
};

template <utils::FloatOrComplex T = long double>
GivensPair<T> GetGivensCoefficients(T first_elem, T second_elem) {
    auto sqrt_abs = std::sqrt(std::norm(first_elem) + std::norm(second_elem));

    if (utils::IsZeroFloating(sqrt_abs)) {
        return {T{1}, T{0}};
    }

    return {first_elem / sqrt_abs, -second_elem / sqrt_abs};
}

template <utils::FloatOrComplex T = long double>
void GivensLeftRotation(Matrix<T> &matrix, IndexType from, IndexType to,
                        T first, T second) {
    auto [cos, sin] = GetGivensCoefficients(first, second);

    for (IndexType i = 0; i < matrix.Columns(); ++i) {
        auto cp_from = matrix(from, i);
        auto cp_to = matrix(to, i);

        if constexpr (utils::details::IsFloatComplexT<T>::value) {
            matrix(from, i) = std::conj(cos) * cp_from - std::conj(sin) * cp_to;
        } else {
            matrix(from, i) = cos * cp_from - sin * cp_to;
        }

        matrix(to, i) = cos * cp_to + sin * cp_from;
    }
}

template <utils::FloatOrComplex T = long double>
void GivensRightRotation(Matrix<T> &matrix, IndexType from, IndexType to,
                         T first, T second) {
    auto [cos, sin] = GetGivensCoefficients(first, second);

    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        auto cp_from = matrix(i, from);
        auto cp_to = matrix(i, to);

        if constexpr (utils::details::IsFloatComplexT<T>::value) {
            matrix(i, from) = std::conj(cos) * cp_from + std::conj(sin) * cp_to;
        } else {
            matrix(i, from) = cos * cp_from + sin * cp_to;
        }

        if constexpr (utils::details::IsFloatComplexT<T>::value) {
            matrix(i, to) = cos * cp_to - sin * cp_from;
        } else {
            matrix(i, to) = cos * cp_to - sin * cp_from;
        }
    }
}
} // namespace matrix_lib::algorithms
