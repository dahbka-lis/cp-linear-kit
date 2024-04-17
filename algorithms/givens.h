#pragma once

#include "../types/matrix.h"

namespace matrix_lib::algorithms {
namespace details {
template <utils::FloatOrComplex T = long double>
struct GivensPair {
    T cos = T{0};
    T sin = T{0};
};

template <utils::FloatOrComplex T = long double>
inline GivensPair<T> GetGivensCoefficients(T first_elem, T second_elem) {
    auto sqrt_abs = std::sqrt(std::norm(first_elem) + std::norm(second_elem));

    if (utils::IsZeroFloating(sqrt_abs)) {
        return {T{1}, T{0}};
    }

    return {first_elem / sqrt_abs, -second_elem / sqrt_abs};
}
} // namespace details

using IndexType = matrix_lib::details::Types::IndexType;

template <utils::MutableMatrixType M>
inline void GivensLeftRotation(M &matrix, IndexType f_row, IndexType s_row,
                               typename M::ElemType first,
                               typename M::ElemType second) {
    using T = typename M::ElemType;
    auto [cos, sin] = details::GetGivensCoefficients(first, second);

    for (IndexType i = 0; i < matrix.Columns(); ++i) {
        auto cp_from = matrix(f_row, i);
        auto cp_to = matrix(s_row, i);

        if constexpr (utils::details::IsFloatComplexT<T>::value) {
            matrix(f_row, i) =
                std::conj(cos) * cp_from - std::conj(sin) * cp_to;
        } else {
            matrix(f_row, i) = cos * cp_from - sin * cp_to;
        }

        matrix(s_row, i) = cos * cp_to + sin * cp_from;
    }

    matrix.RoundZeroes();
}

template <utils::MutableMatrixType M>
inline void GivensRightRotation(M &matrix, IndexType f_col, IndexType s_col,
                                typename M::ElemType first,
                                typename M::ElemType second) {
    using T = typename M::ElemType;
    auto [cos, sin] = details::GetGivensCoefficients(first, second);

    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        auto cp_from = matrix(i, f_col);
        auto cp_to = matrix(i, s_col);

        if constexpr (utils::details::IsFloatComplexT<T>::value) {
            matrix(i, f_col) =
                std::conj(cos) * cp_from - std::conj(sin) * cp_to;
        } else {
            matrix(i, f_col) = cos * cp_from - sin * cp_to;
        }

        matrix(i, s_col) = cos * cp_to + sin * cp_from;
    }

    matrix.RoundZeroes();
}
} // namespace matrix_lib::algorithms
