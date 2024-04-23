#pragma once

#include "../types/types_details.h"

namespace LinearKit::Algorithm {
namespace Details {
template <Utils::FloatOrComplex T = long double>
struct GivensPair {
    T cos = T{0};
    T sin = T{0};
};

template <Utils::FloatOrComplex T = long double>
inline GivensPair<T> GetGivensCoefficients(T first_elem, T second_elem) {
    auto sqrt_abs = std::sqrt(std::norm(first_elem) + std::norm(second_elem));

    if (Utils::IsZeroFloating(sqrt_abs)) {
        return {T{1}, T{0}};
    }

    return {first_elem / sqrt_abs, -second_elem / sqrt_abs};
}
} // namespace Details

using IndexType = LinearKit::Details::Types::IndexType;

template <MatrixUtils::MutableMatrixType M>
inline void GivensLeftRotation(M &matrix, IndexType f_row, IndexType s_row,
                               typename M::ElemType first,
                               typename M::ElemType second) {
    using T = typename M::ElemType;
    auto [cos, sin] = Details::GetGivensCoefficients(first, second);

    for (IndexType i = 0; i < matrix.Columns(); ++i) {
        auto cp_from = matrix(f_row, i);
        auto cp_to = matrix(s_row, i);

        if constexpr (Utils::Details::IsFloatComplexT<T>::value) {
            matrix(f_row, i) =
                std::conj(cos) * cp_from - std::conj(sin) * cp_to;
        } else {
            matrix(f_row, i) = cos * cp_from - sin * cp_to;
        }

        matrix(s_row, i) = cos * cp_to + sin * cp_from;
    }
}

template <MatrixUtils::MutableMatrixType M>
inline void GivensRightRotation(M &matrix, IndexType f_col, IndexType s_col,
                                typename M::ElemType first,
                                typename M::ElemType second) {
    using T = typename M::ElemType;
    auto [cos, sin] = Details::GetGivensCoefficients(first, second);

    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        auto cp_from = matrix(i, f_col);
        auto cp_to = matrix(i, s_col);

        if constexpr (Utils::Details::IsFloatComplexT<T>::value) {
            matrix(i, f_col) =
                std::conj(cos) * cp_from - std::conj(sin) * cp_to;
        } else {
            matrix(i, f_col) = cos * cp_from - sin * cp_to;
        }

        matrix(i, s_col) = cos * cp_to + sin * cp_from;
    }
}
} // namespace LinearKit::Algorithm
