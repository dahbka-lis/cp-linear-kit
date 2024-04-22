#pragma once

#include "../matrix_utils/is_matrix_type.h"
#include "householder.h"

namespace matrix_lib::algorithms {
namespace details {
template <utils::FloatOrComplex T = long double>
struct BidiagonalBasis {
    Matrix<T> U;
    Matrix<T> B;
    Matrix<T> VT;
};

template <utils::MutableMatrixType F, utils::MutableMatrixType S>
void RowToReal(F &B, S &U, IndexType idx) {
    auto coeff = std::conj(B(idx, idx) / std::abs(B(idx, idx)));
    auto B_row = B.GetRow(idx);
    auto U_row = U.GetRow(idx);

    B_row *= coeff;
    U_row *= coeff;
}

template <utils::MutableMatrixType F, utils::MutableMatrixType S>
void ColumnToReal(F &B, S &V, IndexType idx) {
    auto coeff = std::conj(B(idx, idx + 1) / std::abs(B(idx, idx + 1)));
    auto B_row = B.GetColumn(idx + 1);
    auto V_col = V.GetColumn(idx + 1);

    B_row *= coeff;
    V_col *= coeff;
}
} // namespace details

using IndexType = matrix_lib::details::Types::IndexType;

template <utils::MatrixType M>
inline details::BidiagonalBasis<typename M::ElemType>
Bidiagonalize(const M &matrix) {
    using T = typename M::ElemType;

    Matrix<T> B = matrix;
    Matrix<T> U = Matrix<T>::Identity(B.Rows());
    Matrix<T> V = Matrix<T>::Identity(B.Columns());

    for (IndexType col = 0; col < std::min(B.Rows(), B.Columns()); ++col) {
        Matrix<T> col_reduction =
            B.GetSubmatrix({col, B.Rows()}, {col, col + 1});
        HouseholderReduction(col_reduction);

        HouseholderLeftReflection(B, col_reduction, col, col);
        HouseholderLeftReflection(U, col_reduction, col);

        if constexpr (utils::details::IsFloatComplexT<T>::value)
            details::RowToReal(B, U, col);

        if (col + 1 >= B.Columns())
            continue;

        Matrix<T> row_reduction =
            B.GetSubmatrix({col, col + 1}, {col + 1, B.Columns()});
        HouseholderReduction(row_reduction);

        HouseholderRightReflection(B, row_reduction, col + 1, col);
        HouseholderRightReflection(V, row_reduction, col + 1);

        if constexpr (utils::details::IsFloatComplexT<T>::value)
            details::ColumnToReal(B, V, col);
    }

    U.Conjugate();
    V.Conjugate();
    B.RoundZeroes();
    return {std::move(U), std::move(B), std::move(V)};
}
} // namespace matrix_lib::algorithms
