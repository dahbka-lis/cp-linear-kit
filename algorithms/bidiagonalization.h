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

        if (col + 2 < B.Columns()) {
            Matrix<T> row_reduction =
                B.GetSubmatrix({col, col + 1}, {col + 1, B.Columns()});
            HouseholderReduction(row_reduction);

            HouseholderRightReflection(B, row_reduction, col + 1, col);
            HouseholderRightReflection(V, row_reduction, col + 1);
        }
    }

    U.Conjugate();
    V.Conjugate();
    B.RoundZeroes();
    return {std::move(U), std::move(B), std::move(V)};
}
} // namespace matrix_lib::algorithms
