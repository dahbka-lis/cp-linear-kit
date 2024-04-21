#pragma once

#include "../matrix_utils/checks.h"
#include "../matrix_utils/is_matrix_type.h"
#include "householder.h"

namespace matrix_lib::algorithms {
namespace details {
template <utils::FloatOrComplex T = long double>
struct HessenbergBasis {
    Matrix<T> H;
    Matrix<T> Q;
};
} // namespace details

using IndexType = matrix_lib::details::Types::IndexType;

template <utils::MatrixType M>
inline details::HessenbergBasis<typename M::ElemType>
GetHessenbergForm(const M &matrix) {
    using T = typename M::ElemType;

    assert(utils::IsSquare(matrix) && "Hessenberg form for square matrices");

    Matrix<T> Q = Matrix<T>::Identity(matrix.Rows());
    Matrix<T> H = matrix;

    for (IndexType col = 0; col < matrix.Rows() - 2; ++col) {
        Matrix<T> vec = H.GetSubmatrix({col + 1, H.Rows()}, {col, col + 1});
        HouseholderReduction(vec);

        HouseholderLeftReflection(Q, vec, col + 1);
        HouseholderLeftReflection(H, vec, col + 1, col);
        HouseholderRightReflection(H, vec.Conjugate(), col + 1, 0);
    }

    Q.Conjugate();
    H.RoundZeroes();
    return {std::move(H), std::move(Q)};
}
} // namespace matrix_lib::algorithms
