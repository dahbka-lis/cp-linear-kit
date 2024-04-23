#pragma once

#include "../matrix_utils/is_matrix_type.h"
#include "../utils/sign.h"

namespace LinearKit::Algorithm {
using IndexType = LinearKit::Details::Types::IndexType;

template <MatrixUtils::MutableMatrixType M>
inline void HouseholderReduction(M &vector) {
    vector(0, 0) -= Utils::Sign(vector(0, 0)) * vector.GetEuclideanNorm();
    vector.Normalize();
}

template <MatrixUtils::MutableMatrixType M, MatrixUtils::MatrixType V>
inline void HouseholderLeftReflection(M &matrix, const V &vec,
                                      IndexType row = 0, IndexType c_from = 0,
                                      IndexType c_to = -1) {
    using T = typename M::ElemType;

    if (c_to == -1) {
        c_to = matrix.Columns();
    }

    MatrixView<T> sub =
        matrix.GetSubmatrix({row, row + vec.Rows()}, {c_from, c_to});
    sub -= (T{2} * vec) * (Matrix<T>::Conjugated(vec) * sub);
}

template <MatrixUtils::MutableMatrixType M, MatrixUtils::MatrixType V>
inline void HouseholderRightReflection(M &matrix, const V &vec,
                                       IndexType col = 0, IndexType r_from = 0,
                                       IndexType r_to = -1) {
    using T = typename M::ElemType;

    if (r_to == -1) {
        r_to = matrix.Rows();
    }

    MatrixView<T> sub =
        matrix.GetSubmatrix({r_from, r_to}, {col, col + vec.Columns()});
    sub -= (sub * Matrix<T>::Conjugated(vec)) * (T{2} * vec);
}
} // namespace LinearKit::Algorithm
