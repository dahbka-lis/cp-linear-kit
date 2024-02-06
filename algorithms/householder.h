#pragma once

#include "../types/matrix.h"
#include "../utils/sign.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T>
void HouseholderReduction(Matrix<T> &vector) {
    vector(0, 0) += utils::Sign(vector(0, 0)) * vector.GetEuclideanNorm();
    vector.Normalize();
}

template <utils::FloatOrComplex T>
void HouseholderLeftReflection(Matrix<T> &matrix, const Matrix<T> &vec,
                               IndexType row = 0, IndexType c_from = 0,
                               IndexType c_to = -1) {
    if (c_to == -1) {
        c_to = matrix.Columns();
    }

    Matrix<T> sub = matrix.GetSubmatrix(row, row + vec.Rows(), c_from, c_to);
    sub -= (T{2} * vec) * (Matrix<T>::Conjugated(vec) * sub);
    matrix.AssignSubmatrix(sub, row, c_from);
}

template <utils::FloatOrComplex T>
void HouseholderRightReflection(Matrix<T> &matrix, const Matrix<T> &vec,
                                IndexType col = 0, IndexType r_from = 0,
                                IndexType r_to = -1) {
    if (r_to == -1) {
        r_to = matrix.Rows();
    }

    Matrix<T> sub =
        matrix.GetSubmatrix(r_from, r_to, col, col + matrix.Columns());
    sub -= (sub * Matrix<T>::Conjugated(vec)) * (T{2} * vec);
    matrix.AssignSubmatrix(sub, r_from, col);
}
} // namespace matrix_lib::algorithms
