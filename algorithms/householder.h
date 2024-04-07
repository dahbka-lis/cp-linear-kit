#pragma once

#include "../types/types_details.h"
#include "../utils/sign.h"

namespace matrix_lib::algorithms {
using IndexType = details::Types::IndexType;

template <utils::FloatOrComplex T = long double>
void HouseholderReduction(MatrixView<T> &vector) {
    vector(0, 0) += utils::Sign(vector(0, 0)) * vector.GetEuclideanNorm();
    vector.Normalize();
}

template <utils::FloatOrComplex T = long double>
void HouseholderReduction(Matrix<T> &vector) {
    auto view = vector.View();
    HouseholderReduction(view);
}

template <utils::FloatOrComplex T = long double>
void HouseholderLeftReflection(MatrixView<T> &matrix, const Matrix<T> &vec,
                               IndexType row = 0, IndexType c_from = 0,
                               IndexType c_to = -1) {
    if (c_to == -1) {
        c_to = matrix.Columns();
    }

    MatrixView<T> sub =
        matrix.GetSubmatrix({row, row + vec.Rows()}, {c_from, c_to});
    sub -= (T{2} * vec) * (ConstMatrixView<T>::Conjugated(vec) * sub);
}

template <utils::FloatOrComplex T = long double>
void HouseholderLeftReflection(Matrix<T> &matrix, const Matrix<T> &vec,
                               IndexType row = 0, IndexType c_from = 0,
                               IndexType c_to = -1) {
    auto view = matrix.View();
    HouseholderLeftReflection(view, vec, row, c_from, c_to);
}

template <utils::FloatOrComplex T = long double>
void HouseholderRightReflection(MatrixView<T> &matrix, const Matrix<T> &vec,
                                IndexType col = 0, IndexType r_from = 0,
                                IndexType r_to = -1) {
    if (r_to == -1) {
        r_to = matrix.Rows();
    }

    MatrixView<T> sub =
        matrix.GetSubmatrix({r_from, r_to}, {col, col + vec.Columns()});
    sub -= (sub * ConstMatrixView<T>::Conjugated(vec)) * (T{2} * vec);
}

template <utils::FloatOrComplex T = long double>
void HouseholderRightReflection(Matrix<T> &matrix, const Matrix<T> &vec,
                                IndexType col = 0, IndexType r_from = 0,
                                IndexType r_to = -1) {
    HouseholderRightReflection(matrix.View(), vec, col, r_from, r_to);
}
} // namespace matrix_lib::algorithms
