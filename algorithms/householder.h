#pragma once

#include "../types/matrix/matrix.h"

namespace matrix_lib::algorithm {
template <utils::FloatOrComplex T>
void HouseholderReduction(Matrix<T> &vector) {
    vector(0, 0) -= vector.GetEuclideanNorm();
    vector.Normalize();
}

template <utils::FloatOrComplex T>
void HouseholderLeftReflection(Matrix<T> &matrix) {
    auto r_vector = matrix.GetColumn(0);
    HouseholderReduction(r_vector);
    matrix -= T{2} * (r_vector * Matrix<T>::Transposed(r_vector)) * matrix;
}

template <utils::FloatOrComplex T>
void HouseholderRightReflection(Matrix<T> &matrix) {
    auto r_vector = matrix.GetColumn(matrix.Columns() - 1);
    HouseholderReduction(r_vector);
    matrix -= T{2} * matrix * (r_vector * Matrix<T>::Transposed(r_vector));
}
} // namespace matrix_lib::algorithm
