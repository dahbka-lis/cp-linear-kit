#pragma once

#include "../types/matrix.h"
#include "householder.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T>
struct PairQR {
    Matrix<T> Q;
    Matrix<T> R;
};

template <utils::FloatOrComplex T>
PairQR<T> HouseholderQR(const Matrix<T> &matrix) {
    assert(matrix.Rows() >= matrix.Columns() &&
           "Transpose the matrix for decompose it to QR.");
    PairQR<T> pair = {Matrix<T>::Identity(matrix.Rows()), matrix};

    for (IndexType col = 0; col < matrix.Columns(); ++col) {
        Matrix<T> vec = pair.R.GetSubmatrix(col, pair.R.Rows(), col, col + 1);
        HouseholderReduction(vec);

        HouseholderLeftReflection(pair.R, vec, col, col);
        HouseholderLeftReflection(pair.Q, vec, col);
    }

    pair.Q.Transpose();
    return pair;
}
} // namespace matrix_lib::algorithms
