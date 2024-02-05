#pragma once

#include "../types/matrix/matrix.h"
#include "householder.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T>
struct PairQR {
    Matrix<T> Q;
    Matrix<T> R;
};

template <utils::FloatOrComplex T>
PairQR<T> DecomposeQR(const Matrix<T> &matrix) {
    assert(matrix.Rows() == matrix.Columns() &&
           "QR Decomposition for square matrices.");

    PairQR<T> pair = {Matrix<T>::Identity(matrix.Rows()), matrix};

    for (IndexType col = 0; col < matrix.Rows() - 1; ++col) {
        Matrix<T> vec =
            pair.R.GetColumn(col).GetSubmatrix(col, pair.R.Rows(), 0, 1);
        HouseholderReduction(vec);

        HouseholderLeftReflection(pair.R, vec, col, col);
        HouseholderLeftReflection(pair.Q, vec, col);
    }

    pair.Q.Transpose();
    return pair;
}
} // namespace matrix_lib::algorithms
