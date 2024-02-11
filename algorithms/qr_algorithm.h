#pragma once

#include "qr_decomposition.h"

namespace matrix_lib::algorithms {
template <utils::FloatOrComplex T>
Matrix<T> GetEigenvaluesNative(const Matrix<T> &matrix,
                               std::size_t it_cnt = 100) {
    assert(matrix.Rows() == matrix.Columns() &&
           "Eigenvalues with QR Algorithm for square matrix.");

    auto copy = matrix;
    for (std::size_t i = 0; i < it_cnt; ++i) {
        auto [Q, R] = HouseholderQR(copy);
        copy = R * Q;
    }

    return copy.GetDiag(true);
}
} // namespace matrix_lib::algorithms
