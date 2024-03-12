#pragma once

#include "givens.h"
#include "householder.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T = long double>
struct PairQR {
    Matrix<T> Q;
    Matrix<T> R;
};

template <utils::FloatOrComplex T = long double>
PairQR<T> HouseholderQR(const MatrixView<T> &matrix) {
    auto Q = Matrix<T>::Identity(matrix.Rows());
    auto R = matrix.Copy();

    for (IndexType col = 0; col < std::min(matrix.Rows(), matrix.Columns());
         ++col) {
        auto vec = R.GetSubmatrix(col, R.Rows(), col, col + 1).Copy();
        HouseholderReduction(vec);

        HouseholderLeftReflection(R, vec, col, col);
        HouseholderLeftReflection(Q, vec, col);
    }

    Q.Conjugate();
    R.RoundZeroes();
    return {Q, R};
}

template <utils::FloatOrComplex T = long double>
PairQR<T> HouseholderQR(const Matrix<T> &matrix) {
    return HouseholderQR(matrix.View());
}

template <utils::FloatOrComplex T = long double>
PairQR<T> GivensQR(const MatrixView<T> &matrix) {
    auto Q = Matrix<T>::Identity(matrix.Rows());
    auto R = matrix.Copy();

    for (IndexType col = 0; col < std::min(matrix.Rows(), matrix.Columns());
         ++col) {
        for (IndexType row = matrix.Rows() - 2; row + 1 > col; --row) {
            auto first = R(row, col);
            auto second = R(row + 1, col);

            GivensLeftRotation(R, row, row + 1, first, second);
            GivensLeftRotation(Q, row, row + 1, first, second);
        }
    }

    Q.Conjugate();
    R.RoundZeroes();
    return {Q, R};
}

template <utils::FloatOrComplex T = long double>
PairQR<T> GivensQR(const Matrix<T> &matrix) {
    return GivensQR(matrix.View());
}
} // namespace matrix_lib::algorithms
