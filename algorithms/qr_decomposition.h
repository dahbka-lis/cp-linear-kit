#pragma once

#include "givens.h"
#include "householder.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T>
struct PairQR {
    Matrix<T> Q;
    Matrix<T> R;
};

template <utils::FloatOrComplex T>
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

template <utils::FloatOrComplex T>
PairQR<T> HouseholderQR(const Matrix<T> &matrix) {
    return HouseholderQR(matrix.View());
}

template <utils::FloatOrComplex T>
PairQR<T> GivensQR(const MatrixView<T> &matrix) {
    auto Q = Matrix<T>::Identity(matrix.Rows());
    auto R = matrix.Copy();

    for (IndexType col = 0; col < std::min(matrix.Rows(), matrix.Columns());
         ++col) {
        for (IndexType row = matrix.Rows() - 2; row + 1 > col; --row) {
            auto sub_R = R.GetSubmatrix(row, row + 2, 0, R.Columns());
            auto sub_Q = Q.GetSubmatrix(row, row + 2, 0, Q.Columns());
            auto givens =
                GetGivensMatrix(2, 0, 1, sub_R(0, col), sub_R(1, col));

            R.AssignSubmatrix(givens * sub_R, row, 0);
            Q.AssignSubmatrix(givens * sub_Q, row, 0);
        }
    }

    Q.Conjugate();
    R.RoundZeroes();
    return {Q, R};
}

template <utils::FloatOrComplex T>
PairQR<T> GivensQR(const Matrix<T> &matrix) {
    return GivensQR(matrix.View());
}
} // namespace matrix_lib::algorithms
