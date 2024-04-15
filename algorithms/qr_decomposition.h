#pragma once

#include "../matrix_utils/checks.h"
#include "../types/types_details.h"
#include "givens.h"
#include "householder.h"

namespace matrix_lib::algorithms {
using IndexType = details::Types::IndexType;

template <utils::FloatOrComplex T = long double>
struct PairQR {
    Matrix<T> Q;
    Matrix<T> R;
};

template <utils::FloatOrComplex T>
PairQR<T> HessenbergQR(const ConstMatrixView<T> &matrix);

template <utils::FloatOrComplex T = long double>
PairQR<T> HouseholderQR(const ConstMatrixView<T> &matrix) {
    if (utils::IsHessenberg(matrix)) {
        return HessenbergQR(matrix);
    }

    Matrix<T> Q = Matrix<T>::Identity(matrix.Rows());
    Matrix<T> R = matrix;

    for (IndexType col = 0; col < std::min(matrix.Rows(), matrix.Columns());
         ++col) {
        Matrix<T> vec = R.GetSubmatrix({col, R.Rows()}, {col, col + 1});
        HouseholderReduction(vec);

        HouseholderLeftReflection(R, vec, col, col);
        HouseholderLeftReflection(Q, vec, col);
    }

    Q.Conjugate();
    R.RoundZeroes();
    return {std::move(Q), std::move(R)};
}

template <utils::FloatOrComplex T = long double>
PairQR<T> HouseholderQR(const MatrixView<T> &matrix) {
    return HouseholderQR(matrix.ConstView());
}

template <utils::FloatOrComplex T = long double>
PairQR<T> HouseholderQR(const Matrix<T> &matrix) {
    return HouseholderQR(matrix.View());
}

template <utils::FloatOrComplex T = long double>
PairQR<T> GivensQR(const ConstMatrixView<T> &matrix) {
    if (utils::IsHessenberg(matrix)) {
        return HessenbergQR(matrix);
    }

    Matrix<T> Q = Matrix<T>::Identity(matrix.Rows());
    Matrix<T> R = matrix;

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
    return {std::move(Q), std::move(R)};
}

template <utils::FloatOrComplex T = long double>
PairQR<T> GivensQR(const MatrixView<T> &matrix) {
    return GivensQR(matrix.ConstView());
}

template <utils::FloatOrComplex T = long double>
PairQR<T> GivensQR(const Matrix<T> &matrix) {
    return GivensQR(matrix.View());
}

template <utils::FloatOrComplex T = long double>
PairQR<T> HessenbergQR(const ConstMatrixView<T> &matrix) {
    assert(utils::IsHessenberg(matrix) &&
           "Hessenberg QR for hessenberg form of matrix.");

    Matrix<T> Q = Matrix<T>::Identity(matrix.Rows());
    Matrix<T> R = matrix;

    for (IndexType i = 0; i < std::min(R.Rows() - 1, R.Columns()); ++i) {
        auto first = R(i, i);
        auto second = R(i + 1, i);

        GivensLeftRotation(R, i, i + 1, first, second);
        GivensLeftRotation(Q, i, i + 1, first, second);
    }

    Q.Conjugate();
    R.RoundZeroes();
    return {std::move(Q), std::move(R)};
}

template <utils::FloatOrComplex T = long double>
PairQR<T> HessenbergQR(const MatrixView<T> &matrix) {
    return HessenbergQR(matrix.ConstView());
}

template <utils::FloatOrComplex T = long double>
PairQR<T> HessenbergQR(const Matrix<T> &matrix) {
    return HessenbergQR(matrix.View());
}
} // namespace matrix_lib::algorithms
