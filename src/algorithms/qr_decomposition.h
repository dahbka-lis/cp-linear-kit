#pragma once

#include "../matrix_utils/checks.h"
#include "givens.h"
#include "householder.h"

namespace LinearKit::Algorithm {
namespace Details {
template <Utils::FloatOrComplex T = long double>
struct PairQR {
    Matrix<T> Q;
    Matrix<T> R;
};
} // namespace Details

using IndexType = LinearKit::Details::Types::IndexType;

template <MatrixUtils::MatrixType M>
inline Details::PairQR<typename M::ElemType> HessenbergQR(const M &matrix) {
    using T = typename M::ElemType;

    assert(MatrixUtils::IsHessenberg(matrix) &&
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

template <MatrixUtils::MatrixType M>
inline Details::PairQR<typename M::ElemType> HouseholderQR(const M &matrix) {
    using T = typename M::ElemType;

    if (MatrixUtils::IsHessenberg(matrix)) {
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

template <MatrixUtils::MatrixType M>
inline Details::PairQR<typename M::ElemType> GivensQR(const M &matrix) {
    using T = typename M::ElemType;

    if (MatrixUtils::IsHessenberg(matrix)) {
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
} // namespace LinearKit::Algorithm
