#pragma once

#include "../matrix_utils/checks.h"
#include "householder.h"

namespace matrix_lib::algorithms {
using IndexType = details::Types::IndexType;

template <utils::FloatOrComplex T = long double>
struct HessenbergBasis {
    Matrix<T> H;
    Matrix<T> Q;
};

template <utils::FloatOrComplex T = long double>
HessenbergBasis<T> GetHessenbergForm(const ConstMatrixView<T> &matrix) {
    assert(utils::IsSquare(matrix) && "Hessenberg form for square matrices");

    Matrix<T> Q = Matrix<T>::Identity(matrix.Rows());
    Matrix<T> H = matrix;

    for (IndexType col = 0; col < std::min(matrix.Rows(), matrix.Columns()) - 2;
         ++col) {
        Matrix<T> vec = H.GetSubmatrix({col + 1, H.Rows()}, {col, col + 1});
        HouseholderReduction(vec);

        HouseholderLeftReflection(Q, vec, col + 1);
        HouseholderLeftReflection(H, vec, col + 1, col);
        HouseholderRightReflection(H, vec.Conjugate(), col + 1, col);
    }

    Q.Conjugate();
    H.RoundZeroes();
    return {H, Q};
}

template <utils::FloatOrComplex T = long double>
HessenbergBasis<T> GetHessenbergForm(const MatrixView<T> &matrix) {
    return GetHessenbergForm(matrix.ConstView());
}

template <utils::FloatOrComplex T = long double>
HessenbergBasis<T> GetHessenbergForm(const Matrix<T> &matrix) {
    return GetHessenbergForm(matrix.View());
}
} // namespace matrix_lib::algorithms
