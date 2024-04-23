#pragma once

#include "../matrix_utils/cast_matrix.h"
#include "bidiagonalization.h"
#include "qr_algorithm_bidiag.h"

namespace LinearKit::Algorithm {
namespace Details {
template <Utils::FloatOrComplex T>
struct SingularBasis {
    Matrix<T> U;
    Matrix<T> S;
    Matrix<T> VT;
};

template <MatrixUtils::MutableMatrixType M>
inline void ToPositiveSingular(M &S, M &VT) {
    using T = typename M::ElemType;

    for (IndexType i = 0; i < std::min(S.Rows(), S.Columns()); ++i) {
        if (S(i, i) >= 0)
            continue;

        S(i, i) *= -1;
        for (IndexType j = 0; j < S.Columns(); ++j) {
            VT(i, j) *= -1;
        }
    }
}

template <MatrixUtils::MutableMatrixType M>
inline void SwapColumns(M &matrix, IndexType first, IndexType second) {
    assert(first < matrix.Columns() && second < matrix.Columns() &&
           "Wrong column index.");

    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        std::swap(matrix(i, first), matrix(i, second));
    }
}

template <MatrixUtils::MutableMatrixType M>
inline void SwapRows(M &matrix, IndexType first, IndexType second) {
    assert(first < matrix.Rows() && second < matrix.Rows() &&
           "Wrong row index.");

    for (IndexType i = 0; i < matrix.Columns(); ++i) {
        std::swap(matrix(first, i), matrix(second, i));
    }
}

template <MatrixUtils::MutableMatrixType M>
inline void SortSingular(M &U, M &S, M &VT) {
    using T = typename M::ElemType;
    auto min_size = std::min(S.Rows(), S.Columns());

    for (IndexType i = 0; i < min_size; ++i) {
        for (IndexType j = 0; j < min_size - i - 1; ++j) {
            if (S(j, j) >= S(j + 1, j + 1))
                continue;

            std::swap(S(j, j), S(j + 1, j + 1));
            SwapColumns(U, j, j + 1);
            SwapRows(VT, j, j + 1);
        }
    }
}
} // namespace Details

template <MatrixUtils::MatrixType M>
inline Details::SingularBasis<typename M::ElemType> SVD(const M &matrix) {
    using T = typename M::ElemType;

    if (matrix.Rows() < matrix.Columns()) {
        auto [U, S, VT] = SVD(Matrix<typename M::ElemType>::Conjugated(matrix));
        U.Conjugate();
        VT.Conjugate();
        S.Transpose();
        return {std::move(VT), std::move(S), std::move(U)};
    }

    auto [U1, B, VT1] = Bidiagonalize(matrix);
    auto B_real = MatrixUtils::CastMatrix<long double>(B);
    auto [U_real, S_real, VT_real] = BidiagAlgorithmQR(B_real);

    Details::ToPositiveSingular(S_real, VT_real);
    Details::SortSingular(U_real, S_real, VT_real);

    auto S_d = S_real.GetDiag();
    S_d.Transpose();

    auto S = MatrixUtils::CastMatrix<T>(S_d);
    auto U2 = MatrixUtils::CastMatrix<T>(U_real);
    auto VT2 = MatrixUtils::CastMatrix<T>(VT_real);

    auto VT = VT2 * VT1;
    auto U = U1 * U2;

    return {std::move(U), std::move(S), std::move(VT)};
}
} // namespace LinearKit::Algorithm
