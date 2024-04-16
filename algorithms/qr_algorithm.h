#pragma once

#include "../matrix_utils/checks.h"
#include "givens.h"
#include "qr_decomposition.h"

namespace matrix_lib::algorithms {
namespace details {
template <utils::MatrixType M>
inline typename M::ElemType GetWilkinsonShift(const M &matrix) {
    assert(matrix.Rows() == 2 && matrix.Columns() == 2 &&
           "Wilkinson shift for 2x2 matrix.");
    assert(matrix(0, 1) == matrix(1, 0) &&
           "Wilkinson shift for symmetric matrix.");

    auto d = (matrix(0, 0) - matrix(1, 1)) / 2;
    auto coefficient =
        std::abs(d) + std::sqrt(d * d + matrix(0, 1) * matrix(0, 1));

    return matrix(1, 1) -
           (utils::Sign(d) * matrix(0, 1) * matrix(0, 1)) / coefficient;
}

template <utils::MatrixType M>
inline typename M::ElemType GetBidiagWilkinsonShift(const M &S) {
    using T = typename M::ElemType;

    auto r = S.Rows();
    auto c = S.Columns();

    auto minor = S.GetSubmatrix({r - 2, r}, {c - 2, c});
    auto BB = Matrix<T>(2);

    BB(0, 0) = minor(0, 0) * minor(0, 0);
    BB(1, 0) = minor(0, 0) * minor(0, 1);
    BB(0, 1) = BB(1, 0);
    BB(1, 1) = minor(0, 1) * minor(0, 1) + minor(1, 1) * minor(1, 1);

    if (r >= 3) {
        BB(0, 0) += S(r - 3, c - 2) * S(r - 3, c - 2);
    }

    return GetWilkinsonShift(BB);
}

template <utils::FloatOrComplex T = long double>
struct SpectralPair {
    Matrix<T> D;
    Matrix<T> U;
};

template <utils::FloatOrComplex T>
struct DiagBasisQR {
    Matrix<T> U;
    Matrix<T> D;
    Matrix<T> VT;
};
} // namespace details

template <utils::MatrixType M>
inline details::SpectralPair<typename M::ElemType>
GetSpecDecomposition(const M &matrix,
                     typename M::ElemType shift = typename M::ElemType{0},
                     std::size_t it_cnt = 100) {
    using T = typename M::ElemType;

    assert(utils::IsHermitian(matrix) &&
           "Spectral decomposition for hermitian matrix.");

    Matrix<T> D = matrix;
    Matrix<T> shift_I = Matrix<T>::Identity(D.Rows()) * shift;
    Matrix<T> U = Matrix<T>::Identity(D.Rows());

    for (std::size_t i = 0; i < it_cnt; ++i) {
        auto [Q, R] = HouseholderQR(D - shift_I);
        D = R * Q + shift_I;
        U *= Q;

        D.RoundZeroes();
    }

    return {std::move(D), std::move(U)};
}

template <utils::MatrixType M>
inline details::DiagBasisQR<typename M::ElemType>
BidiagAlgorithmQR(const M &B, IndexType it_cnt = 100) {
    using T = typename M::ElemType;

    Matrix<T> D = B;
    IndexType size = std::min(D.Rows(), D.Columns());

    auto U = Matrix<T>::Identity(D.Rows());
    auto VT = Matrix<T>::Identity(D.Columns());

    while (--it_cnt) {
        auto shift = details::GetBidiagWilkinsonShift(D);
        for (IndexType i = 0; i < size; ++i) {
            if (i + 1 < D.Columns()) {
                auto f_elem = (i > 0) ? D(i - 1, i) : D(0, 0) * D(0, 0) - shift;
                auto s_elem = (i > 0) ? D(i - 1, i + 1) : D(0, 1) * D(0, 0);

                GivensLeftRotation(VT, i, i + 1, f_elem, s_elem);
                GivensRightRotation(D, i, i + 1, f_elem, s_elem);
            }

            if (i + 1 < D.Rows()) {
                GivensRightRotation(U, i, i + 1, D(i, i), D(i + 1, i));
                GivensLeftRotation(D, i, i + 1, D(i, i), D(i + 1, i));
            }
        }

        D.RoundZeroes();
    }

    return {std::move(U), std::move(D), std::move(VT)};
}
} // namespace matrix_lib::algorithms
