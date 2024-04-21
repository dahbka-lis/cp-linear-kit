#pragma once

#include "bidiagonalization.h"
#include "qr_algorithm.h"

namespace matrix_lib::algorithms {
namespace details {
template <utils::FloatOrComplex T>
struct SingularBasis {
    Matrix<T> U;
    Matrix<T> S;
    Matrix<T> VT;
};

template <utils::MutableMatrixType M>
inline void ToPositiveSingular(M &S, M &VT) {
    using T = typename M::ElemType;

    for (IndexType i = 0; i < std::min(S.Rows(), S.Columns()); ++i) {
        if (!utils::IsZeroFloating(S(i, i)))
            continue;

        S(i, i) *= -1;
        for (IndexType j = 0; j < S.Columns(); ++j) {
            VT(i, j) *= -1;
        }
    }
}

template <utils::MutableMatrixType M>
inline void SwapColumns(M &matrix, IndexType first, IndexType second) {
    assert(first < matrix.Columns() && second < matrix.Columns() &&
           "Wrong column index.");

    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        std::swap(matrix(i, first), matrix(i, second));
    }
}

template <utils::MutableMatrixType M>
inline void SwapRows(M &matrix, IndexType first, IndexType second) {
    assert(first < matrix.Rows() && second < matrix.Rows() &&
           "Wrong row index.");

    for (IndexType i = 0; i < matrix.Columns(); ++i) {
        std::swap(matrix(first, i), matrix(second, i));
    }
}

template <utils::MutableMatrixType M>
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

template <utils::MatrixType M>
inline Matrix<typename M::ElemType> GetTruncSingular(const M &S_full) {
    using T = typename M::ElemType;

    IndexType zero_idx = 0;
    for (zero_idx = 0; zero_idx < std::min(S_full.Rows(), S_full.Columns());
         ++zero_idx) {
        if (utils::IsZeroFloating(S_full(zero_idx, zero_idx)))
            break;
    }

    Matrix<T> S(1, zero_idx);
    for (IndexType i = 0; i < zero_idx; ++i) {
        S(0, i) = S_full(i, i);
    }

    return S;
}
} // namespace details

template <utils::MatrixType M>
    requires utils::details::FloatingPoint<typename M::ElemType>
inline details::SingularBasis<typename M::ElemType> SVD(const M &matrix) {
    using T = typename M::ElemType;

    if (matrix.Rows() < matrix.Columns()) {
        auto [U, S, VT] = SVD(Matrix<typename M::ElemType>::Transposed(matrix));
        U.Transpose();
        VT.Transpose();
        return {std::move(VT), std::move(S), std::move(U)};
    }

    auto [U1, B, VT1] = Bidiagonalize(matrix);
    auto [U2, S_full, VT2] = BidiagAlgorithmQR(B);

    auto VT = VT2 * VT1;
    auto U = U1 * U2;

    details::ToPositiveSingular(S_full, VT);
    details::SortSingular(U, S_full, VT);

    auto S = details::GetTruncSingular(S_full);
    return {std::move(U), std::move(S), std::move(VT)};
}
} // namespace matrix_lib::algorithms
