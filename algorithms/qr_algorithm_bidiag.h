#pragma once

#include "../matrix_utils/checks.h"
#include "../matrix_utils/join.h"
#include "../matrix_utils/split.h"
#include "givens.h"
#include "qr_decomposition.h"

namespace matrix_lib::algorithms {
namespace details {
template <utils::FloatOrComplex T>
struct DiagBasisQR {
    Matrix<T> U;
    Matrix<T> D;
    Matrix<T> VT;
};
} // namespace details

template <utils::MatrixType M>
inline details::DiagBasisQR<typename M::ElemType>
BidiagAlgorithmQR(const M &B, IndexType it_cnt = 100);

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
    auto BB = Matrix<T>::Conjugated(minor) * minor;

    if (r >= 3) {
        BB(0, 0) += S(r - 3, c - 2) * S(r - 3, c - 2);
    }

    return GetWilkinsonShift(BB);
}

template <utils::MatrixType M>
inline typename M::ElemType GetBidiagThreshold(const M &matrix) {
    using T = typename M::ElemType;
    auto size = std::min(matrix.Rows(), matrix.Columns() - 1);

    T threshold = 0;
    for (IndexType i = 0; i < size; ++i) {
        auto abs_sum = std::abs(matrix(i, i)) + std::abs(matrix(i, i + 1));
        threshold = std::max(threshold, abs_sum);
    }

    return threshold;
}

template <utils::MatrixType M>
inline details::DiagBasisQR<typename M::ElemType> SplitBidiagQR(const M &D,
                                                                IndexType idx) {
    auto [D1, D2] = utils::Split(D, idx, idx);
    auto [U1, S1, VT1] = BidiagAlgorithmQR(D1);
    auto [U2, S2, VT2] = BidiagAlgorithmQR(D2);

    auto U = utils::Join(U1, U2);
    auto S = utils::Join(S1, S2);
    auto VT = utils::Join(VT1, VT2);

    return {std::move(U), std::move(S), std::move(VT)};
}

template <utils::MutableMatrixType M>
inline details::DiagBasisQR<typename M::ElemType>
CancellationBidiagQR(M &D, M &U, IndexType idx) {
    for (IndexType k = idx + 1; k < std::min(D.Columns(), D.Rows()); ++k) {
        auto remove = D(idx, k);
        auto next = D(k, k);

        algorithms::GivensLeftRotation(D, k, idx, next, remove);
        algorithms::GivensRightRotation(U, k, idx, next, remove);
    }

    auto [Us, S, VT] = SplitBidiagQR(D, idx);
    return {std::move(U * Us), std::move(S), std::move(VT)};
}

template <utils::MutableMatrixType M>
inline void StepBidiagQR(M &U, M &D, M &VT) {
    using T = typename M::ElemType;

    T shift = GetBidiagWilkinsonShift(D);
    auto size = std::min(D.Rows(), D.Columns());

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
}
} // namespace details

template <utils::MatrixType M>
inline details::DiagBasisQR<typename M::ElemType>
BidiagAlgorithmQR(const M &B, IndexType it_cnt) {
    using T = typename M::ElemType;

    Matrix<T> D = B;
    Matrix<T> U = Matrix<T>::Identity(D.Rows());
    Matrix<T> VT = Matrix<T>::Identity(D.Columns());

    if (D.Columns() == 1) {
        return {std::move(U), std::move(D), std::move(VT)};
    }

    auto threshold = details::GetBidiagThreshold(D);
    auto eps = utils::Eps<T> * threshold;

    for (IndexType i = 0; i < std::min(D.Rows(), D.Columns() - 1); ++i) {
        if (std::abs(D(i, i + 1)) < eps) {
            return details::SplitBidiagQR(D, i);
        }

        if (std::abs(D(i, i)) < eps) {
            return details::CancellationBidiagQR(D, U, i);
        }
    }

    it_cnt *= D.Columns();
    while (--it_cnt) {
        details::StepBidiagQR(U, D, VT);
        D.RoundZeroes(eps);
    }

    return {std::move(U), std::move(D), std::move(VT)};
}
} // namespace matrix_lib::algorithms
