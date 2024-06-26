#pragma once

#include "../matrix_utils/checks.h"
#include "../matrix_utils/join.h"
#include "../matrix_utils/split.h"
#include "givens.h"
#include "qr_decomposition.h"
#include "wilkinson.h"

namespace LinearKit::Algorithm {
namespace Details {
template <Utils::FloatOrComplex T>
struct DiagBasisQR {
    Matrix<T> U;
    Matrix<T> D;
    Matrix<T> VT;
};
} // namespace Details

template <MatrixUtils::MatrixType M>
Details::DiagBasisQR<typename M::ElemType>
BidiagAlgorithmQR(const M &B, IndexType it_cnt = 10);

namespace Details {
template <MatrixUtils::MatrixType M>
typename M::ElemType GetBidiagThreshold(const M &matrix) {
    using T = typename M::ElemType;

    long double threshold = 0;
    for (IndexType i = 0; i < matrix.Columns() - 1; ++i) {
        long double abs_sum =
            std::abs(matrix(i, i)) + std::abs(matrix(i, i + 1));
        threshold = std::max(threshold, abs_sum);
    }

    return threshold;
}

template <MatrixUtils::MatrixType M>
Details::DiagBasisQR<typename M::ElemType> SplitBidiagQR(const M &D,
                                                         IndexType idx) {
    auto [D1, D2] = MatrixUtils::Split(D, idx, idx);
    auto [U1, S1, VT1] = BidiagAlgorithmQR(D1);
    auto [U2, S2, VT2] = BidiagAlgorithmQR(D2);

    auto U = MatrixUtils::Join(U1, U2);
    auto S = MatrixUtils::Join(S1, S2);
    auto VT = MatrixUtils::Join(VT1, VT2);

    return {std::move(U), std::move(S), std::move(VT)};
}

template <MatrixUtils::MutableMatrixType M>
Details::DiagBasisQR<typename M::ElemType> CancellationBidiagQR(M &D, M &U,
                                                                IndexType idx) {
    for (IndexType k = idx + 1; k < std::min(D.Columns(), D.Rows()); ++k) {
        auto remove = D(idx, k);
        auto next = D(k, k);

        Algorithm::GivensLeftRotation(D, k, idx, next, remove);
        Algorithm::GivensRightRotation(U, k, idx, next, remove);
    }

    auto [Us, S, VT] = SplitBidiagQR(D, idx);
    return {std::move(U * Us), std::move(S), std::move(VT)};
}

template <MatrixUtils::MutableMatrixType M>
void StepBidiagQR(M &U, M &D, M &VT) {
    using T = typename M::ElemType;
    T shift = GetBidiagWilkinsonShift(D);

    for (IndexType i = 0; i < D.Columns() - 1; ++i) {
        auto f_elem = (i > 0) ? D(i - 1, i) : D(0, 0) * D(0, 0) - shift;
        auto s_elem = (i > 0) ? D(i - 1, i + 1) : D(0, 1) * D(0, 0);

        GivensLeftRotation(VT, i, i + 1, f_elem, s_elem);
        GivensRightRotation(D, i, i + 1, f_elem, s_elem);

        GivensRightRotation(U, i, i + 1, D(i, i), D(i + 1, i));
        GivensLeftRotation(D, i, i + 1, D(i, i), D(i + 1, i));
    }
}
} // namespace Details

template <MatrixUtils::MatrixType M>
Details::DiagBasisQR<typename M::ElemType> BidiagAlgorithmQR(const M &B,
                                                             IndexType it_cnt) {
    using T = typename M::ElemType;

    Matrix<T> D = B;
    Matrix<T> U = Matrix<T>::Identity(D.Rows());
    Matrix<T> VT = Matrix<T>::Identity(D.Columns());

    if (D.Columns() <= 1) {
        return {std::move(U), std::move(D), std::move(VT)};
    }

    it_cnt *= D.Columns();
    while (--it_cnt) {
        auto threshold = Details::GetBidiagThreshold(D);
        auto eps = Utils::Eps<T> * threshold;

        if (MatrixUtils::IsDiagonal(D, eps)) {
            break;
        }

        for (IndexType i = 0; i < D.Columns() - 1; ++i) {
            if (std::abs(D(i, i)) <= eps) {
                auto [Uc, Sc, VTc] = Details::CancellationBidiagQR(D, U, i);
                Sc.RoundZeroes(eps);
                return {std::move(Uc), std::move(Sc), std::move(VTc * VT)};
            }
        }

        for (IndexType i = 0; i < std::min(D.Rows(), D.Columns() - 1); ++i) {
            if (std::abs(D(i, i + 1)) <= eps) {
                auto [U_split, S_split, VT_split] =
                    Details::SplitBidiagQR(D, i);
                S_split.RoundZeroes();
                return {std::move(U * U_split), std::move(S_split),
                        std::move(VT_split * VT)};
            }
        }

        Details::StepBidiagQR(U, D, VT);
    }

    D.RoundZeroes();
    return {std::move(U), std::move(D), std::move(VT)};
}
} // namespace LinearKit::Algorithm
