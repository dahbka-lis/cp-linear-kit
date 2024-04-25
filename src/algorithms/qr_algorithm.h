#pragma once

#include "hessenberg.h"
#include "qr_decomposition.h"

namespace LinearKit::Algorithm {
namespace Details {
template <Utils::FloatOrComplex T = long double>
struct SpectralPair {
    Matrix<T> D;
    Matrix<T> U;
};
} // namespace Details

template <MatrixUtils::MatrixType M>
inline Details::SpectralPair<typename M::ElemType>
GetSpecDecomposition(const M &matrix,
                     typename M::ElemType shift = typename M::ElemType{0},
                     std::size_t it_cnt = 50) {
    using T = typename M::ElemType;

    assert(MatrixUtils::IsSymmetric(matrix) &&
           "Spectral decomposition for symmetric matrices.");

    auto [D, U] = GetHessenbergForm(matrix);
    for (IndexType i = 0; i < it_cnt * D.Rows(); ++i) {
        if constexpr (Utils::Details::IsFloatComplexT<T>::value) {
            if (MatrixUtils::IsUpperTriangular(D))
                break;
        } else {
            if (MatrixUtils::IsDiagonal(D))
                break;
        }

        auto shift_I = Matrix<T>::Identity(D.Rows()) * shift;
        auto [Q, R] = HouseholderQR(D - shift_I);
        D = R * Q + shift_I;
        U *= Q;
    }

    D.RoundZeroes();
    return {std::move(D), std::move(U)};
}
} // namespace LinearKit::Algorithm
