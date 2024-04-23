#pragma once

#include "hessenberg.h"
#include "qr_decomposition.h"
#include "wilkinson.h"
#include <iostream>

namespace matrix_lib::algorithms {
namespace details {
template <utils::FloatOrComplex T = long double>
struct SpectralPair {
    Matrix<T> D;
    Matrix<T> U;
};
} // namespace details

template <utils::MatrixType M>
inline details::SpectralPair<typename M::ElemType>
GetSpecDecomposition(const M &matrix,
                     typename M::ElemType shift = typename M::ElemType{0},
                     std::size_t it_cnt = 50) {
    using T = typename M::ElemType;

    assert(utils::IsHermitian(matrix) &&
           "Spectral decomposition for hermitian matrix.");

    auto [D, U] = GetHessenbergForm(matrix);
    for (IndexType i = 0; i < it_cnt * D.Rows(); ++i) {
        if constexpr (utils::details::IsFloatComplexT<T>::value) {
            if (utils::IsUpperTriangular(D))
                break;
        } else {
            if (utils::IsDiagonal(D))
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
} // namespace matrix_lib::algorithms
