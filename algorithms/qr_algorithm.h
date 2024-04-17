#pragma once

#include "qr_decomposition.h"

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
} // namespace matrix_lib::algorithms
