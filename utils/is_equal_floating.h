#pragma once

#include "is_float_complex.h"

#include <cmath>
#include <limits>

namespace matrix_lib::utils {
namespace details {
template <FloatOrComplex T>
struct TypeEpsilon {
    static constexpr T kValue = 1e-6;
};

template <>
struct TypeEpsilon<double> {
    static constexpr double kValue = 1e-10;
};

template <>
struct TypeEpsilon<long double> {
    static constexpr long double kValue = 1e-15;
};

template <typename T>
struct TypeEpsilon<std::complex<T>> {
    static constexpr T kValue = TypeEpsilon<T>::kValue;
};
} // namespace details

template <FloatOrComplex T>
static constexpr auto Eps = details::TypeEpsilon<T>::kValue;

template <FloatOrComplex T = long double>
inline bool IsEqualFloating(T lhs, T rhs, T eps = T{0}) {
    if (eps == T{0}) {
        eps = Eps<T>;
    }

    if constexpr (details::IsFloatComplexT<T>::value) {
        auto is_equal_real =
            std::abs(std::abs(lhs.real()) - std::abs(rhs.real())) < eps.real();
        auto is_equal_imag =
            std::abs(std::abs(lhs.imag()) - std::abs(rhs.imag())) < eps.real();
        return is_equal_real && is_equal_imag;
    } else {
        return std::abs(std::abs(lhs) - std::abs(rhs)) < eps;
    }
}

template <FloatOrComplex T = long double>
inline bool IsZeroFloating(T lhs, T eps = T{0}) {
    return IsEqualFloating<T>(lhs, T{0}, eps);
}
} // namespace matrix_lib::utils
