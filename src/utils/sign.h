#pragma once

#include "are_equal_floating.h"
#include "is_float_complex.h"

namespace LinearKit::Utils {
template <Utils::FloatOrComplex T = long double>
T Sign(T value) {
    if constexpr (Utils::Details::IsFloatComplexT<T>::value) {
        if (Utils::IsZeroFloating(value)) {
            return T{1};
        }
        return value / std::sqrt(std::norm(value));
    } else {
        return (value >= T{0}) ? T{1} : T{-1};
    }
}
} // namespace LinearKit::Utils
