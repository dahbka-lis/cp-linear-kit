#pragma once

#include <complex>
#include <type_traits>

namespace LinearKit::Utils {
namespace Details {
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template <typename T>
struct IsFloatComplexT : std::false_type {};

template <FloatingPoint T>
struct IsFloatComplexT<std::complex<T>> : std::true_type {};
} // namespace Details

template <typename T>
concept FloatOrComplex = Details::FloatingPoint<std::remove_cv_t<T>> ||
                         Details::IsFloatComplexT<std::remove_cv_t<T>>::value;
} // namespace LinearKit::Utils
