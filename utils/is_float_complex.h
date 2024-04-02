#pragma once

#include <complex>
#include <type_traits>

namespace matrix_lib::utils {
namespace details {
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template <typename T>
struct IsFloatComplexT : std::false_type {};

template <FloatingPoint T>
struct IsFloatComplexT<std::complex<T>> : std::true_type {};
} // namespace details

template <typename T>
concept FloatOrComplex = details::FloatingPoint<std::remove_cv_t<T>> ||
                         details::IsFloatComplexT<std::remove_cv_t<T>>::value;
} // namespace matrix_lib::utils
