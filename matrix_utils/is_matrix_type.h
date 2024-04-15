#pragma once

#include "../types/types_details.h"

namespace matrix_lib::utils {
namespace details {
template <typename T>
struct IsMatrixT : std::false_type {};

template <FloatOrComplex T>
struct IsMatrixT<Matrix<T>> : std::true_type {};

template <FloatOrComplex T>
struct IsMatrixT<MatrixView<T>> : std::true_type {};

template <FloatOrComplex T>
struct IsMatrixT<ConstMatrixView<T>> : std::true_type {};

template <typename T>
struct IsMutableMatrixT : std::false_type {};

template <FloatOrComplex T>
struct IsMutableMatrixT<Matrix<T>> : std::true_type {};

template <FloatOrComplex T>
struct IsMutableMatrixT<MatrixView<T>> : std::true_type {};
} // namespace details

template <typename T>
concept MatrixType = details::IsMatrixT<std::remove_cv_t<T>>::value;

template <typename T>
concept MutableMatrixType =
    details::IsMutableMatrixT<std::remove_cv_t<T>>::value;
} // namespace matrix_lib::utils
