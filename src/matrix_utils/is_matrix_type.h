#pragma once

#include "../types/types_details.h"

namespace LinearKit::MatrixUtils {
namespace Details {
template <typename T>
struct IsMatrixT : std::false_type {};

template <Utils::FloatOrComplex T>
struct IsMatrixT<Matrix<T>> : std::true_type {};

template <Utils::FloatOrComplex T>
struct IsMatrixT<MatrixView<T>> : std::true_type {};

template <Utils::FloatOrComplex T>
struct IsMatrixT<ConstMatrixView<T>> : std::true_type {};

template <typename T>
struct IsMutableMatrixT : std::false_type {};

template <Utils::FloatOrComplex T>
struct IsMutableMatrixT<Matrix<T>> : std::true_type {};

template <Utils::FloatOrComplex T>
struct IsMutableMatrixT<MatrixView<T>> : std::true_type {};
} // namespace Details

template <typename T>
concept MatrixType = Details::IsMatrixT<std::remove_cv_t<T>>::value;

template <typename T>
concept MutableMatrixType =
    Details::IsMutableMatrixT<std::remove_cv_t<T>>::value;
} // namespace LinearKit::MatrixUtils
