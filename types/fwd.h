#pragma once

#include "../utils/is_float_complex.h"

namespace matrix_lib {
template <utils::FloatOrComplex T>
class Matrix;

template <utils::FloatOrComplex T>
class MatrixView;

template <utils::FloatOrComplex T>
class ConstMatrixView;
} // namespace matrix_lib
