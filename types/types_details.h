#pragma once

#include "../utils/is_equal_floating.h"
#include "../utils/is_float_complex.h"

#include <cassert>
#include <functional>
#include <utility>

namespace matrix_lib {
template <utils::FloatOrComplex T>
class Matrix;

template <utils::FloatOrComplex T>
class MatrixView;

template <utils::FloatOrComplex T>
class ConstMatrixView;

namespace details {
struct Types {
    using IndexType = std::ptrdiff_t;

    template <utils::FloatOrComplex T>
    using Function = std::function<void(T &)>;

    template <utils::FloatOrComplex T>
    using FunctionIndexes = std::function<void(T &, IndexType, IndexType)>;

    template <utils::FloatOrComplex T>
    using ConstFunction = std::function<void(const T &)>;

    template <utils::FloatOrComplex T>
    using ConstFunctionIndexes =
        std::function<void(const T &, IndexType, IndexType)>;

    struct Segment {
        IndexType begin = 0;
        IndexType end = 1;
    };
};
} // namespace details
} // namespace matrix_lib
