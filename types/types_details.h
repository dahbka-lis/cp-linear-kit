#pragma once

#include "../utils/is_equal_floating.h"
#include "../utils/is_float_complex.h"

#include <cassert>
#include <functional>
#include <utility>

namespace LinearKit {
template <Utils::FloatOrComplex T>
class Matrix;

template <Utils::FloatOrComplex T>
class MatrixView;

template <Utils::FloatOrComplex T>
class ConstMatrixView;

namespace Details {
struct Types {
    using IndexType = std::ptrdiff_t;

    template <Utils::FloatOrComplex T>
    using Function = std::function<void(T &)>;

    template <Utils::FloatOrComplex T>
    using FunctionIndexes = std::function<void(T &, IndexType, IndexType)>;

    template <Utils::FloatOrComplex T>
    using ConstFunction = std::function<void(const T &)>;

    template <Utils::FloatOrComplex T>
    using ConstFunctionIndexes =
        std::function<void(const T &, IndexType, IndexType)>;

    struct Segment {
        IndexType begin = 0;
        IndexType end = 1;
    };

    struct MatrixState {
        bool is_transposed = false;
        bool is_conjugated = false;
    };
};
} // namespace Details
} // namespace LinearKit
