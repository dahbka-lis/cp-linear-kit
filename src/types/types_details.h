#pragma once

#include "../utils/are_equal_floating.h"
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

    enum class TransposeState { Normal, Transposed };

    enum class ConjugateState { Normal, Conjugated };

    struct MatrixState {
        TransposeState is_transposed = TransposeState::Normal;
        ConjugateState is_conjugated = ConjugateState::Normal;
    };

    static TransposeState SwitchState(TransposeState state) {
        return state == TransposeState::Normal ? TransposeState::Transposed
                                               : TransposeState::Normal;
    }

    static ConjugateState SwitchState(ConjugateState state) {
        return state == ConjugateState::Normal ? ConjugateState::Conjugated
                                               : ConjugateState::Normal;
    }
};
} // namespace Details
} // namespace LinearKit
