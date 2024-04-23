#pragma once

#include "../utils/is_float_complex.h"
#include "is_matrix_type.h"

namespace LinearKit::MatrixUtils {
template <Utils::FloatOrComplex T, MatrixType M>
Matrix<T> CastMatrix(const M &matrix) {
    using F = typename M::ElemType;

    Matrix<T> result(matrix.Rows(), matrix.Columns());
    result.ApplyForEach([&](T &val, auto i, auto j) {
        if constexpr (Utils::Details::IsFloatComplexT<F>::value) {
            val = T{matrix(i, j).real()};
        } else {
            val = T{matrix(i, j)};
        }
    });

    return result;
}
} // namespace LinearKit::MatrixUtils
