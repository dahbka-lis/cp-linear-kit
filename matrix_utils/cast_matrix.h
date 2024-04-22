#pragma once

#include "../types/types_details.h"
#include "../utils/is_float_complex.h"
#include "is_matrix_type.h"

namespace matrix_lib::utils {
template <utils::FloatOrComplex T, utils::MatrixType M>
Matrix<T> CastMatrix(const M &matrix) {
    using F = typename M::ElemType;

    Matrix<T> result(matrix.Rows(), matrix.Columns());
    result.ApplyForEach([&](T &val, auto i, auto j) {
        if constexpr (utils::details::IsFloatComplexT<F>::value) {
            val = T{matrix(i, j).real()};
        } else {
            val = T{matrix(i, j)};
        }
    });

    return result;
}
} // namespace matrix_lib::utils
