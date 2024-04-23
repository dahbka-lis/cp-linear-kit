#pragma once

#include "../types/types_details.h"
#include "is_matrix_type.h"

namespace LinearKit::MatrixUtils {
using IndexType = LinearKit::Details::Types::IndexType;

template <MatrixType M>
Matrix<typename M::ElemType> Join(const M &first, const M &second) {
    using T = typename M::ElemType;
    Matrix<T> result(first.Rows() + second.Rows(),
                     first.Columns() + second.Columns());

    for (IndexType i = 0; i < first.Rows(); ++i) {
        for (IndexType j = 0; j < first.Columns(); ++j) {
            result(i, j) = first(i, j);
        }
    }

    for (IndexType i = 0; i < second.Rows(); ++i) {
        for (IndexType j = 0; j < second.Columns(); ++j) {
            result(i + first.Rows(), j + first.Columns()) = second(i, j);
        }
    }

    return result;
}
} // namespace LinearKit::Utils
