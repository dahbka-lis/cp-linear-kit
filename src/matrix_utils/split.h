#pragma once

#include "../types/types_details.h"
#include "is_matrix_type.h"

namespace LinearKit::MatrixUtils {
namespace Details {
template <Utils::FloatOrComplex T>
struct SplitPair {
    Matrix<T> first;
    Matrix<T> second;
};
} // namespace Details

using IndexType = LinearKit::Details::Types::IndexType;

template <MatrixType M>
Details::SplitPair<typename M::ElemType> Split(const M &matrix, IndexType row,
                                               IndexType col) {
    using T = typename M::ElemType;

    assert(row >= 0 && row < matrix.Rows() && "Wrong row index.");
    assert(col >= 0 && col < matrix.Columns() && "Wrong column index.");

    Matrix<T> first = matrix.GetSubmatrix({0, row + 1}, {0, col + 1});
    Matrix<T> second = matrix.GetSubmatrix({row + 1, matrix.Rows()},
                                           {col + 1, matrix.Columns()});

    return {first, second};
}
} // namespace LinearKit::MatrixUtils
