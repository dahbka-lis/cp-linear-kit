#pragma once

#include "householder.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T>
void ToBidiagonalForm(Matrix<T> &matrix) {
    for (IndexType col = 0; col < std::min(matrix.Rows(), matrix.Columns());
         ++col) {
        auto col_reduction =
            matrix.GetSubmatrix(col, matrix.Rows(), col, col + 1).Copy();

        HouseholderReduction(col_reduction);
        HouseholderLeftReflection(matrix, col_reduction, col, col);

        if (col + 2 < matrix.Columns()) {
            auto row_reduction =
                matrix.GetSubmatrix(col, col + 1, col + 1, matrix.Columns())
                    .Copy();

            HouseholderReduction(row_reduction);
            HouseholderRightReflection(matrix, row_reduction, col + 1, col);
        }
    }
}

template <utils::FloatOrComplex T>
Matrix<T> GetBidiagonalForm(const Matrix<T> &matrix) {
    auto result = matrix;
    ToBidiagonalForm(result);
    return result;
}
} // namespace matrix_lib::algorithms
