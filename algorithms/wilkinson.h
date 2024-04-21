#pragma once

#include "../matrix_utils/is_matrix_type.h"
#include "../types/types_details.h"
#include "../utils/sign.h"

namespace matrix_lib::algorithms {
template <utils::MatrixType M>
inline typename M::ElemType GetWilkinsonShift(const M &matrix,
                                              IndexType end_idx) {
    using T = typename M::ElemType;

    if (matrix.Rows() == 0) {
        return T{0};
    }

    assert(end_idx >= 2 && end_idx <= matrix.Rows() && "Wrong end index.");

    auto d =
        (matrix(end_idx - 2, end_idx - 2) - matrix(end_idx - 1, end_idx - 1)) /
        T{2};
    auto tmp =
        matrix(end_idx - 1, end_idx - 2) * matrix(end_idx - 1, end_idx - 2);
    auto coefficient = d + utils::Sign(d) * std::sqrt(d * d + tmp);

    return matrix(end_idx - 1, end_idx - 1) - tmp / coefficient;
}

template <utils::MatrixType M>
inline typename M::ElemType GetBidiagWilkinsonShift(const M &S) {
    using T = typename M::ElemType;
    assert(S.Columns() >= 2 && "Wrong columns count.");

    auto sub_idx = S.Columns();
    Matrix<T> S_gram =
        S.GetSubmatrix({sub_idx - 2, sub_idx}, {sub_idx - 2, sub_idx});

    S_gram.Conjugate();
    S_gram *= S.GetSubmatrix({sub_idx - 2, sub_idx}, {sub_idx - 2, sub_idx});

    if (sub_idx >= 3) {
        S_gram(0, 0) +=
            S(sub_idx - 3, sub_idx - 2) * S(sub_idx - 3, sub_idx - 2);
    }

    return GetWilkinsonShift(S_gram, 2);
}
} // namespace matrix_lib::algorithms
