#pragma once

#include "../types/types_details.h"
#include "is_matrix_type.h"

namespace matrix_lib::utils {
using IndexType = matrix_lib::details::Types::IndexType;

template <utils::MatrixType F, utils::MatrixType S>
inline bool
AreEqualMatrices(const F &first, const S &second,
                 typename F::ElemType eps = typename F::ElemType{0}) {
    using T = F::ElemType;

    if (first.Rows() != second.Rows() || first.Columns() != second.Columns()) {
        return false;
    }

    for (IndexType i = 0; i < first.Rows(); ++i) {
        for (IndexType j = 0; j < first.Columns(); ++j) {
            if (!utils::IsEqualFloating(first(i, j), second(i, j), eps)) {
                return false;
            }
        }
    }

    return true;
}

template <utils::MatrixType M>
inline bool IsSquare(const M &matrix) {
    return matrix.Rows() == matrix.Columns();
}

template <utils::MatrixType M>
inline bool IsUnitary(const M &matrix,
                      typename M::ElemType eps = typename M::ElemType{0}) {
    using T = M::ElemType;

    if (!IsSquare(matrix)) {
        return false;
    }

    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        auto column = matrix.GetColumn(i);
        if (!utils::IsEqualFloating(column.GetEuclideanNorm(), T{1}, eps)) {
            return false;
        }
    }

    return true;
}

template <utils::MatrixType M>
inline bool IsSymmetric(const M &matrix,
                        typename M::ElemType eps = typename M::ElemType{0}) {
    if (!IsSquare(matrix)) {
        return false;
    }

    for (IndexType i = 1; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < i; ++j) {
            if (!utils::IsEqualFloating(matrix(i, j), matrix(j, i), eps)) {
                return false;
            }
        }
    }

    return true;
}

template <utils::MatrixType M>
inline bool IsHermitian(const M &matrix,
                        typename M::ElemType eps = typename M::ElemType{0}) {
    if (!IsSquare(matrix)) {
        return false;
    }

    if constexpr (details::IsFloatComplexT<typename M::ElemType>::value) {
        for (IndexType i = 0; i < matrix.Rows(); ++i) {
            for (IndexType j = 0; j <= i; ++j) {
                if (!utils::IsEqualFloating(matrix(i, j),
                                            std::conj(matrix(j, i)), eps)) {
                    return false;
                }
            }
        }
    } else {
        return IsSymmetric(matrix, eps);
    }

    return true;
}

template <utils::MatrixType M>
inline bool IsNormal(const M &matrix,
                     typename M::ElemType eps = typename M::ElemType{0}) {
    using T = M::ElemType;

    auto m1 = matrix * Matrix<T>::Conjugated(matrix);
    auto m2 = Matrix<T>::Conjugated(matrix) * matrix;
    return AreEqualMatrices(m1, m2, eps);
}

template <utils::MatrixType M>
inline bool
IsUpperTriangular(const M &matrix,
                  typename M::ElemType eps = typename M::ElemType{0}) {
    for (IndexType i = 1; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < std::min(i, matrix.Columns()); ++j) {
            if (!utils::IsZeroFloating(matrix(i, j), eps)) {
                return false;
            }
        }
    }

    return true;
}

template <utils::MatrixType M>
inline bool IsDiagonal(const M &matrix,
                       typename M::ElemType eps = typename M::ElemType{0}) {
    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < matrix.Columns(); ++j) {
            if (i != j && !utils::IsZeroFloating(matrix(i, j), eps)) {
                return false;
            }
        }
    }

    return true;
}

template <utils::MatrixType M>
inline bool IsBidiagonal(const M &matrix,
                         typename M::ElemType eps = typename M::ElemType{0}) {
    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < matrix.Columns(); ++j) {
            if (i != j && i + 1 != j &&
                !utils::IsZeroFloating(matrix(i, j), eps)) {
                return false;
            }
        }
    }

    return true;
}

template <utils::MatrixType M>
inline bool IsHessenberg(const M &matrix,
                         typename M::ElemType eps = typename M::ElemType{0}) {
    for (IndexType i = 2; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < i - 1; ++j) {
            if (!utils::IsZeroFloating(matrix(i, j), eps)) {
                return false;
            }
        }
    }

    return true;
}
} // namespace matrix_lib::utils
