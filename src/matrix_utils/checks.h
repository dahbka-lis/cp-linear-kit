#pragma once

#include "is_matrix_type.h"

namespace LinearKit::MatrixUtils {
using IndexType = LinearKit::Details::Types::IndexType;

template <MatrixType F, MatrixType S>
bool AreEqualMatrices(const F &first, const S &second,
                      typename F::ElemType eps = typename F::ElemType{0}) {
    using T = F::ElemType;

    if (first.Rows() != second.Rows() || first.Columns() != second.Columns()) {
        return false;
    }

    for (IndexType i = 0; i < first.Rows(); ++i) {
        for (IndexType j = 0; j < first.Columns(); ++j) {
            if (!Utils::AreEqualFloating(first(i, j), second(i, j), eps)) {
                return false;
            }
        }
    }

    return true;
}

template <MatrixType M>
bool IsSquare(const M &matrix) {
    return matrix.Rows() == matrix.Columns();
}

template <MatrixType M>
bool IsUnitary(const M &matrix,
               typename M::ElemType eps = typename M::ElemType{0}) {
    using T = M::ElemType;

    if (!IsSquare(matrix)) {
        return false;
    }

    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        auto column = matrix.GetColumn(i);
        if (!Utils::AreEqualFloating(column.GetEuclideanNorm(), T{1}, eps)) {
            return false;
        }
    }

    return true;
}

template <MatrixType M>
bool IsSymmetric(const M &matrix,
                 typename M::ElemType eps = typename M::ElemType{0}) {
    if (!IsSquare(matrix)) {
        return false;
    }

    for (IndexType i = 1; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < i; ++j) {
            if (!Utils::AreEqualFloating(matrix(i, j), matrix(j, i), eps)) {
                return false;
            }
        }
    }

    return true;
}

template <MatrixType M>
bool IsHermitian(const M &matrix,
                 typename M::ElemType eps = typename M::ElemType{0}) {
    using T = typename M::ElemType;

    if (!IsSquare(matrix)) {
        return false;
    }

    if constexpr (Utils::Details::IsFloatComplexT<T>::value) {
        for (IndexType i = 0; i < matrix.Rows(); ++i) {
            for (IndexType j = 0; j <= i; ++j) {
                if (!Utils::AreEqualFloating(matrix(i, j),
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

template <MatrixType M>
bool IsNormal(const M &matrix,
              typename M::ElemType eps = typename M::ElemType{0}) {
    using T = M::ElemType;

    auto m1 = matrix * Matrix<T>::Conjugated(matrix);
    auto m2 = Matrix<T>::Conjugated(matrix) * matrix;
    return AreEqualMatrices(m1, m2, eps);
}

template <MatrixType M>
bool IsUpperTriangular(const M &matrix,
                       typename M::ElemType eps = typename M::ElemType{0}) {
    for (IndexType i = 1; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < std::min(i, matrix.Columns()); ++j) {
            if (!Utils::IsZeroFloating(matrix(i, j), eps)) {
                return false;
            }
        }
    }

    return true;
}

template <MatrixType M>
bool IsDiagonal(const M &matrix,
                typename M::ElemType eps = typename M::ElemType{0}) {
    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < matrix.Columns(); ++j) {
            if (i != j && !Utils::IsZeroFloating(matrix(i, j), eps)) {
                return false;
            }
        }
    }

    return true;
}

template <MatrixType M>
bool IsBidiagonal(const M &matrix,
                  typename M::ElemType eps = typename M::ElemType{0}) {
    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < matrix.Columns(); ++j) {
            if (i != j && i + 1 != j &&
                !Utils::IsZeroFloating(matrix(i, j), eps)) {
                return false;
            }
        }
    }

    return true;
}

template <MatrixType M>
bool IsHessenberg(const M &matrix,
                  typename M::ElemType eps = typename M::ElemType{0}) {
    for (IndexType i = 2; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < i - 1; ++j) {
            if (!Utils::IsZeroFloating(matrix(i, j), eps)) {
                return false;
            }
        }
    }

    return true;
}
} // namespace LinearKit::MatrixUtils
