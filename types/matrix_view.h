#pragma once

#include "../utils/is_equal_floating.h"
#include "../utils/is_float_complex.h"
#include "const_matrix_view.h"
#include "fwd.h"

#include <cassert>
#include <functional>
#include <utility>
#include <vector>

namespace matrix_lib {
template <utils::FloatOrComplex T = long double>
class MatrixView {
    using IndexType = std::ptrdiff_t;
    using Segment = ConstMatrixView<T>::Segment;
    using Function = std::function<void(T &)>;
    using FunctionIndexes = std::function<void(T &, IndexType, IndexType)>;
    using ConstFunction = ConstMatrixView<T>::ConstFunction;
    using ConstFunctionIndexes = ConstMatrixView<T>::ConstFunctionIndexes;

public:
    explicit MatrixView(Matrix<T> &matrix, Segment row = {-1, -1},
                        Segment col = {-1, -1})
        : ptr_(&matrix),
          row_(ConstMatrixView<T>::MakeSegment(row, matrix.Rows())),
          column_(ConstMatrixView<T>::MakeSegment(col, matrix.Columns())) {}

    MatrixView(const MatrixView &rhs) = default;

    MatrixView(MatrixView &&rhs) noexcept
        : ptr_(std::exchange(rhs.ptr_, nullptr)),
          row_(std::exchange(rhs.row_, {0, 1})),
          column_(std::exchange(rhs.column_, {0, 1})){};

    MatrixView &operator=(const MatrixView &lhs) = default;

    MatrixView &operator=(MatrixView &&rhs) noexcept {
        ptr_ = std::exchange(rhs.ptr_, nullptr);
        row_ = std::exchange(rhs.row_, {0, 1});
        column_ = std::exchange(rhs.column_, {0, 1});
        return *this;
    }

    // operators+
    MatrixView &operator+=(const ConstMatrixView<T> &rhs) {
        assert(Rows() == rhs.Rows() && Columns() == rhs.Columns() &&
               "Matrices must be of the same size for addition.");

        ApplyToEach(
            [&](T &value, IndexType i, IndexType j) { value += rhs(i, j); });
        return *this;
    }
    MatrixView &operator+=(const MatrixView &rhs) {
        return *this += rhs.ConstView();
    }
    MatrixView &operator+=(const Matrix<T> &rhs) { return *this += rhs.View(); }
    friend Matrix<T> operator+(const MatrixView<T> &lhs,
                               const MatrixView<T> &rhs) {
        return lhs.ConstView() + rhs.ConstView();
    }
    friend Matrix<T> operator+(const MatrixView<T> &lhs,
                               const ConstMatrixView<T> &rhs) {
        return rhs + lhs;
    }
    friend Matrix<T> operator+(const MatrixView<T> &lhs, const Matrix<T> &rhs) {
        return lhs.ConstView() + rhs.View();
    }
    // - - - - -

    // operators-
    MatrixView &operator-=(const ConstMatrixView<T> &rhs) {
        assert(Rows() == rhs.Rows() && Columns() == rhs.Columns() &&
               "Matrices must be of the same size for addition.");

        ApplyToEach(
            [&](T &value, IndexType i, IndexType j) { value -= rhs(i, j); });
        return *this;
    }
    MatrixView &operator-=(const MatrixView &rhs) {
        return *this -= rhs.ConstView();
    }
    MatrixView &operator-=(const Matrix<T> &rhs) { return *this -= rhs.View(); }
    friend Matrix<T> operator-(const MatrixView<T> &lhs,
                               const MatrixView<T> &rhs) {
        return lhs.ConstView() - rhs.ConstView();
    }
    friend Matrix<T> operator-(const MatrixView<T> &lhs,
                               const ConstMatrixView<T> &rhs) {
        return lhs.ConstView() - rhs;
    }
    friend Matrix<T> operator-(const MatrixView<T> &lhs, const Matrix<T> &rhs) {
        return lhs.ConstView() - rhs.View();
    }
    // - - - - -

    // operators*
    MatrixView &operator*=(const ConstMatrixView<T> &rhs) {
        assert(rhs.Columns() == rhs.Rows() &&
               "Matrix must be square for multiplication.");
        auto res = *this * rhs;
        ApplyToEach([&](T &val, IndexType i, IndexType j) { val = res(i, j); });
        return *this;
    }
    MatrixView &operator*=(const MatrixView &rhs) {
        return *this *= rhs.ConstView();
    }
    MatrixView &operator*=(const Matrix<T> &rhs) { return *this *= rhs.View(); }
    friend Matrix<T> operator*(const MatrixView &lhs, const MatrixView &rhs) {
        return lhs * rhs.ConstView();
    }
    friend Matrix<T> operator*(const MatrixView &lhs,
                               const ConstMatrixView<T> &rhs) {
        return lhs.ConstView() * rhs;
    }
    friend Matrix<T> operator*(const MatrixView &lhs, const Matrix<T> &rhs) {
        return lhs * rhs.View();
    }
    // - - - - -

    T &operator()(IndexType row_idx, IndexType col_idx) {
        assert(row_idx >= 0 && row_idx < Rows() && "Invalid row index.");
        assert(col_idx >= 0 && col_idx < Columns() && "Invalid column index.");
        return (*ptr_)(row_.begin + row_idx, column_.begin + col_idx);
    }

    T operator()(IndexType row_idx, IndexType col_idx) const {
        assert(row_idx >= 0 && row_idx < Rows() && "Invalid row index.");
        assert(col_idx >= 0 && col_idx < Columns() && "Invalid column index.");
        return (*ptr_)(row_.begin + row_idx, column_.begin + col_idx);
    }

    [[nodiscard]] IndexType Rows() const { return row_.end - row_.begin; }

    [[nodiscard]] IndexType Columns() const {
        return column_.end - column_.begin;
    }

    MatrixView &ApplyToEach(Function func) {
        for (IndexType i = row_.begin; i < row_.end; ++i) {
            for (IndexType j = column_.begin; j < column_.end; ++j) {
                func((*ptr_)(i, j));
            }
        }

        return *this;
    }

    MatrixView &ApplyToEach(FunctionIndexes func) {
        for (IndexType i = row_.begin; i < row_.end; ++i) {
            for (IndexType j = column_.begin; j < column_.end; ++j) {
                func((*ptr_)(i, j), i - row_.begin, j - column_.begin);
            }
        }

        return *this;
    }

    MatrixView &ApplyToEach(ConstFunction func) const {
        ConstView().ApplyToEach(func);
        return *this;
    }

    MatrixView &ApplyToEach(ConstFunctionIndexes func) const {
        ConstView().ApplyToEach(func);
        return *this;
    }

    T GetEuclideanNorm() const {
        return ConstView().GetEuclideanNorm();
    }

    Matrix<T> GetDiag() const {
        return ConstView().GetDiag();
    }

    MatrixView GetRow(IndexType index) {
        assert(index < Rows() &&
               "Index must be less than the number of matrix rows.");

        return MatrixView(*ptr_, {row_.begin + index, row_.begin + index + 1},
                          {column_.begin, column_.end});
    }

    MatrixView GetColumn(IndexType index) {
        assert(index < Columns() &&
               "Index must be less than the number of matrix columns.");

        return MatrixView(*ptr_, {row_.begin, row_.end},
                          {column_.begin + index, column_.begin + index + 1});
    }

    MatrixView<T> GetSubmatrix(Segment row, Segment col) {
        auto [r_from, r_to] = row;
        auto [c_from, c_to] = col;

        assert(
            r_from >= 0 && r_to <= Rows() &&
            "The row indices do not match the number of rows in the matrix.");
        assert(r_from < r_to && "The row index for the start of the submatrix "
                                "must be less than the end index.");
        assert(c_from >= 0 && c_to <= Columns() &&
               "The column indices do not match the number of columns in the "
               "matrix.");
        assert(c_from < c_to && "The column index for the start of the "
                                "submatrix must be less than the end index.");

        return MatrixView<T>(*ptr_, {row_.begin + r_from, row_.begin + r_to},
                             {column_.begin + c_from, column_.begin + c_to});
    }

    ConstMatrixView<T> ConstView() const {
        return ConstMatrixView<T>(*ptr_, {row_.begin, row_.end},
                                  {column_.begin, column_.end});
    }

    static Matrix<T> Transposed(const MatrixView &rhs) {
        return Matrix<T>::Transposed(rhs.Copy());
    }

    static Matrix<T> Conjugated(const MatrixView &rhs) {
        return Matrix<T>::Conjugated(rhs.Copy());
    }

    static Matrix<T> Normalized(const MatrixView &rhs) {
        return Matrix<T>::Normalized(rhs.Copy());
    }

    static Matrix<T> Identity(IndexType size) {
        return Matrix<T>::Identity(size);
    }

    friend std::ostream &operator<<(std::ostream &ostream,
                                    const MatrixView &matrix) {
        ostream << '{';
        for (std::size_t i = 0; i < matrix.Rows(); ++i) {
            ostream << '{';
            for (std::size_t j = 0; j < matrix.Columns(); ++j) {
                ostream << matrix(i, j);
                if (j + 1 < matrix.Columns()) {
                    ostream << ' ';
                }
            }

            ostream << '}';
            if (i + 1 < matrix.Rows()) {
                ostream << '\n';
            }
        }
        ostream << '}';
        return ostream;
    }

private:
    Matrix<T> *ptr_;
    Segment row_;
    Segment column_;
};
} // namespace matrix_lib
