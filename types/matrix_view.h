#pragma once

#include "const_matrix_view.h"
#include "types_details.h"

namespace matrix_lib {
template <utils::FloatOrComplex T = long double>
class MatrixView {
    using IndexType = details::Types::IndexType;
    using Segment = details::Types::Segment;
    using Function = details::Types::Function<T>;
    using FunctionIndexes = details::Types::FunctionIndexes<T>;
    using ConstFunction = details::Types::ConstFunction<T>;
    using ConstFunctionIndexes = details::Types::ConstFunctionIndexes<T>;

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

    MatrixView &operator*=(T scalar) {
        ApplyToEach([&](T &val) { val *= scalar; });
        return *this;
    }

    friend Matrix<T> operator*(const MatrixView &lhs, T scalar) {
        return lhs.ConstView() * scalar;
    }

    friend Matrix<T> operator*(T scalar, const MatrixView &lhs) {
        return lhs.ConstView() * scalar;
    }

    MatrixView &operator/=(T scalar) {
        ApplyToEach([&](T &val) { val /= scalar; });
        return *this;
    }

    friend Matrix<T> operator/(const MatrixView &lhs, T scalar) {
        return lhs.ConstView() / scalar;
    }

    friend Matrix<T> operator/(T scalar, const MatrixView &lhs) {
        return lhs.ConstView() / scalar;
    }

    friend bool operator==(const MatrixView &lhs, const MatrixView &rhs) {
        return lhs.ConstView() == rhs.ConstView();
    }

    friend bool operator==(const MatrixView &lhs,
                           const ConstMatrixView<T> &rhs) {
        return lhs.ConstView() == rhs;
    }

    friend bool operator==(const MatrixView &lhs, const Matrix<T> &rhs) {
        return lhs.ConstView() == rhs.View();
    }

    friend bool operator!=(const MatrixView &lhs, const MatrixView &rhs) {
        return !(lhs == rhs);
    }

    friend bool operator!=(const MatrixView &lhs,
                           const ConstMatrixView<T> &rhs) {
        return !(lhs == rhs);
    }

    friend bool operator!=(const MatrixView &lhs, const Matrix<T> &rhs) {
        return !(lhs == rhs);
    }

    T &operator()(IndexType row_idx, IndexType col_idx) {
        assert(!IsNullMatrixPointer() && "Matrix pointer is null.");
        assert(row_idx >= 0 && row_idx < Rows() && "Invalid row index.");
        assert(col_idx >= 0 && col_idx < Columns() && "Invalid column index.");
        return (*ptr_)(row_.begin + row_idx, column_.begin + col_idx);
    }

    T operator()(IndexType row_idx, IndexType col_idx) const {
        assert(!IsNullMatrixPointer() && "Matrix pointer is null.");
        assert(row_idx >= 0 && row_idx < Rows() && "Invalid row index.");
        assert(col_idx >= 0 && col_idx < Columns() && "Invalid column index.");
        return (*ptr_)(row_.begin + row_idx, column_.begin + col_idx);
    }

    [[nodiscard]] IndexType Rows() const { return row_.end - row_.begin; }

    [[nodiscard]] IndexType Columns() const {
        return column_.end - column_.begin;
    }

    MatrixView &ApplyToEach(Function func) {
        assert(!IsNullMatrixPointer() && "Matrix pointer is null.");

        for (IndexType i = row_.begin; i < row_.end; ++i) {
            for (IndexType j = column_.begin; j < column_.end; ++j) {
                func((*ptr_)(i, j));
            }
        }

        return *this;
    }

    MatrixView &ApplyToEach(FunctionIndexes func) {
        assert(!IsNullMatrixPointer() && "Matrix pointer is null.");

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

    T GetEuclideanNorm() const { return ConstView().GetEuclideanNorm(); }

    Matrix<T> GetDiag() const { return ConstView().GetDiag(); }

    MatrixView GetRow(IndexType index) {
        assert(!IsNullMatrixPointer() && "Matrix pointer is null.");
        assert(index < Rows() &&
               "Index must be less than the number of matrix rows.");

        return MatrixView(*ptr_, {row_.begin + index, row_.begin + index + 1},
                          {column_.begin, column_.end});
    }

    MatrixView GetColumn(IndexType index) {
        assert(!IsNullMatrixPointer() && "Matrix pointer is null.");
        assert(index < Columns() &&
               "Index must be less than the number of matrix columns.");

        return MatrixView(*ptr_, {row_.begin, row_.end},
                          {column_.begin + index, column_.begin + index + 1});
    }

    MatrixView<T> GetSubmatrix(Segment row, Segment col) {
        auto [r_from, r_to] = ConstMatrixView<T>::MakeSegment(row, Rows());
        auto [c_from, c_to] = ConstMatrixView<T>::MakeSegment(col, Columns());

        assert(!IsNullMatrixPointer() && "Matrix pointer is null.");
        assert(row_.begin + r_from < Rows() && "Invalid row index.");
        assert(row_.begin + r_to <= Rows() && "Invalid row index.");
        assert(column_.begin + c_from < Columns() && "Invalid column index.");
        assert(column_.begin + c_to <= Columns() && "Invalid column index.");

        return MatrixView<T>(*ptr_, {row_.begin + r_from, row_.begin + r_to},
                             {column_.begin + c_from, column_.begin + c_to});
    }

    MatrixView<T> &Normalize() {
        assert(Rows() == 1 || Columns() == 1 && "Normalize only for vectors.");

        auto norm = GetEuclideanNorm();
        if (!utils::IsZeroFloating(norm)) {
            (*this) /= norm;
        } else {
            RoundZeroes();
        }

        return *this;
    }

    MatrixView<T> &RoundZeroes() {
        ApplyToEach(
            [](T &el) { el = (utils::IsZeroFloating(el)) ? T{0} : el; });
        return *this;
    }

    ConstMatrixView<T> ConstView() const {
        return ConstMatrixView<T>(*ptr_, {row_.begin, row_.end},
                                  {column_.begin, column_.end});
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
    [[nodiscard]] bool IsNullMatrixPointer() const { return ptr_ == nullptr; }

    Matrix<T> *ptr_;
    Segment row_;
    Segment column_;
};
} // namespace matrix_lib
