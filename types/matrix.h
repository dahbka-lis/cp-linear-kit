#pragma once

#include "../matrix_utils/is_matrix_type.h"
#include "const_matrix_view.h"
#include "matrix_view.h"
#include "types_details.h"

#include <ostream>
#include <vector>

namespace matrix_lib {
template <utils::FloatOrComplex T = long double>
class Matrix {
    using Data = std::vector<T>;
    using IndexType = details::Types::IndexType;
    using Segment = details::Types::Segment;
    using Function = details::Types::Function<T>;
    using FunctionIndexes = details::Types::FunctionIndexes<T>;
    using ConstFunction = details::Types::ConstFunction<T>;
    using ConstFunctionIndexes = details::Types::ConstFunctionIndexes<T>;

public:
    using ElemType = T;

    Matrix() = default;

    explicit Matrix(IndexType sq_size)
        : cols_(CorrectSize(sq_size)), buffer_(cols_ * cols_, T{0}) {
    }

    Matrix(IndexType row_cnt, IndexType col_cnt, T value = T{0})
        : cols_(CorrectSize(col_cnt)),
          buffer_(cols_ * CorrectSize(row_cnt), value) {
        if (row_cnt == IndexType{0}) {
            cols_ = IndexType{0};
        }
    }

    Matrix(std::initializer_list<std::initializer_list<T>> list)
        : cols_(list.begin()->size()) {
        buffer_.reserve(Columns() * list.size());

        for (auto sublist : list) {
            assert(
                sublist.size() == cols_ &&
                "Size of matrix rows must be equal to the number of columns.");

            for (auto value : sublist) {
                buffer_.push_back(value);
            }
        }
    }

    Matrix(const ConstMatrixView<T> &rhs) : Matrix(rhs.Rows(), rhs.Columns()) {
        rhs.ForEach([&](const T &val, IndexType i, IndexType j) {
            (*this)(i, j) = val;
        });
    }

    Matrix(const MatrixView<T> &rhs) : Matrix(rhs.ConstView()) {
    }

    Matrix(const Matrix &rhs) = default;

    Matrix(Matrix &&rhs) noexcept
        : cols_(std::exchange(rhs.cols_, 0)), buffer_(std::move(rhs.buffer_)) {
    }

    Matrix &operator=(const Matrix &rhs) = default;

    Matrix &operator=(Matrix &&rhs) noexcept {
        cols_ = std::exchange(rhs.cols_, 0);
        buffer_ = std::move(rhs.buffer_);
        return *this;
    }

    T &operator()(IndexType row_idx, IndexType col_idx) {
        assert(Columns() * row_idx + col_idx < buffer_.size() &&
               "Requested indexes are outside the matrix boundaries.");
        return buffer_[Columns() * row_idx + col_idx];
    }

    T operator()(IndexType row_idx, IndexType col_idx) const {
        assert(Columns() * row_idx + col_idx < buffer_.size() &&
               "Requested indexes are outside the matrix boundaries.");
        return buffer_[Columns() * row_idx + col_idx];
    }

    [[nodiscard]] IndexType Rows() const {
        return (cols_ == 0) ? IndexType{0} : buffer_.size() / cols_;
    }

    [[nodiscard]] IndexType Columns() const {
        return cols_;
    }

    MatrixView<T> View() {
        return MatrixView<T>(*this);
    }

    ConstMatrixView<T> View() const {
        return ConstMatrixView<T>(*this);
    }

    Matrix &ApplyForEach(Function func) {
        View().ApplyForEach(func);
        return *this;
    }

    Matrix &ApplyForEach(FunctionIndexes func) {
        View().ApplyForEach(func);
        return *this;
    }

    const Matrix &ForEach(ConstFunction func) const {
        View().ForEach(func);
        return *this;
    }

    const Matrix &ForEach(ConstFunctionIndexes func) const {
        View().ForEach(func);
        return *this;
    }

    T GetEuclideanNorm() const {
        return View().GetEuclideanNorm();
    }

    Matrix GetDiag() const {
        return View().GetDiag();
    }

    MatrixView<T> GetRow(IndexType index) {
        return View().GetRow(index);
    }

    MatrixView<T> GetColumn(IndexType index) {
        return View().GetColumn(index);
    }

    MatrixView<T> GetSubmatrix(Segment row, Segment col) {
        return View().GetSubmatrix(row, col);
    }

    ConstMatrixView<T> GetRow(IndexType index) const {
        return View().GetRow(index);
    }

    ConstMatrixView<T> GetColumn(IndexType index) const {
        return View().GetColumn(index);
    }

    ConstMatrixView<T> GetSubmatrix(Segment row, Segment col) const {
        return View().GetSubmatrix(row, col);
    }

    Matrix &Transpose() {
        std::vector<bool> visited(buffer_.size(), false);
        IndexType last_idx = buffer_.size() - 1;

        for (IndexType i = 1; i < buffer_.size(); ++i) {
            if (visited[i]) {
                continue;
            }

            auto swap_idx = i;
            do {
                swap_idx = (swap_idx == last_idx)
                               ? last_idx
                               : (Rows() * swap_idx) % last_idx;
                std::swap(buffer_[swap_idx], buffer_[i]);
                visited[swap_idx] = true;
            } while (swap_idx != i);
        }

        cols_ = Rows();
        return *this;
    }

    Matrix &Conjugate() {
        Transpose();

        if constexpr (utils::details::IsFloatComplexT<T>::value) {
            ApplyForEach([](T &val) { val = std::conj(val); });
        }

        return *this;
    }

    Matrix &Normalize() {
        View().Normalize();
        return *this;
    }

    Matrix &RoundZeroes() {
        View().RoundZeroes();
        return *this;
    }

    static MatrixView<T> Transposed(MatrixView<T> &rhs) {
        auto view = rhs;
        view.Transpose();
        return view;
    }

    static MatrixView<T> Transposed(Matrix<T> &rhs) {
        MatrixView<T> view = rhs.View();
        return Matrix::Transposed(view);
    }

    static ConstMatrixView<T> Transposed(const ConstMatrixView<T> &rhs) {
        ConstMatrixView<T> res = ConstMatrixView<T>(
            *rhs.ptr_, rhs.column_, rhs.row_,
            {!rhs.state_.is_transposed, rhs.state_.is_conjugated});
        return res;
    }

    static ConstMatrixView<T> Transposed(const MatrixView<T> &rhs) {
        ConstMatrixView<T> view = rhs.ConstView();
        return Matrix::Transposed(view);
    }

    static ConstMatrixView<T> Transposed(const Matrix<T> &rhs) {
        ConstMatrixView<T> view = rhs.View();
        return Matrix::Transposed(view);
    }

    static MatrixView<T> Conjugated(MatrixView<T> &rhs) {
        auto view = rhs;
        view.Conjugate();
        return view;
    }

    static MatrixView<T> Conjugated(Matrix<T> &rhs) {
        auto view = rhs.View();
        return Matrix::Conjugated(view);
    }

    static ConstMatrixView<T> Conjugated(const ConstMatrixView<T> &rhs) {
        ConstMatrixView<T> res = ConstMatrixView<T>(
            *rhs.ptr_, rhs.column_, rhs.row_,
            {!rhs.state_.is_transposed, !rhs.state_.is_conjugated});
        return res;
    }

    static ConstMatrixView<T> Conjugated(const MatrixView<T> &rhs) {
        ConstMatrixView<T> view = rhs.ConstView();
        return Matrix::Conjugated(view);
    }

    static ConstMatrixView<T> Conjugated(const Matrix<T> &rhs) {
        ConstMatrixView<T> view = rhs.View();
        return Matrix::Conjugated(view);
    }

    static Matrix Normalized(const Matrix &rhs) {
        return Matrix::Normalized(rhs.View());
    }

    static Matrix Normalized(const MatrixView<T> &rhs) {
        return Matrix::Normalized(rhs.ConstView());
    }

    static Matrix Normalized(const ConstMatrixView<T> &rhs) {
        Matrix res = rhs;
        res.Normalize();
        return res;
    }

    static Matrix Identity(IndexType size) {
        Matrix res(size);
        res.ApplyForEach([&](T &el, IndexType i, IndexType j) {
            el = (i == j) ? T{1} : T{0};
        });
        return res;
    }

    static Matrix Diagonal(const Matrix &vec) {
        return Diagonal(vec.View());
    }

    static Matrix Diagonal(const MatrixView<T> &vec) {
        return Diagonal(vec.ConstView());
    }

    static Matrix Diagonal(const ConstMatrixView<T> &vec) {
        assert(vec.Rows() == 1 ||
               vec.Columns() == 1 &&
                   "Creating a diagonal matrix for vectors only.");

        Matrix<T> res(std::max(vec.Rows(), vec.Columns()));
        vec.ForEach([&](const T &val, IndexType i, IndexType j) {
            auto idx = std::max(i, j);
            res(idx, idx) = val;
        });

        return res;
    }

    friend std::ostream &operator<<(std::ostream &ostream,
                                    const Matrix &matrix) {
        ostream << '[';
        for (std::size_t i = 0; i < matrix.Rows(); ++i) {
            ostream << '[';
            for (std::size_t j = 0; j < matrix.Columns(); ++j) {
                ostream << matrix(i, j);
                if (j + 1 < matrix.Columns()) {
                    ostream << ' ';
                }
            }

            ostream << ']';
            if (i + 1 < matrix.Rows()) {
                ostream << '\n';
            }
        }
        ostream << ']';
        return ostream;
    }

private:
    static IndexType CorrectSize(IndexType size) {
        return std::max(IndexType{0}, size);
    }

    IndexType cols_ = 0;
    Data buffer_;
};

using IndexType = details::Types::IndexType;

template <utils::MatrixType F, utils::MatrixType S>
Matrix<typename F::ElemType> operator+(const F &lhs, const S &rhs) {
    using T = typename F::ElemType;

    assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() &&
           "Matrices must have the same size for sum.");

    Matrix<T> result = lhs;
    for (IndexType i = 0; i < lhs.Rows(); ++i) {
        for (IndexType j = 0; j < lhs.Columns(); ++j) {
            result(i, j) += rhs(i, j);
        }
    }

    return result;
}

template <utils::MutableMatrixType F, utils::MatrixType S>
F &operator+=(F &lhs, const S &rhs) {
    assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() &&
           "Matrices must have the same size for sum.");

    for (IndexType i = 0; i < lhs.Rows(); ++i) {
        for (IndexType j = 0; j < lhs.Columns(); ++j) {
            lhs(i, j) += rhs(i, j);
        }
    }

    return lhs;
}

template <utils::MatrixType F, utils::MatrixType S>
Matrix<typename F::ElemType> operator-(const F &lhs, const S &rhs) {
    using T = typename F::ElemType;

    assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() &&
           "Matrices must have the same size for subtraction.");

    Matrix<T> result = lhs;
    for (IndexType i = 0; i < lhs.Rows(); ++i) {
        for (IndexType j = 0; j < lhs.Columns(); ++j) {
            result(i, j) -= rhs(i, j);
        }
    }

    return result;
}

template <utils::MutableMatrixType F, utils::MatrixType S>
F &operator-=(F &lhs, const S &rhs) {
    assert(lhs.Rows() == rhs.Rows() && lhs.Columns() == rhs.Columns() &&
           "Matrices must have the same size for sum.");

    for (IndexType i = 0; i < lhs.Rows(); ++i) {
        for (IndexType j = 0; j < lhs.Columns(); ++j) {
            lhs(i, j) -= rhs(i, j);
        }
    }

    return lhs;
}

template <utils::MatrixType F, utils::MatrixType S>
Matrix<typename F::ElemType> operator*(const F &lhs, const S &rhs) {
    using T = typename F::ElemType;

    if (lhs.Rows() == 0 || rhs.Rows() == 0) {
        return Matrix<T>();
    }

    assert(lhs.Columns() == rhs.Rows() && "Matrix multiplication mismatch.");

    Matrix<T> result(lhs.Rows(), rhs.Columns());

    for (IndexType i = 0; i < lhs.Rows(); ++i) {
        for (IndexType j = 0; j < rhs.Columns(); ++j) {
            T sum = 0;
            for (IndexType k = 0; k < lhs.Columns(); ++k) {
                sum += lhs(i, k) * rhs(k, j);
            }
            result(i, j) = sum;
        }
    }

    result.RoundZeroes();
    return result;
}

template <utils::MutableMatrixType F, utils::MatrixType S>
F &operator*=(F &lhs, const S &rhs) {
    if (lhs.Rows() == 0 || rhs.Rows() == 0) {
        return lhs;
    }

    assert(lhs.Columns() == rhs.Rows() && rhs.Rows() == rhs.Columns() &&
           "Matrix multiplication mismatch.");

    auto result = lhs * rhs;
    for (IndexType i = 0; i < lhs.Rows(); ++i) {
        for (IndexType j = 0; j < lhs.Columns(); ++j) {
            lhs(i, j) = result(i, j);
        }
    }

    lhs.RoundZeroes();
    return lhs;
}

template <utils::MatrixType F>
Matrix<typename F::ElemType> operator*(const F &lhs,
                                       typename F::ElemType scalar) {
    using T = typename F::ElemType;
    Matrix<T> result = lhs;

    for (IndexType i = 0; i < lhs.Rows(); ++i) {
        for (IndexType j = 0; j < lhs.Columns(); ++j) {
            result(i, j) *= scalar;
        }
    }

    result.RoundZeroes();
    return result;
}

template <utils::MatrixType F>
Matrix<typename F::ElemType> operator*(typename F::ElemType scalar,
                                       const F &rhs) {
    return rhs * scalar;
}

template <utils::MutableMatrixType F>
F &operator*=(F &lhs, typename F::ElemType scalar) {
    for (IndexType i = 0; i < lhs.Rows(); ++i) {
        for (IndexType j = 0; j < lhs.Columns(); ++j) {
            lhs(i, j) *= scalar;
        }
    }

    lhs.RoundZeroes();
    return lhs;
}

template <utils::MatrixType F>
Matrix<typename F::ElemType> operator/(const F &lhs,
                                       typename F::ElemType scalar) {
    using T = typename F::ElemType;
    Matrix<T> result = lhs;

    for (IndexType i = 0; i < lhs.Rows(); ++i) {
        for (IndexType j = 0; j < lhs.Columns(); ++j) {
            result(i, j) /= scalar;
        }
    }

    result.RoundZeroes();
    return result;
}

template <utils::MatrixType F>
Matrix<typename F::ElemType> operator/(typename F::ElemType scalar,
                                       const F &rhs) {
    return rhs / scalar;
}

template <utils::MutableMatrixType F>
F &operator/=(F &lhs, typename F::ElemType scalar) {
    for (IndexType i = 0; i < lhs.Rows(); ++i) {
        for (IndexType j = 0; j < lhs.Columns(); ++j) {
            lhs(i, j) /= scalar;
        }
    }

    lhs.RoundZeroes();
    return lhs;
}

template <utils::MatrixType F, utils::MatrixType S>
bool operator==(const F &lhs, const S &rhs) {
    if (lhs.Rows() != rhs.Rows() || lhs.Columns() != rhs.Columns()) {
        return false;
    }

    for (IndexType i = 0; i < lhs.Rows(); ++i) {
        for (IndexType j = 0; j < lhs.Columns(); ++j) {
            if (lhs(i, j) != rhs(i, j)) {
                return false;
            }
        }
    }

    return true;
}

template <utils::MatrixType F, utils::MatrixType S>
bool operator!=(const F &lhs, const S &rhs) {
    return !(lhs == rhs);
}
} // namespace matrix_lib
