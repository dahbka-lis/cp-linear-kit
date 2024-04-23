#pragma once

#include "matrix.h"
#include "matrix_view.h"
#include "types_details.h"

namespace LinearKit {
template <Utils::FloatOrComplex T = long double>
class ConstMatrixView {
    friend class Matrix<T>;
    friend class MatrixView<T>;

    using IndexType = Details::Types::IndexType;
    using Segment = Details::Types::Segment;
    using MatrixState = Details::Types::MatrixState;
    using ConstFunction = Details::Types::ConstFunction<T>;
    using ConstFunctionIndexes = Details::Types::ConstFunctionIndexes<T>;

public:
    using ElemType = T;

    explicit ConstMatrixView(const Matrix<T> &matrix, Segment row = {-1, -1},
                             Segment col = {-1, -1},
                             MatrixState state = {false, false})
        : ptr_(&matrix), state_(state) {
        row_ = MakeSegment(row, (state_.is_transposed) ? matrix.Columns()
                                                       : matrix.Rows());
        column_ = MakeSegment(col, (state_.is_transposed) ? matrix.Rows()
                                                          : matrix.Columns());
    }

    ConstMatrixView(const ConstMatrixView &rhs) = default;

    ConstMatrixView(ConstMatrixView &&rhs) noexcept
        : ptr_(std::exchange(rhs.ptr_, nullptr)),
          row_(std::exchange(rhs.row_, {0, 1})),
          column_(std::exchange(rhs.column_, {0, 1})),
          state_(std::exchange(rhs.state_, MatrixState{})){};

    ConstMatrixView &operator=(const ConstMatrixView &lhs) = default;

    ConstMatrixView &operator=(ConstMatrixView &&rhs) noexcept {
        ptr_ = std::exchange(rhs.ptr_, nullptr);
        row_ = std::exchange(rhs.row_, {0, 1});
        column_ = std::exchange(rhs.column_, {0, 1});
        state_ = std::exchange(rhs.state_, MatrixState{});
        return *this;
    }

    T operator()(IndexType row_idx, IndexType col_idx) const {
        assert(ptr_ != nullptr && "Matrix pointer is null.");

        if constexpr (Utils::Details::IsFloatComplexT<T>::value) {
            if (state_.is_transposed && state_.is_conjugated) {
                return std::conj(
                    (*ptr_)(column_.begin + col_idx, row_.begin + row_idx));
            } else if (state_.is_transposed) {
                return (*ptr_)(column_.begin + col_idx, row_.begin + row_idx);
            } else if (state_.is_conjugated) {
                return std::conj(
                    (*ptr_)(row_.begin + row_idx, column_.begin + col_idx));
            }

            return (*ptr_)(row_.begin + row_idx, column_.begin + col_idx);
        }

        if (state_.is_transposed) {
            return (*ptr_)(column_.begin + col_idx, row_.begin + row_idx);
        }

        return (*ptr_)(row_.begin + row_idx, column_.begin + col_idx);
    }

    [[nodiscard]] IndexType Rows() const {
        assert(ptr_ != nullptr && "Matrix pointer is null.");
        auto min = (state_.is_transposed) ? ptr_->Columns() : ptr_->Rows();
        return std::min(row_.end - row_.begin, min);
    }

    [[nodiscard]] IndexType Columns() const {
        assert(ptr_ != nullptr && "Matrix pointer is null.");
        auto min = (state_.is_transposed) ? ptr_->Rows() : ptr_->Columns();
        return std::min(column_.end - column_.begin, min);
    }

    const ConstMatrixView &ForEach(ConstFunction func) const {
        for (IndexType i = 0; i < Rows(); ++i) {
            for (IndexType j = 0; j < Columns(); ++j) {
                func((*this)(i, j));
            }
        }

        return *this;
    }

    const ConstMatrixView &ForEach(ConstFunctionIndexes func) const {
        for (IndexType i = 0; i < Rows(); ++i) {
            for (IndexType j = 0; j < Columns(); ++j) {
                func((*this)(i, j), i, j);
            }
        }

        return *this;
    }

    T GetEuclideanNorm() const {
        assert(Rows() == 1 ||
               Columns() == 1 && "Euclidean norm only for vectors.");

        T sq_sum = T{0};
        ForEach([&](const T &value) { sq_sum += std::norm(value); });
        return std::sqrt(sq_sum);
    }

    Matrix<T> GetDiag() const {
        auto size = std::min(Rows(), Columns());

        Matrix<T> res(size, 1);
        for (IndexType i = 0; i < size; ++i) {
            res(i, 0) = (*this)(i, i);
        }

        return res;
    }

    ConstMatrixView GetRow(IndexType index) const {
        assert(ptr_ != nullptr && "Matrix pointer is null.");
        assert(index < Rows() &&
               "Index must be less than the number of matrix rows.");

        return ConstMatrixView(*ptr_,
                               {row_.begin + index, row_.begin + index + 1},
                               {column_.begin, column_.end}, state_);
    }

    ConstMatrixView GetColumn(IndexType index) const {
        assert(ptr_ != nullptr && "Matrix pointer is null.");
        assert(index < Columns() &&
               "Index must be less than the number of matrix columns.");

        return ConstMatrixView(
            *ptr_, {row_.begin, row_.end},
            {column_.begin + index, column_.begin + index + 1}, state_);
    }

    ConstMatrixView<T> GetSubmatrix(Segment row, Segment col) const {
        auto [r_from, r_to] = MakeSegment(row, Rows());
        auto [c_from, c_to] = MakeSegment(col, Columns());

        assert(ptr_ != nullptr && "Matrix pointer is null.");
        assert(row_.begin + r_from < Rows() && "Invalid row index.");
        assert(row_.begin + r_to <= Rows() && "Invalid row index.");
        assert(column_.begin + c_from < Columns() && "Invalid column index.");
        assert(column_.begin + c_to <= Columns() && "Invalid column index.");

        return ConstMatrixView<T>(
            *ptr_, {row_.begin + r_from, row_.begin + r_to},
            {column_.begin + c_from, column_.begin + c_to}, state_);
    }

    friend std::ostream &operator<<(std::ostream &ostream,
                                    const ConstMatrixView &matrix) {
        ostream << '(';
        for (std::size_t i = 0; i < matrix.Rows(); ++i) {
            ostream << '(';
            for (std::size_t j = 0; j < matrix.Columns(); ++j) {
                ostream << matrix(i, j);
                if (j + 1 < matrix.Columns()) {
                    ostream << ' ';
                }
            }

            ostream << ')';
            if (i + 1 < matrix.Rows()) {
                ostream << '\n';
            }
        }
        ostream << ')';
        return ostream;
    }

private:
    static Segment MakeSegment(Segment seg, IndexType max_value) {
        if (seg.end <= 0 || seg.end > max_value) {
            seg.end = max_value;
        }

        if (seg.begin >= seg.end || seg.begin >= max_value || seg.begin < 0) {
            seg.begin = 0;
        }

        return seg;
    }

    const Matrix<T> *ptr_;
    Segment row_;
    Segment column_;
    MatrixState state_;
};
} // namespace LinearKit
