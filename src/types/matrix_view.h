#pragma once

#include "const_matrix_view.h"
#include "matrix.h"
#include "types_details.h"

namespace LinearKit {
template <Utils::FloatOrComplex T = long double>
class MatrixView {
    using IndexType = Details::Types::IndexType;
    using Segment = Details::Types::Segment;
    using TransposeState = Details::Types::TransposeState;
    using ConjugateState = Details::Types::ConjugateState;
    using MatrixState = Details::Types::MatrixState;
    using Function = Details::Types::Function<T>;
    using FunctionIndexes = Details::Types::FunctionIndexes<T>;
    using ConstFunction = Details::Types::ConstFunction<T>;
    using ConstFunctionIndexes = Details::Types::ConstFunctionIndexes<T>;

public:
    using ElemType = std::remove_cv_t<T>;

    explicit MatrixView(Matrix<T> &matrix, Segment row = {-1, -1},
                        Segment col = {-1, -1},
                        MatrixState state = {TransposeState::Normal,
                                             ConjugateState::Normal})
        : ptr_(&matrix),
          row_(ConstMatrixView<T>::MakeSegment(
              row, (state.is_transposed == TransposeState::Transposed)
                       ? matrix.Columns()
                       : matrix.Rows())),
          column_(ConstMatrixView<T>::MakeSegment(
              col, (state.is_transposed == TransposeState::Transposed)
                       ? matrix.Rows()
                       : matrix.Columns())),
          state_(state) {
    }

    MatrixView(const MatrixView &rhs) = default;

    MatrixView(MatrixView &&rhs) noexcept
        : ptr_(std::exchange(rhs.ptr_, nullptr)),
          row_(std::exchange(rhs.row_, {0, 1})),
          column_(std::exchange(rhs.column_, {0, 1})),
          state_(std::exchange(rhs.state_, MatrixState{})){};

    MatrixView &operator=(const MatrixView &lhs) = default;

    MatrixView &operator=(MatrixView &&rhs) noexcept {
        ptr_ = std::exchange(rhs.ptr_, nullptr);
        row_ = std::exchange(rhs.row_, {0, 1});
        column_ = std::exchange(rhs.column_, {0, 1});
        state_ = std::exchange(rhs.state_, MatrixState{});
        return *this;
    }

    T &operator()(IndexType row_idx, IndexType col_idx) {
        assert(ptr_ != nullptr && "Matrix pointer is null.");

        if (state_.is_transposed == TransposeState::Transposed) {
            return (*ptr_)(column_.begin + col_idx, row_.begin + row_idx);
        }

        return (*ptr_)(row_.begin + row_idx, column_.begin + col_idx);
    }

    T operator()(IndexType row_idx, IndexType col_idx) const {
        assert(ptr_ != nullptr && "Matrix pointer is null.");

        if constexpr (Utils::Details::IsFloatComplexT<T>::value) {
            if (state_.is_transposed == TransposeState::Transposed &&
                state_.is_conjugated == ConjugateState::Conjugated) {
                return std::conj(
                    (*ptr_)(column_.begin + col_idx, row_.begin + row_idx));
            } else if (state_.is_transposed == TransposeState::Transposed) {
                return (*ptr_)(column_.begin + col_idx, row_.begin + row_idx);
            } else if (state_.is_conjugated == ConjugateState::Conjugated) {
                return std::conj(
                    (*ptr_)(row_.begin + row_idx, column_.begin + col_idx));
            }

            return (*ptr_)(row_.begin + row_idx, column_.begin + col_idx);
        }

        if (state_.is_transposed == TransposeState::Transposed) {
            return (*ptr_)(column_.begin + col_idx, row_.begin + row_idx);
        }

        return (*ptr_)(row_.begin + row_idx, column_.begin + col_idx);
    }

    [[nodiscard]] IndexType Rows() const {
        assert(ptr_ != nullptr && "Matrix pointer is null.");
        auto min = (state_.is_transposed == TransposeState::Transposed)
                       ? ptr_->Columns()
                       : ptr_->Rows();
        return std::min(row_.end - row_.begin, min);
    }

    [[nodiscard]] IndexType Columns() const {
        assert(ptr_ != nullptr && "Matrix pointer is null.");
        auto min = (state_.is_transposed == TransposeState::Transposed)
                       ? ptr_->Rows()
                       : ptr_->Columns();
        return std::min(column_.end - column_.begin, min);
    }

    MatrixView &ApplyForEach(Function func) {
        assert(ptr_ != nullptr && "Matrix pointer is null.");

        for (IndexType i = 0; i < Rows(); ++i) {
            for (IndexType j = 0; j < Columns(); ++j) {
                func((*this)(i, j));
            }
        }

        return *this;
    }

    MatrixView &ApplyForEach(FunctionIndexes func) {
        assert(ptr_ != nullptr && "Matrix pointer is null.");

        for (IndexType i = 0; i < Rows(); ++i) {
            for (IndexType j = 0; j < Columns(); ++j) {
                func((*this)(i, j), i, j);
            }
        }

        return *this;
    }

    const MatrixView &ForEach(ConstFunction func) const {
        ConstView().ForEach(func);
        return *this;
    }

    const MatrixView &ForEach(ConstFunctionIndexes func) const {
        ConstView().ForEach(func);
        return *this;
    }

    T GetEuclideanNorm() const {
        return ConstView().GetEuclideanNorm();
    }

    Matrix<T> GetDiag() const {
        return ConstView().GetDiag();
    }

    MatrixView &Transpose() {
        state_.is_transposed =
            Details::Types::SwitchState(state_.is_transposed);
        std::swap(row_, column_);
        return *this;
    }

    MatrixView &Conjugate() {
        Transpose();
        state_.is_conjugated =
            Details::Types::SwitchState(state_.is_conjugated);
        return *this;
    }

    MatrixView &Normalize() {
        assert(Rows() == 1 || Columns() == 1 && "Normalize only for vectors.");

        auto norm = GetEuclideanNorm();
        if (!Utils::IsZeroFloating(norm)) {
            (*this) /= norm;
        } else {
            ApplyForEach([](T &val) { val = T{0}; });
        }

        RoundZeroes();
        return *this;
    }

    MatrixView<T> &RoundZeroes(T eps = T{0}) {
        ApplyForEach([&](T &el) {
            if constexpr (Utils::Details::IsFloatComplexT<T>::value) {
                using F = T::value_type;

                auto real = (Utils::IsZeroFloating(el.real(), eps.real()))
                                ? F{0}
                                : el.real();
                auto imag = (Utils::IsZeroFloating(el.imag(), eps.real()))
                                ? F{0}
                                : el.imag();
                el = T{real, imag};
            } else {
                el = (Utils::IsZeroFloating(el, eps)) ? T{0} : el;
            }
        });
        return *this;
    }

    ConstMatrixView<T> ConstView() const {
        return ConstMatrixView<T>(*ptr_, {row_.begin, row_.end},
                                  {column_.begin, column_.end}, state_);
    }

    MatrixView GetRow(IndexType index) {
        assert(ptr_ != nullptr && "Matrix pointer is null.");
        assert(index < Rows() &&
               "Index must be less than the number of matrix rows.");

        return MatrixView(*ptr_, {row_.begin + index, row_.begin + index + 1},
                          {column_.begin, column_.end}, state_);
    }

    MatrixView GetColumn(IndexType index) {
        assert(ptr_ != nullptr && "Matrix pointer is null.");
        assert(index < Columns() &&
               "Index must be less than the number of matrix columns.");

        return MatrixView(*ptr_, {row_.begin, row_.end},
                          {column_.begin + index, column_.begin + index + 1},
                          state_);
    }

    MatrixView<T> GetSubmatrix(Segment row, Segment col) {
        auto [r_from, r_to] = ConstMatrixView<T>::MakeSegment(row, Rows());
        auto [c_from, c_to] = ConstMatrixView<T>::MakeSegment(col, Columns());

        assert(ptr_ != nullptr && "Matrix pointer is null.");
        assert(row_.begin + r_from < Rows() && "Invalid row index.");
        assert(row_.begin + r_to <= Rows() && "Invalid row index.");
        assert(column_.begin + c_from < Columns() && "Invalid column index.");
        assert(column_.begin + c_to <= Columns() && "Invalid column index.");

        return MatrixView<T>(*ptr_, {row_.begin + r_from, row_.begin + r_to},
                             {column_.begin + c_from, column_.begin + c_to},
                             state_);
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
    MatrixState state_;
};
} // namespace LinearKit
