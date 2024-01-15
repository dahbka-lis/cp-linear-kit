#pragma once

#include <cassert>
#include <ostream>
#include <utility>
#include <vector>

namespace linalg_lib {
template <typename T = double> class Matrix {
    using SizeType = std::size_t;
    using VectorType = std::vector<T>;
    using MatrixType = std::vector<std::vector<T>>;

public:
    Matrix() = delete;

    explicit Matrix(SizeType sq_size) : rows_(sq_size), columns_(sq_size) {
        assert(sq_size > 0);
        buffer_ = MatrixType(rows_, VectorType(columns_, T{0}));
    }

    Matrix(SizeType row_cnt, SizeType col_cnt)
        : rows_(row_cnt), columns_(col_cnt) {
        assert(row_cnt > 0);
        assert(col_cnt > 0);

        buffer_ = MatrixType(rows_, VectorType(columns_, T{0}));
    }

    Matrix(SizeType row_cnt, SizeType col_cnt, T value)
        : rows_(row_cnt), columns_(col_cnt) {
        assert(row_cnt > 0);
        assert(col_cnt > 0);

        buffer_ = MatrixType(rows_, VectorType(columns_, value));
    }

    Matrix(std::initializer_list<std::initializer_list<T>> list)
        : rows_(list.size()), columns_(list.begin()->size()) {
        assert(rows_ > 0);
        assert(columns_ > 0);

        buffer_ = MatrixType(rows_, VectorType(columns_));
        auto buf_it = buffer_.begin();

        for (auto &sublist : list) {
            assert(sublist.size() == columns_);

            for (SizeType i = 0; i < columns_; ++i) {
                (*buf_it)[i] = *(sublist.begin() + i);
            }

            ++buf_it;
        }
    }

    explicit Matrix(const std::vector<std::vector<T>> &list)
        : rows_(list.size()), columns_(list[0].size()) {
        assert(rows_ > 0);
        assert(columns_ > 0);

        buffer_ = MatrixType(rows_, VectorType(columns_));

        for (SizeType i = 0; i < rows_; ++i) {
            assert(list[i].size() == columns_);
            buffer_[i] = list[i];
        }
    }

    Matrix(const Matrix &rhs) : rows_(rhs.rows_), columns_(rhs.columns_) {
        buffer_ = rhs.buffer_;
    }

    Matrix(Matrix &&rhs) noexcept {
        std::swap(rows_, rhs.rows_);
        std::swap(columns_, rhs.columns_);
        buffer_ = std::move(rhs.buffer_);
    }

    Matrix &operator=(const Matrix &rhs) {
        rows_ = rhs.rows_;
        columns_ = rhs.columns_;
        buffer_ = rhs.buffer_;
        return *this;
    }

    Matrix &operator=(Matrix &&rhs) noexcept {
        std::swap(rows_, rhs.rows_);
        std::swap(columns_, rhs.columns_);
        buffer_ = std::move(rhs.buffer_);
        return *this;
    }

    ~Matrix() = default;

    Matrix operator+(const Matrix &rhs) {
        if (rows_ != rhs.rows_ || columns_ != rhs.columns_) {
            throw std::runtime_error("Matrix dimension mismatch");
        }

        Matrix res = *this;
        res += rhs;
        return res;
    }

    Matrix &operator+=(const Matrix &rhs) {
        if (rows_ != rhs.rows_ || columns_ != rhs.columns_) {
            throw std::runtime_error("Matrix dimension mismatch");
        }

        for (SizeType i = 0; i < rows_; ++i) {
            for (SizeType j = 0; j < columns_; ++j) {
                buffer_[i][j] += rhs.buffer_[i][j];
            }
        }

        return *this;
    }

    Matrix operator-(const Matrix &rhs) {
        if (rows_ != rhs.rows_ || columns_ != rhs.columns_) {
            throw std::runtime_error("Matrices size mismatch.");
        }

        Matrix res = *this;
        res -= rhs;
        return res;
    }

    Matrix &operator-=(const Matrix &rhs) {
        if (rows_ != rhs.rows_ || columns_ != rhs.columns_) {
            throw std::runtime_error("Matrices size mismatch.");
        }

        for (SizeType i = 0; i < rows_; ++i) {
            for (SizeType j = 0; j < columns_; ++j) {
                buffer_[i][j] -= rhs.buffer_[i][j];
            }
        }

        return *this;
    }

    Matrix operator*(const Matrix &rhs) {
        if (columns_ != rhs.rows_) {
            throw std::runtime_error("Matrix dimension mismatch");
        }

        Matrix result(rows_, rhs.columns_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < rhs.columns_; ++j) {
                T sum = 0;

                for (size_t k = 0; k < columns_; ++k) {
                    sum += (*this)(i, k) * rhs(k, j);
                }

                result(i, j) = sum;
            }
        }

        return result;
    }

    Matrix &operator*=(const Matrix &rhs) {
        Matrix prod = (*this) * rhs;

        std::swap(rows_, prod.rows_);
        std::swap(columns_, prod.columns_);
        buffer_ = std::move(prod.buffer_);

        return *this;
    }

    Matrix operator+(T value) {
        Matrix res = *this;
        res += value;
        return res;
    }

    Matrix &operator+=(T value) {
        for (SizeType i = 0; i < rows_; ++i) {
            for (SizeType j = 0; j < columns_; ++j) {
                buffer_[i][j] += value;
            }
        }

        return *this;
    }

    Matrix operator-(T value) {
        Matrix res = *this;
        res -= value;
        return res;
    }

    Matrix &operator-=(T value) {
        for (SizeType i = 0; i < rows_; ++i) {
            for (SizeType j = 0; j < columns_; ++j) {
                buffer_[i][j] -= value;
            }
        }

        return *this;
    }

    Matrix operator*(T value) {
        Matrix res = *this;
        res *= value;
        return res;
    }

    Matrix &operator*=(T value) {
        for (SizeType i = 0; i < rows_; ++i) {
            for (SizeType j = 0; j < columns_; ++j) {
                buffer_[i][j] *= value;
            }
        }

        return *this;
    }

    Matrix operator/(T value) {
        Matrix res = *this;
        res /= value;
        return res;
    }

    Matrix &operator/=(T value) {
        for (SizeType i = 0; i < rows_; ++i) {
            for (SizeType j = 0; j < columns_; ++j) {
                buffer_[i][j] /= value;
            }
        }

        return *this;
    }

    bool operator==(const Matrix &rhs) const {
        if (rows_ != rhs.rows_ || columns_ != rhs.columns_) {
            return false;
        }

        for (SizeType i = 0; i < rows_; ++i) {
            for (SizeType j = 0; j < columns_; ++j) {
                if (buffer_[i][j] != rhs.buffer_[i][j]) {
                    return false;
                }
            }
        }

        return true;
    }

    bool operator!=(const Matrix &rhs) const {
        if (rows_ != rhs.rows_ || columns_ != rhs.columns_) {
            return true;
        }

        for (SizeType i = 0; i < rows_; ++i) {
            for (SizeType j = 0; j < columns_; ++j) {
                if (buffer_[i][j] != rhs.buffer_[i][j]) {
                    return true;
                }
            }
        }

        return false;
    }

    bool operator<=>(const Matrix &rhs) = delete;

    T &operator()(size_t row, size_t column) {
        if (row >= rows_ || column >= columns_) {
            throw std::runtime_error("Wrong index for matrix");
        }

        return buffer_[row][column];
    }

    T operator()(size_t row, size_t column) const {
        if (row >= rows_ || column >= columns_) {
            throw std::runtime_error("Wrong index for matrix");
        }

        return buffer_[row][column];
    }

    [[nodiscard]] SizeType Rows() const { return rows_; }

    [[nodiscard]] SizeType Columns() const { return columns_; }

    MatrixType GetVectorMatrix() const { return buffer_; }

    void Transpose() {
        MatrixType new_buffer(columns_, VectorType(rows_));

        for (SizeType i = 0; i < rows_; ++i) {
            for (SizeType j = 0; j < columns_; ++j) {
                new_buffer[j][i] = buffer_[i][j];
            }
        }

        std::swap(rows_, columns_);
        buffer_ = std::move(new_buffer);
    }

    Matrix Transposed() const {
        Matrix res = *this;
        res.Transpose();
        return res;
    }

    static Matrix Eye(SizeType size, T default_value = T{1}) {
        Matrix res(size);

        for (SizeType i = 0; i < size; ++i) {
            res.buffer_[i][i] = default_value;
        }

        return res;
    }

    static Matrix Diag(const std::vector<T> &list) {
        Matrix res(list.size());

        for (SizeType i = 0; i < list.size(); ++i) {
            res.buffer_[i][i] = list[i];
        }

        return res;
    }

private:
    SizeType rows_ = 0;
    SizeType columns_ = 0;
    MatrixType buffer_;
};

template <typename T>
std::ostream &operator<<(std::ostream &ostream, const Matrix<T> &matrix) {
    ostream << '[';

    for (size_t i = 0; i < matrix.Rows(); ++i) {
        ostream << '[';

        for (size_t j = 0; j < matrix.Columns(); ++j) {
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
} // namespace linalg_lib