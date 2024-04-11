#include <gtest/gtest.h>

#include "helpers.h"
#include <iostream>

namespace {
template <typename T = long double>
using Complex = std::complex<T>;

template <typename T = long double>
using Matrix = matrix_lib::Matrix<T>;

template <typename T = long double>
using MatrixView = matrix_lib::MatrixView<T>;

using matrix_lib::tests::RandomMatrixGenerator;
using matrix_lib::utils::IsEqualFloating;

TEST(TEST_MATRIX_VIEW, ViewCreate) {
    using Type = long double;
    Matrix<Type> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    {
        MatrixView<Type> view(matrix, {0, 2}, {0, 2});
        ASSERT_TRUE(view == Matrix<Type>({{1, 2}, {4, 5}}));
    }
    {
        MatrixView<Type> view = matrix.View();
        ASSERT_TRUE(view == matrix);
    }
    {
        MatrixView<Type> row = matrix.GetRow(0);
        ASSERT_TRUE(row == Matrix<Type>({{1, 2, 3}}));

        MatrixView<Type> col = matrix.GetColumn(1);
        ASSERT_TRUE(col == Matrix<Type>({{2}, {5}, {8}}));
    }
    {
        MatrixView<Type> sub = matrix.GetSubmatrix({1, 3}, {0, 3});
        ASSERT_TRUE(sub == Matrix<Type>({{4, 5, 6}, {7, 8, 9}}));
    }
}

TEST(TEST_MATRIX_VIEW, ViewEdit) {
    using Type = double;
    Matrix<Type> matrix = Matrix<Type>::Diagonal(Matrix<Type>({{1, 2, 3}}));

    auto view = matrix.GetSubmatrix({1, 3}, {1, 3});
    view(0, 0) = 5;
    view(0, 1) = -5;

    ASSERT_TRUE(view == Matrix<Type>({{5, -5}, {0, 3}}));
    ASSERT_TRUE(matrix == Matrix<Type>({{1, 0, 0}, {0, 5, -5}, {0, 0, 3}}));
}

TEST(TEST_MATRIX_VIEW, CopySemantics) {
    using Type = Complex<long double>;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    MatrixView<Type> v1 = matrix.GetColumn(0);

    {
        auto v2 = v1;
        ASSERT_TRUE(v2 == Matrix<Type>({{1}, {0}, {0}}));
        ASSERT_TRUE(v2 == v1);

        v2(0, 0) = {0, 1};
        ASSERT_TRUE(matrix(0, 0) == Type(0, 1));
        ASSERT_TRUE(v1 == v2);
    }

    ASSERT_TRUE(v1 == Matrix<Type>({{{0, 1}}, {0}, {0}}));
}

TEST(TEST_MATRIX_VIEW, MoveSemantics) {
    using Type = float;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    MatrixView<Type> v1 = matrix.GetColumn(0);

    {
        auto v2 = matrix.GetSubmatrix({1, 3}, {1, 3});
        v1 = std::move(v2);
    }

    ASSERT_TRUE(v1 == Matrix<Type>({{1, 0}, {0, 1}}));
}

void CheckArithmeticSum() {
    using Type = long double;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    Matrix<Type> sum_col = Matrix<Type>({{1}, {1}, {1}});

    auto col = matrix.GetColumn(0);
    ASSERT_TRUE(sum_col + col == Matrix<Type>({{2}, {1}, {1}}));

    col += sum_col;
    ASSERT_TRUE(col == Matrix<Type>({{2}, {1}, {1}}));
    ASSERT_TRUE(matrix == Matrix<Type>({{2, 0, 0}, {1, 1, 0}, {1, 0, 1}}));

    auto col2 = matrix.GetColumn(1);
    col += col2;
    ASSERT_TRUE(col == Matrix<Type>({{2}, {2}, {1}}));
    ASSERT_TRUE(matrix == Matrix<Type>({{2, 0, 0}, {2, 1, 0}, {1, 0, 1}}));
}

void CheckArithmeticDiff() {
    using Type = long double;

    Matrix<Type> matrix = Matrix<Type>::Diagonal(Matrix<Type>({{2}, {1}, {0}}));
    Matrix<Type> diff_row = Matrix<Type>({{-2, -1, 0}});

    auto row = matrix.GetRow(0);
    ASSERT_TRUE(diff_row + row == Matrix<Type>({{0, -1, 0}}));

    row += diff_row;
    ASSERT_TRUE(row == Matrix<Type>({{0, -1, 0}}));
    ASSERT_TRUE(matrix == Matrix<Type>({{0, -1, 0}, {0, 1, 0}, {0, 0, 0}}));

    auto row2 = matrix.GetRow(1);
    row -= row2;
    ASSERT_TRUE(row == Matrix<Type>({{0, -2, 0}}));
    ASSERT_TRUE(matrix == Matrix<Type>({{0, -2, 0}, {0, 1, 0}, {0, 0, 0}}));
}

void CheckArithmeticMulti() {
    using Type = long double;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    Matrix<Type> multi_matrix = Matrix<Type>({{1, 2}, {3, 4}});

    auto sub = matrix.GetSubmatrix({1, 3}, {1, 3});
    ASSERT_TRUE(sub * multi_matrix == Matrix<Type>({{1, 2}, {3, 4}}));
    ASSERT_TRUE(multi_matrix * sub == Matrix<Type>({{1, 2}, {3, 4}}));

    sub *= multi_matrix;
    ASSERT_TRUE(sub == Matrix<Type>({{1, 2}, {3, 4}}));
    ASSERT_TRUE(matrix == Matrix<Type>({{1, 0, 0}, {0, 1, 2}, {0, 3, 4}}));

    auto sub2 = matrix.GetSubmatrix({1, 3}, {1, 3});
    sub *= sub2;
    ASSERT_TRUE(sub == Matrix<Type>({{7, 10}, {15, 22}}));
    ASSERT_TRUE(matrix == Matrix<Type>({{1, 0, 0}, {0, 7, 10}, {0, 15, 22}}));
}

TEST(TEST_MATRIX_VIEW, Arithmetic) {
    CheckArithmeticSum();
    CheckArithmeticDiff();
    CheckArithmeticMulti();
}

TEST(TEST_MATRIX_VIEW, Transpose) {
    using Type = double;

    Matrix<Type> a = {{2, 4}, {6, 8}};
    auto view = a.View();

    view.Transpose();
    ASSERT_TRUE(view == Matrix<Type>({{2, 6}, {4, 8}}));
    ASSERT_TRUE(a == Matrix<Type>({{2, 4}, {6, 8}}));

    auto col = a.GetColumn(0).Transpose();
    ASSERT_TRUE(col == Matrix<Type>({{2, 6}}));
    ASSERT_TRUE(a == Matrix<Type>({{2, 4}, {6, 8}}));
}

TEST(TEST_MATRIX_VIEW, Conjugate) {
    using Type = Complex<long double>;

    Matrix<Type> a = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    auto view = a.View();

    view.Conjugate();
    ASSERT_TRUE(view == Matrix<Type>({{{1, -2}, {5, -6}}, {{3, -4}, {7, -8}}}));
    ASSERT_TRUE(a == Matrix<Type>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));

    auto col = a.GetColumn(0).Conjugate();
    ASSERT_TRUE(col == Matrix<Type>({{{1, -2}, {5, -6}}}));
    ASSERT_TRUE(a == Matrix<Type>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
}

void CheckTransposeArithmeticSum() {
    using Type = long double;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    Matrix<Type> sum_col = Matrix<Type>({{-1}, {0}, {1}});

    auto t_row = matrix.GetRow(0).Transpose();
    ASSERT_TRUE(t_row + sum_col == Matrix<Type>({{0}, {0}, {1}}));

    t_row += sum_col;
    ASSERT_TRUE(t_row == Matrix<Type>({{0}, {0}, {1}}));
    ASSERT_TRUE(matrix == Matrix<Type>({{0, 0, 1}, {0, 1, 0}, {0, 0, 1}}));
}

void CheckTransposeArithmeticDiff() {
    using Type = Complex<long double>;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    Matrix<Type> diff_row = Matrix<Type>({{{0, 1}, {0, 1}, {0, 1}}});

    auto t_col = matrix.GetColumn(1).Conjugate();
    ASSERT_TRUE(t_col - diff_row ==
                Matrix<Type>({{{0, -1}, {1, -1}, {0, -1}}}));

    t_col -= diff_row;
    ASSERT_TRUE(t_col == Matrix<Type>({{{0, 1}, {1, 1}, {0, 1}}}));
    ASSERT_TRUE(
        matrix ==
        Matrix<Type>({{1, {0, -1}, 0}, {0, {1, -1}, 0}, {0, {0, -1}, 1}}));
}

void CheckTransposeArithmeticMulti() {
    using Type = long double;

    Matrix<Type> matrix = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
    Matrix<Type> multi_matrix = {{1, 0, 3}, {0, 2, 0}};

    auto sub = matrix.GetSubmatrix({1, 3}, {0, 3}).Transpose();
    ASSERT_TRUE(sub * multi_matrix ==
                Matrix<Type>({{2, 6, 6}, {2, 6, 6}, {2, 6, 6}}));

    Matrix<Type> square_multi = {{2, 0}, {0, 2}};
    sub *= square_multi;
    ASSERT_TRUE(sub == Matrix<Type>({{4, 6}, {4, 6}, {4, 6}}));
    ASSERT_TRUE(matrix == Matrix<Type>({{1, 1, 1}, {4, 4, 4}, {6, 6, 6}}));
}

TEST(TEST_MATRIX_VIEW, TransposeArithmetic) {
    CheckTransposeArithmeticSum();
    CheckTransposeArithmeticDiff();
    CheckTransposeArithmeticMulti();
}

TEST(TEST_MATRIX_VIEW, ApplyToEach) {
    using Type = long double;

    Matrix<Type> matrix = Matrix<Type>::Identity(2);

    auto row = matrix.GetRow(1);
    row.ApplyToEach([](long double &val) { val = 3; });

    ASSERT_TRUE(row == Matrix<Type>({{3, 3}}));
    ASSERT_TRUE(matrix == Matrix<Type>({{1, 0}, {3, 3}}));

    auto col = matrix.GetColumn(0);
    col.ApplyToEach([](long double &val) { val *= 3; });

    ASSERT_TRUE(col == Matrix<Type>({{3}, {9}}));
    ASSERT_TRUE(matrix == Matrix<Type>({{3, 0}, {9, 3}}));
}

TEST(TEST_MATRIX_VIEW, Normalize) {
}

TEST(TEST_MATRIX_VIEW, DiagonalMatrix) {
}

TEST(TEST_MATRIX_VIEW, Stress) {
}
} // namespace
