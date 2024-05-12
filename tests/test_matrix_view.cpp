#include <gtest/gtest.h>

#include "helpers.h"

namespace {
template <typename T = long double>
using Complex = std::complex<T>;

template <typename T = long double>
using Matrix = LinearKit::Matrix<T>;

template <typename T = long double>
using MatrixView = LinearKit::MatrixView<T>;

using LinearKit::Tests::RandomMatrixGenerator;
using LinearKit::Utils::AreEqualFloating;

TEST(TEST_MATRIX_VIEW, Create) {
    using Type = long double;
    Matrix<Type> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    {
        MatrixView<Type> view(matrix, {0, 2}, {0, 2});
        EXPECT_TRUE(view == Matrix<Type>({{1, 2}, {4, 5}}));
    }
    {
        MatrixView<Type> view = matrix.View();
        EXPECT_TRUE(view == matrix);
    }
    {
        MatrixView<Type> row = matrix.GetRow(0);
        EXPECT_TRUE(row == Matrix<Type>({{1, 2, 3}}));

        MatrixView<Type> col = matrix.GetColumn(1);
        EXPECT_TRUE(col == Matrix<Type>({{2}, {5}, {8}}));
    }
    {
        MatrixView<Type> sub = matrix.GetSubmatrix({1, 3}, {0, 3});
        EXPECT_TRUE(sub == Matrix<Type>({{4, 5, 6}, {7, 8, 9}}));
    }
}

TEST(TEST_MATRIX_VIEW, ViewEdit) {
    using Type = double;
    Matrix<Type> matrix = Matrix<Type>::Diagonal(Matrix<Type>({{1, 2, 3}}));

    auto view = matrix.GetSubmatrix({1, 3}, {1, 3});
    view(0, 0) = 5;
    view(0, 1) = -5;

    EXPECT_TRUE(view == Matrix<Type>({{5, -5}, {0, 3}}));
    EXPECT_TRUE(matrix == Matrix<Type>({{1, 0, 0}, {0, 5, -5}, {0, 0, 3}}));
}

TEST(TEST_MATRIX_VIEW, CopySemantics) {
    using Type = Complex<long double>;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    MatrixView<Type> v1 = matrix.GetColumn(0);

    {
        auto v2 = v1;
        EXPECT_TRUE(v2 == Matrix<Type>({{1}, {0}, {0}}));
        EXPECT_TRUE(v2 == v1);

        v2(0, 0) = {0, 1};
        EXPECT_TRUE(matrix(0, 0) == Type(0, 1));
        EXPECT_TRUE(v1 == v2);
    }

    EXPECT_TRUE(v1 == Matrix<Type>({{{0, 1}}, {0}, {0}}));
}

TEST(TEST_MATRIX_VIEW, MoveSemantics) {
    using Type = float;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    MatrixView<Type> v1 = matrix.GetColumn(0);

    {
        auto v2 = matrix.GetSubmatrix({1, 3}, {1, 3});
        v1 = std::move(v2);
    }

    EXPECT_TRUE(v1 == Matrix<Type>({{1, 0}, {0, 1}}));
}

void CheckArithmeticSum() {
    using Type = long double;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    Matrix<Type> sum_col = Matrix<Type>({{1}, {1}, {1}});

    auto col = matrix.GetColumn(0);
    EXPECT_TRUE(sum_col + col == Matrix<Type>({{2}, {1}, {1}}));

    col += sum_col;
    EXPECT_TRUE(col == Matrix<Type>({{2}, {1}, {1}}));
    EXPECT_TRUE(matrix == Matrix<Type>({{2, 0, 0}, {1, 1, 0}, {1, 0, 1}}));

    auto col2 = matrix.GetColumn(1);
    col += col2;
    EXPECT_TRUE(col == Matrix<Type>({{2}, {2}, {1}}));
    EXPECT_TRUE(matrix == Matrix<Type>({{2, 0, 0}, {2, 1, 0}, {1, 0, 1}}));
}

void CheckArithmeticDiff() {
    using Type = long double;

    Matrix<Type> matrix = Matrix<Type>::Diagonal(Matrix<Type>({{2}, {1}, {0}}));
    Matrix<Type> diff_row = Matrix<Type>({{-2, -1, 0}});

    auto row = matrix.GetRow(0);
    EXPECT_TRUE(diff_row + row == Matrix<Type>({{0, -1, 0}}));

    row += diff_row;
    EXPECT_TRUE(row == Matrix<Type>({{0, -1, 0}}));
    EXPECT_TRUE(matrix == Matrix<Type>({{0, -1, 0}, {0, 1, 0}, {0, 0, 0}}));

    auto row2 = matrix.GetRow(1);
    row -= row2;
    EXPECT_TRUE(row == Matrix<Type>({{0, -2, 0}}));
    EXPECT_TRUE(matrix == Matrix<Type>({{0, -2, 0}, {0, 1, 0}, {0, 0, 0}}));
}

void CheckArithmeticMulti() {
    using Type = long double;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    Matrix<Type> multi_matrix = Matrix<Type>({{1, 2}, {3, 4}});

    auto sub = matrix.GetSubmatrix({1, 3}, {1, 3});
    EXPECT_TRUE(sub * multi_matrix == Matrix<Type>({{1, 2}, {3, 4}}));
    EXPECT_TRUE(multi_matrix * sub == Matrix<Type>({{1, 2}, {3, 4}}));

    sub *= multi_matrix;
    EXPECT_TRUE(sub == Matrix<Type>({{1, 2}, {3, 4}}));
    EXPECT_TRUE(matrix == Matrix<Type>({{1, 0, 0}, {0, 1, 2}, {0, 3, 4}}));

    auto sub2 = matrix.GetSubmatrix({1, 3}, {1, 3});
    sub *= sub2;
    EXPECT_TRUE(sub == Matrix<Type>({{7, 10}, {15, 22}}));
    EXPECT_TRUE(matrix == Matrix<Type>({{1, 0, 0}, {0, 7, 10}, {0, 15, 22}}));
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
    EXPECT_TRUE(view == Matrix<Type>({{2, 6}, {4, 8}}));
    EXPECT_TRUE(a == Matrix<Type>({{2, 4}, {6, 8}}));

    auto col = a.GetColumn(0).Transpose();
    EXPECT_TRUE(col == Matrix<Type>({{2, 6}}));
    EXPECT_TRUE(a == Matrix<Type>({{2, 4}, {6, 8}}));
}

TEST(TEST_MATRIX_VIEW, Conjugate) {
    using Type = Complex<long double>;

    Matrix<Type> a = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    auto view = a.View();

    view.Conjugate();
    EXPECT_TRUE(view == Matrix<Type>({{{1, -2}, {5, -6}}, {{3, -4}, {7, -8}}}));
    EXPECT_TRUE(a == Matrix<Type>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));

    auto col = a.GetColumn(0).Conjugate();
    EXPECT_TRUE(col == Matrix<Type>({{{1, -2}, {5, -6}}}));
    EXPECT_TRUE(a == Matrix<Type>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
}

void CheckTransposeArithmeticSum() {
    using Type = long double;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    Matrix<Type> sum_col = Matrix<Type>({{-1}, {0}, {1}});

    auto t_row = matrix.GetRow(0).Transpose();
    EXPECT_TRUE(t_row + sum_col == Matrix<Type>({{0}, {0}, {1}}));

    t_row += sum_col;
    EXPECT_TRUE(t_row == Matrix<Type>({{0}, {0}, {1}}));
    EXPECT_TRUE(matrix == Matrix<Type>({{0, 0, 1}, {0, 1, 0}, {0, 0, 1}}));
}

void CheckTransposeArithmeticDiff() {
    using Type = Complex<long double>;

    Matrix<Type> matrix = Matrix<Type>::Identity(3);
    Matrix<Type> diff_row = Matrix<Type>({{{0, 1}, {0, 1}, {0, 1}}});

    auto t_col = matrix.GetColumn(1).Conjugate();
    EXPECT_TRUE(t_col - diff_row ==
                Matrix<Type>({{{0, -1}, {1, -1}, {0, -1}}}));

    t_col -= diff_row;
    EXPECT_TRUE(t_col == Matrix<Type>({{{0, 1}, {1, 1}, {0, 1}}}));
    EXPECT_TRUE(
        matrix ==
        Matrix<Type>({{1, {0, -1}, 0}, {0, {1, -1}, 0}, {0, {0, -1}, 1}}));
}

void CheckTransposeArithmeticMulti() {
    using Type = long double;

    Matrix<Type> matrix = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
    Matrix<Type> multi_matrix = {{1, 0, 3}, {0, 2, 0}};

    auto sub = matrix.GetSubmatrix({1, 3}, {0, 3}).Transpose();
    EXPECT_TRUE(sub * multi_matrix ==
                Matrix<Type>({{2, 6, 6}, {2, 6, 6}, {2, 6, 6}}));

    Matrix<Type> square_multi = {{2, 0}, {0, 2}};
    sub *= square_multi;
    EXPECT_TRUE(sub == Matrix<Type>({{4, 6}, {4, 6}, {4, 6}}));
    EXPECT_TRUE(matrix == Matrix<Type>({{1, 1, 1}, {4, 4, 4}, {6, 6, 6}}));
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
    row.ApplyForEach([](long double &val) { val = 3; });

    EXPECT_TRUE(row == Matrix<Type>({{3, 3}}));
    EXPECT_TRUE(matrix == Matrix<Type>({{1, 0}, {3, 3}}));

    auto col = matrix.GetColumn(0);
    col.ApplyForEach([](long double &val) { val *= 3; });

    EXPECT_TRUE(col == Matrix<Type>({{3}, {9}}));
    EXPECT_TRUE(matrix == Matrix<Type>({{3, 0}, {9, 3}}));
}

TEST(TEST_MATRIX_VIEW, Normalize) {
    using Type = long double;
    Matrix<Type> matrix(4, 4, 1);

    auto col = matrix.GetColumn(0);
    col.Normalize();

    auto res = Matrix<Type>({{0.5, 1., 1., 1.},
                             {0.5, 1., 1., 1.},
                             {0.5, 1., 1., 1.},
                             {0.5, 1., 1., 1.}});

    EXPECT_TRUE(col.GetEuclideanNorm() == 1);
    EXPECT_TRUE(matrix == res);
}

TEST(TEST_MATRIX_VIEW, Diagonal) {
    using Type = long double;

    Matrix<Type> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto view = matrix.GetSubmatrix({1, -1}, {1, -1});

    auto diag = view.GetDiag();
    EXPECT_TRUE(diag == Matrix<Type>({{5}, {9}}));

    auto diag_matrix = Matrix<Type>::Diagonal(diag);
    EXPECT_TRUE(diag_matrix == Matrix<Type>({{5, 0}, {0, 9}}));
}

std::pair<int32_t, int32_t> GetMinMaxSize(int32_t first, int32_t second) {
    return {std::min(first, second), std::max(first, second)};
}

TEST(TEST_MATRIX_VIEW, Stress) {
    using Type = Complex<long double>;
    using MatrixGenerator = RandomMatrixGenerator<Type>;

    const size_t it_count = 100u;

    for (int32_t seed = 1; seed < 10; ++seed) {
        MatrixGenerator gen(seed);

        for (size_t it = 0; it < it_count; ++it) {
            auto [v_row, m_row] = GetMinMaxSize(gen.GetRandomMatrixSize() + 1,
                                                gen.GetRandomMatrixSize() + 1);
            auto [v_col, m_col] = GetMinMaxSize(gen.GetRandomMatrixSize() + 1,
                                                gen.GetRandomMatrixSize() + 1);

            if (v_row == m_row || v_col == m_col) {
                continue;
            }

            auto id = it % 4;
            if (id == 0) {
                auto m1 = gen.GetRandomMatrix(m_row, m_col);
                auto m2 = gen.GetRandomMatrix(m_row, m_col);

                auto v1 = m1.GetSubmatrix({0, v_row}, {0, v_col});
                auto v2 = m2.GetSubmatrix({0, v_row}, {0, v_col});

                auto m3 = v1 + v2;
                v1 += v2;

                auto v3 = m3.GetRow(0);
                auto v4 = v1.GetRow(0);
                v3 += v4;
            } else if (id == 1) {
                auto m1 = gen.GetRandomMatrix(m_row, m_col);
                auto m2 = gen.GetRandomMatrix(m_row, m_col);

                auto v1 = m1.GetSubmatrix({0, v_row}, {0, v_col});
                auto v2 = m2.GetSubmatrix({0, v_row}, {0, v_col});

                auto m3 = v1 - v2;
                v1 -= v2;

                auto v3 = m3.GetRow(0);
                auto v4 = v1.GetRow(0);
                v3 -= v4;
            } else if (id == 2) {
                auto size = std::max(m_row, m_col);
                auto scalar = gen.GetRandomTypeNumber();

                auto m1 = gen.GetRandomMatrix(size, size);
                auto m2 = gen.GetRandomMatrix(size, size);

                auto v1 = m1.GetSubmatrix({0, v_row}, {0, v_col});
                auto v2 = m1.GetSubmatrix({0, v_col}, {0, v_row});

                auto m3 = v1 * v2;
                v2 /= scalar;

                auto m4 = gen.GetRandomMatrix(v_col, v_col);
                v1 *= m4;
                v1 *= scalar;
            } else if (id == 3) {
                auto size = std::max(m_row, m_col);
                auto scalar = gen.GetRandomTypeNumber();

                auto m1 = gen.GetRandomMatrix(size, size);
                auto m2 = gen.GetRandomMatrix(size, size);

                auto v1 = m1.GetSubmatrix({0, v_row}, {0, v_col});
                auto v2 = m1.GetSubmatrix({0, v_row}, {0, v_col});
                v1.Conjugate();

                auto m3 = v1 * v2;
                v2 /= scalar;

                auto m4 = gen.GetRandomMatrix(v_col, v_col);
                v1.Transpose();
                v1 *= m4;
                v1 *= scalar;
            }
        }
    }
}
} // namespace
