#include <gtest/gtest.h>

#include "../types/matrix/matrix.h"

#include <random>
#include <vector>

namespace {
template <typename T>
using MatrixType = std::vector<std::vector<T>>;
using std::complex;

using namespace matrix_lib;

template <typename T>
void CompareMatrices(const Matrix<T> &m1, const MatrixType<T> &m2) {
    ASSERT_EQ(m1.Rows(), m2.size()) << "Wrong size of matrix rows";

    for (size_t i = 0; i < m2.size(); ++i) {
        ASSERT_EQ(m1.Columns(), m2[i].size()) << "Wrong size of matrix columns";

        for (size_t j = 0; j < m2[i].size(); ++j) {
            ASSERT_EQ(m1(i, j), m2[i][j]) << "Matrices are not equal";
        }
    }
}

TEST(TEST_MATRIX, BasicConstructors) {
    {
        using Matrix = Matrix<float>;

        Matrix square(5);
        EXPECT_EQ(square.Rows(), 5);
        EXPECT_EQ(square.Columns(), 5);

        Matrix rect(2, 3);
        EXPECT_EQ(rect.Rows(), 2);
        EXPECT_EQ(rect.Columns(), 3);
    }
    {
        using Matrix = Matrix<double>;

        Matrix matrix = {{1, 2, 3}, {4, 5, 6}};
        MatrixType<double> expect = {{1, 2, 3}, {4, 5, 6}};
        CompareMatrices(matrix, expect);
    }
    {
        using Matrix = Matrix<long double>;

        MatrixType<long double> expect = {{0, 0}, {1, 1}, {2, 2}};
        Matrix matrix(expect);
        CompareMatrices(matrix, expect);
    }
}

TEST(TEST_MATRIX, CopySemantics) {
    using Matrix = Matrix<double>;

    MatrixType<double> default_matrix = {{1.0, 7.4}, {4.1, 5.6}};
    Matrix m1(2);

    {
        Matrix m2(default_matrix);
        m1 = m2;
        EXPECT_TRUE(m1 == m2);

        m2(0, 0) = 0.3;
        EXPECT_FALSE(m1 == m2);
    }

    CompareMatrices(m1, default_matrix);
}

TEST(TEST_MATRIX, MoveSemantics) {
    using Matrix = Matrix<complex<float>>;

    MatrixType<complex<float>> default_matrix = {
        {1, -1}, {0, 2}, {-1, 0}, {-2, 1}};
    Matrix m1(4, 1);

    {
        Matrix m2(default_matrix);
        m1 = std::move(m2);
    }

    CompareMatrices(m1, default_matrix);
}

void CheckArithmeticSum() {
    using Matrix = Matrix<double>;

    Matrix m1 = {{1, 2, 3}, {4, 5, 6}};
    Matrix m2 = {{7, 8, 9}, {10, 11, 12}};
    CompareMatrices(m1 + m2, {{8, 10, 12}, {14, 16, 18}});

    m1 += m1;
    m1 += m2;
    CompareMatrices(m1, {{9, 12, 15}, {18, 21, 24}});

    Matrix m3 = {{1, 1}, {1, 1}};
    EXPECT_ANY_THROW(m1 + m3);
}

void CheckArithmeticDiff() {
    using Matrix = Matrix<double>;

    Matrix m1 = {{9, 4}, {5, 1}, {12, 9}};
    Matrix m2 = {{-3, 0}, {1, 4}, {6, -12}};
    CompareMatrices(m1 - m2, {{12, 4}, {4, -3}, {6, 21}});

    m1 -= m2;
    m1 -= m2;
    CompareMatrices(m1, {{15, 4}, {3, -7}, {0, 33}});

    Matrix m3 = {{0}, {0}, {0}};
    EXPECT_ANY_THROW(m1 - m3);
}

void CheckArithmeticMulti() {
    using Matrix = Matrix<double>;

    Matrix m1 = {{8, 6, 1}, {8, 5, 1}};
    Matrix m2 = {{1, 2}, {-4, 2}, {0, -3}};

    CompareMatrices(m1 * m2, {{-16, 25}, {-12, 23}});
    CompareMatrices(m2 * m1, {{24, 16, 3}, {-16, -14, -2}, {-24, -15, -3}});

    auto m3 = m2 * m1;
    auto eye = Matrix::Eye(3);

    EXPECT_TRUE(m3 * eye == m3);
    EXPECT_TRUE(eye * m3 == m3);

    Matrix m4 = {{1, 2}};
    EXPECT_ANY_THROW(m3 * m4);
}

TEST(TEST_MATRIX, Arithmetic) {
    CheckArithmeticSum();
    CheckArithmeticDiff();
    CheckArithmeticMulti();
}

TEST(TEST_MATRIX, Transpose) {
    using Matrix = Matrix<float>;

    Matrix m1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    m1.Transpose();
    CompareMatrices(m1, {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}});

    Matrix m2 = {{0, 0}, {2, 2}, {4, 4}};
    auto m3 = m2.Transposed();

    CompareMatrices(m2, {{0, 0}, {2, 2}, {4, 4}});
    CompareMatrices(m3, {{0, 2, 4}, {0, 2, 4}});
}

TEST(TEST_MATRIX, Conjugate) {
    using Matrix = Matrix<complex<double>>;

    Matrix m1 = {{{1, 2}, {7, -3}}, {{0, -1}, {-5, 1}}, {{4, 0}, {2, -2}}};
    m1.Conjugate();
    CompareMatrices(m1,
                    {{{1, -2}, {0, 1}, {4, 0}}, {{7, 3}, {-5, -1}, {2, 2}}});

    Matrix m2 = {{{1, 0}, {1, 1}}, {{0, -1}, {1, 0}}};
    auto m3 = m2.Conjugated();

    CompareMatrices(m2, {{{1, 0}, {1, 1}}, {{0, -1}, {1, 0}}});
    CompareMatrices(m3, {{{1, 0}, {0, 1}}, {{1, -1}, {1, 0}}});
}

TEST(TEST_MATRIX, DiagonalMethods) {
    std::vector<double> diag = {1, 2, 3, 4, 5};

    auto matrix = Matrix<double>::Diagonal(diag);
    EXPECT_EQ(matrix.Rows(), diag.size());
    EXPECT_EQ(matrix.Columns(), diag.size());

    auto matrix_diag = matrix.GetDiag();
    EXPECT_EQ(diag.size(), matrix_diag.size());

    for (size_t i = 0; i < diag.size(); ++i) {
        EXPECT_DOUBLE_EQ(matrix_diag[i], diag[i]);
        EXPECT_DOUBLE_EQ(matrix(i, i), diag[i]);
    }
}

constexpr std::size_t seed = 110u;
constexpr std::size_t matrix_size = 100u;
constexpr std::size_t it_cnt = 100u;

std::mt19937 rng(seed);
std::uniform_int_distribution<int> op_dist(0, 4);
std::uniform_int_distribution<int> scalar_dist(0, 3);
std::uniform_int_distribution<int> num_dist(-50, 50);

Matrix<long double> GenerateRandomMatrix() {
    Matrix<long double> matrix(matrix_size);

    for (size_t i = 0; i < matrix_size; ++i) {
        for (size_t j = 0; j < matrix_size; ++j) {
            matrix(i, j) = num_dist(rng);
        }
    }

    return matrix;
}

Matrix<long double> ConstructRandomMatrix() {
    switch (scalar_dist(rng)) {
    case 0:
        return {matrix_size, matrix_size,
                static_cast<long double>(num_dist(rng))};
    case 1:
        return Matrix<long double>::Eye(matrix_size, num_dist(rng));
    case 2:
        return GenerateRandomMatrix();
    default:
        return Matrix<long double>::Eye(matrix_size);
    }
}

TEST(TEST_MATRIX, Stress) {
    Matrix<long double> matrix = GenerateRandomMatrix();

    for (size_t it = 0; it < it_cnt; ++it) {
        auto test_matrix = ConstructRandomMatrix();

        // Test operators
        try {
            switch (op_dist(rng)) {
            case 0:
                matrix += test_matrix;
                break;

            case 1:
                matrix -= test_matrix;
                break;

            case 2:
                matrix = matrix * test_matrix;
                break;

            case 3:
                matrix = test_matrix * matrix;
                break;

            default:
                matrix.Transpose();
            }
        } catch (...) {
            std::cerr << "Stress (operators) FAILED! Seed: " << seed
                      << ", iteration: " << it << std::endl;
            ASSERT_TRUE(false);
            break;
        }

        // Test scalar operators
        try {
            auto scalar = num_dist(rng);
            switch (scalar_dist(rng)) {
            case 0:
                matrix += scalar;
                break;

            case 1:
                matrix -= scalar;
                break;

            case 2:
                matrix *= scalar;
                break;

            default:
                if (scalar != 0) {
                    matrix /= num_dist(rng);
                }
            }
        } catch (...) {
            std::cerr << "Stress (scalar) FAILED! Seed: " << seed
                      << ", iteration: " << it << std::endl;
            ASSERT_TRUE(false);
            break;
        }
    }
}
} // namespace
