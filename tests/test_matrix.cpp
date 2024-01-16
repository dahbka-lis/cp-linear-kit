#include <gtest/gtest.h>

#include "../types/matrix/matrix.h"

#include <vector>

template <typename T>
using MatrixType = std::vector<std::vector<T>>;
using namespace linalg_lib;

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
        Matrix<int> square(5);
        EXPECT_EQ(square.Rows(), 5);
        EXPECT_EQ(square.Columns(), 5);

        Matrix<int> rect(2, 3);
        EXPECT_EQ(rect.Rows(), 2);
        EXPECT_EQ(rect.Columns(), 3);
    }
    {
        Matrix<int> matrix = {{1, 2, 3}, {4, 5, 6}};
        MatrixType<int> expect = {{1, 2, 3}, {4, 5, 6}};
        CompareMatrices(matrix, expect);
    }
    {
        MatrixType<int> expect = {{0, 0}, {1, 1}, {2, 2}};
        Matrix<int> matrix(expect);
        CompareMatrices(matrix, expect);
    }
}

TEST(TEST_MATRIX, CopySemantics) {
    MatrixType<double> default_matrix = {{1.0, 7.4}, {4.1, 5.6}};
    Matrix<double> m1(2);

    {
        Matrix<double> m2(default_matrix);
        m1 = m2;
        EXPECT_TRUE(m1 == m2);

        m2(0, 0) = 0.3;
        EXPECT_FALSE(m1 == m2);
    }

    CompareMatrices(m1, default_matrix);
}

TEST(TEST_MATRIX, MoveSemantics) {
    MatrixType<int> default_matrix = {{1}, {0}, {-1}, {-2}};
    Matrix<int> m1(4, 1);

    {
        Matrix<int> m2(default_matrix);
        m1 = std::move(m2);
    }

    CompareMatrices(m1, default_matrix);
}

void CheckArithmeticSum() {
    Matrix<int> m1 = {{1, 2, 3}, {4, 5, 6}};
    Matrix<int> m2 = {{7, 8, 9}, {10, 11, 12}};
    CompareMatrices(m1 + m2, {{8, 10, 12}, {14, 16, 18}});

    m1 += m1;
    m1 += m2;
    CompareMatrices(m1, {{9, 12, 15}, {18, 21, 24}});

    Matrix<int> m3 = {{1, 1}, {1, 1}};
    EXPECT_ANY_THROW(m1 + m3);
}

void CheckArithmeticDiff() {
    Matrix<int> m1 = {{9, 4}, {5, 1}, {12, 9}};
    Matrix<int> m2 = {{-3, 0}, {1, 4}, {6, -12}};
    CompareMatrices(m1 - m2, {{12, 4}, {4, -3}, {6, 21}});

    m1 -= m2;
    m1 -= m2;
    CompareMatrices(m1, {{15, 4}, {3, -7}, {0, 33}});

    Matrix<int> m3 = {{0}, {0}, {0}};
    EXPECT_ANY_THROW(m1 - m3);
}

void CheckArithmeticMulti() {
    Matrix<int> m1 = {{8, 6, 1}, {8, 5, 1}};
    Matrix<int> m2 = {{1, 2}, {-4, 2}, {0, -3}};

    CompareMatrices(m1 * m2, {{-16, 25}, {-12, 23}});
    CompareMatrices(m2 * m1, {{24, 16, 3}, {-16, -14, -2}, {-24, -15, -3}});

    auto m3 = m2 * m1;
    auto eye = Matrix<int>::Eye(3);

    EXPECT_TRUE(m3 * eye == m3);
    EXPECT_TRUE(eye * m3 == m3);

    Matrix<int> m4 = {{1, 2}};
    EXPECT_ANY_THROW(m3 * m4);
}

TEST(TEST_MATRIX, Arithmetic) {
    CheckArithmeticSum();
    CheckArithmeticDiff();
    CheckArithmeticMulti();
}

TEST(TEST_MATRIX, Transpose) {
    Matrix<int> m1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    m1.Transpose();
    CompareMatrices(m1, {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}});

    Matrix<int> m2 = {{0, 0}, {2, 2}, {4, 4}};
    auto m3 = m2.Transposed();
    CompareMatrices(m2, {{0, 0}, {2, 2}, {4, 4}});
    CompareMatrices(m3, {{0, 2, 4}, {0, 2, 4}});
}
