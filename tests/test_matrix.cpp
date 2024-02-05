#include <gtest/gtest.h>

#include "../types/matrix/matrix.h"

#include <random>

namespace {
template <typename T>
using Complex = std::complex<T>;

template <typename T>
using Matrix = matrix_lib::Matrix<T>;

template <typename T>
void CompareMatrices(const Matrix<T> &m1,
                     const std::vector<std::vector<T>> &m2) {
    ASSERT_EQ(m1.Rows(), m2.size()) << "Matrix row size mismatch for compare.";

    for (size_t i = 0; i < m2.size(); ++i) {
        ASSERT_EQ(m1.Columns(), m2[i].size())
            << "Matrix column size mismatch for compare.";

        for (size_t j = 0; j < m2[i].size(); ++j) {
            ASSERT_TRUE(matrix_lib::utils::IsEqualFloating(m1(i, j), m2[i][j]))
                << "Matrices are not equal.";
        }
    }
}

TEST(TEST_MATRIX, BasicConstructors) {
    {
        Matrix<float> square(5);
        EXPECT_EQ(square.Rows(), 5);
        EXPECT_EQ(square.Columns(), 5);

        Matrix<double> rect(2, 3);
        EXPECT_EQ(rect.Rows(), 2);
        EXPECT_EQ(rect.Columns(), 3);
    }
    {
        Matrix<long double> matrix = {{1, 2, 3}, {4, 5, 6}};
        CompareMatrices(matrix, {{1, 2, 3}, {4, 5, 6}});
    }
    {
        Matrix<Complex<float>> matrix = {{0, 0}, {1, 1}, {2, 2}};
        CompareMatrices(matrix, {{0, 0}, {1, 1}, {2, 2}});
    }
}

TEST(TEST_MATRIX, CopySemantics) {
    using Matrix = Matrix<double>;
    Matrix m1(2);

    {
        Matrix m2 = {{1.0, 7.4}, {4.1, 5.6}};
        m1 = m2;
        EXPECT_TRUE(m1 == m2);

        m2(0, 0) = 0.3;
        EXPECT_FALSE(m1 == m2);
    }

    CompareMatrices(m1, {{1.0, 7.4}, {4.1, 5.6}});
}

TEST(TEST_MATRIX, MoveSemantics) {
    using Matrix = Matrix<Complex<float>>;
    Matrix m1;

    {
        Matrix m2 = {{{1, -1}, {0, 2}, {-1, 0}, {-2, 1}}};
        m1 = std::move(m2);
    }

    CompareMatrices(m1, {{{1, -1}, {0, 2}, {-1, 0}, {-2, 1}}});
}

void CheckArithmeticSum() {
    using Matrix = Matrix<double>;

    Matrix m1 = {{1, 2, 3}, {4, 5, 6}};
    Matrix m2 = {{7, 8, 9}, {10, 11, 12}};
    CompareMatrices(m1 + m2, {{8, 10, 12}, {14, 16, 18}});

    m1 += m1;
    m1 += m2;
    CompareMatrices(m1, {{9, 12, 15}, {18, 21, 24}});
}

void CheckArithmeticDiff() {
    using Matrix = Matrix<double>;

    Matrix m1 = {{9, 4}, {5, 1}, {12, 9}};
    Matrix m2 = {{-3, 0}, {1, 4}, {6, -12}};
    CompareMatrices(m1 - m2, {{12, 4}, {4, -3}, {6, 21}});

    m1 -= m2;
    m1 -= m2;
    CompareMatrices(m1, {{15, 4}, {3, -7}, {0, 33}});
}

void CheckArithmeticMulti() {
    using Matrix = Matrix<double>;

    Matrix m1 = {{8, 6, 1}, {8, 5, 1}};
    Matrix m2 = {{1, 2}, {-4, 2}, {0, -3}};

    CompareMatrices(m1 * m2, {{-16, 25}, {-12, 23}});
    CompareMatrices(m2 * m1, {{24, 16, 3}, {-16, -14, -2}, {-24, -15, -3}});

    auto m3 = m2 * m1;
    auto identity = Matrix::Identity(3);

    EXPECT_TRUE(m3 * identity == m3);
    EXPECT_TRUE(identity * m3 == m3);
}

TEST(TEST_MATRIX, Arithmetic) {
    CheckArithmeticSum();
    CheckArithmeticDiff();
    CheckArithmeticMulti();
}

TEST(TEST_MATRIX, Transpose) {
    using Matrix = Matrix<float>;

    {
        Matrix m1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        m1.Transpose();
        CompareMatrices(m1, {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}});
    }
    {
        Matrix m2 = {{0, 0}, {2, 2}, {4, 4}};
        auto m3 = Matrix::Transposed(m2);

        CompareMatrices(m2, {{0, 0}, {2, 2}, {4, 4}});
        CompareMatrices(m3, {{0, 2, 4}, {0, 2, 4}});
    }
}

TEST(TEST_MATRIX, Conjugate) {
    using Matrix = Matrix<Complex<double>>;

    {
        Matrix m1 = {{{1, 2}, {7, -3}}, {{0, -1}, {-5, 1}}, {{4, 0}, {2, -2}}};
        m1.Conjugate();
        CompareMatrices(
            m1, {{{1, -2}, {0, 1}, {4, 0}}, {{7, 3}, {-5, -1}, {2, 2}}});
    }
    {
        Matrix m2 = {{{1, 0}, {1, 1}}, {{0, -1}, {1, 0}}};
        auto m3 = Matrix::Conjugated(m2);

        CompareMatrices(m2, {{{1, 0}, {1, 1}}, {{0, -1}, {1, 0}}});
        CompareMatrices(m3, {{{1, 0}, {0, 1}}, {{1, -1}, {1, 0}}});
    }
}

TEST(TEST_MATRIX, DiagonalMatrix) {
    std::vector<double> diag = {1, 2, 3, 4, 5};

    auto matrix = Matrix<double>(diag);
    EXPECT_EQ(matrix.Rows(), diag.size());
    EXPECT_EQ(matrix.Columns(), diag.size());

    auto matrix_diag = matrix.GetDiag();
    EXPECT_EQ(diag.size(), matrix_diag.Rows());

    for (size_t i = 0; i < diag.size(); ++i) {
        EXPECT_DOUBLE_EQ(matrix_diag(i, 0), diag[i]);
        EXPECT_DOUBLE_EQ(matrix(i, i), diag[i]);
    }
}

template <typename T>
struct Matrices {
    Matrix<T> left;
    Matrix<T> right;
};

template <typename T>
class RandomGenerator {
    using IntDistribution = std::uniform_int_distribution<int32_t>;

public:
    RandomGenerator(int32_t seed, int32_t from, int32_t to)
        : rng_(seed), random_value_(from, to),
          random_matrix_size_(kMatrixMinSize, kMatrixMaxSize) {}

    int32_t GetRandomValue() { return random_value_(rng_); }

    int32_t GetRandomMatrixSize() { return random_matrix_size_(rng_); }

    Matrix<T> GenerateRandomMatrix(int32_t row, int32_t col) {
        Matrix<T> result(row, col);
        result.ApplyToEach([&](T &val) { val = GetRandomValue(); });
        return result;
    }

    Matrices<T> GenerateEqualDimMatrices() {
        auto size = GetRandomMatrixSize();
        auto m1 = GenerateRandomMatrix(size, size);
        auto m2 = GenerateRandomMatrix(size, size);

        return {m1, m2};
    }

    Matrices<T> GenerateColRowMatrices() {
        auto col_row = GetRandomMatrixSize();
        auto m1_row = GetRandomMatrixSize();
        auto m2_col = GetRandomMatrixSize();

        auto m1 = GenerateRandomMatrix(m1_row, col_row);
        auto m2 = GenerateRandomMatrix(col_row, m2_col);

        return {m1, m2};
    }

    static constexpr std::size_t kIterationCnt = 1000;
    static constexpr std::size_t kMatrixMinSize = 1;
    static constexpr std::size_t kMatrixMaxSize = 100;

private:
    std::mt19937 rng_;
    IntDistribution random_value_;
    IntDistribution random_matrix_size_;
};

TEST(TEST_MATRIX, Stress) {
    using Type = long double;
    using RandomGenerator = RandomGenerator<Type>;
    using Matrix = Matrix<Type>;

    for (int32_t seed = 1; seed < 10; ++seed) {
        RandomGenerator gen(seed, -50, 50);

        for (size_t it = 0; it < RandomGenerator::kIterationCnt; ++it) {
            try {
                int32_t random_id = (gen.GetRandomValue() + 50) % 4;

                if (random_id == 0) {
                    auto [m1, m2] = gen.GenerateEqualDimMatrices();
                    m1 += m2;
                    m2 = m1 + m2;
                } else if (random_id == 1) {
                    auto [m1, m2] = gen.GenerateEqualDimMatrices();
                    m1 -= m2;
                    m2 = m1 - m2;
                } else if (random_id == 2) {
                    auto [m1, m2] = gen.GenerateColRowMatrices();
                    m1 *= m2;
                    m2 = m1 * Matrix::Transposed(m2);
                } else {
                    auto [m1, m2] = gen.GenerateColRowMatrices();
                    m1.Transpose();
                    m2.Conjugate();
                }
            } catch (...) {
                std::cerr << "Stress (operators) FAILED! Seed: " << seed
                          << ", iteration: " << it << std::endl;
                ASSERT_TRUE(false);
                break;
            }
        }
    }
}
} // namespace
