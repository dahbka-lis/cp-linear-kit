#include <gtest/gtest.h>

#include "../src/algorithms/bidiagonalization.h"
#include "../src/matrix_utils/checks.h"
#include "helpers.h"

namespace {
template <typename T = long double>
using Complex = std::complex<T>;

template <typename T = long double>
using Matrix = LinearKit::Matrix<T>;

using namespace LinearKit::Algorithm;
using namespace LinearKit::Utils;
using namespace LinearKit::MatrixUtils;
using LinearKit::Tests::RandomMatrixGenerator;

template <MatrixType M, MatrixType F, MatrixType S, MatrixType K>
void CheckBidiag(const M &matrix, const F &U, const S &B, const K &VT) {
    EXPECT_TRUE(IsUnitary(U));
    EXPECT_TRUE(IsUnitary(VT));
    EXPECT_TRUE(IsBidiagonal(B));
    EXPECT_TRUE(AreEqualMatrices(matrix, U * B * VT));
}

TEST(TEST_BIDIAG, BidiagClear) {
    using Matrix = Matrix<long double>;

    Matrix matrix;

    auto [U, B, VT] = Bidiagonalize(matrix);
    CheckBidiag(matrix, U, B, VT);
}

TEST(TEST_BIDIAG, BidiagSquare) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {
        {1, 2, 2, 4}, {6, 6, 7, 8}, {9, 10, 11, 11}, {12, 13, 14, 17}};

    auto [U, B, VT] = Bidiagonalize(matrix);
    CheckBidiag(matrix, U, B, VT);
}

TEST(TEST_BIDIAG, BidiagRectangle) {
    using Matrix = Matrix<long double>;

    {
        Matrix matrix = {{1, 2, 3, 4, 5, 6}};

        auto [U, B, VT] = Bidiagonalize(matrix);
        CheckBidiag(matrix, U, B, VT);
    }
    {
        Matrix matrix = {{4, 4, 5}, {4, 1, 2}, {7, 9, 3}, {1, 1, 2}};

        auto [U, B, VT] = Bidiagonalize(matrix);
        CheckBidiag(matrix, U, B, VT);
    }
}

TEST(TEST_BIDIAG, BidiagComplex) {
    using Matrix = Matrix<Complex<long double>>;

    Matrix matrix = {{{1, 2}, {3, 4}, {-1, 2}}, {{7, -3}, {-3, 2}, {-5, -3}}};

    auto [U, B, VT] = Bidiagonalize(matrix);
    CheckBidiag(matrix, U, B, VT);
}

TEST(TEST_BIDIAG, BidiagView) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {
        {1, 2, 2, 4}, {6, 6, 7, 8}, {9, 10, 11, 11}, {12, 13, 14, 17}};
    auto view = matrix.GetSubmatrix({1, -1}, {0, -1});

    auto [U, B, VT] = Bidiagonalize(view);
    CheckBidiag(view, U, B, VT);
}

TEST(TEST_BIDIAG, Stress) {
    using Type = Complex<long double>;
    using MatrixGenerator = RandomMatrixGenerator<Type>;
    using Matrix = Matrix<Type>;

    const size_t it_count = 10u;

    for (int32_t seed = 1; seed < 10; ++seed) {
        MatrixGenerator gen(seed);

        for (size_t it = 0; it < it_count; ++it) {
            int32_t rows = gen.GetRandomMatrixSize();
            int32_t columns = gen.GetRandomMatrixSize();

            auto matrix = gen.GetRandomMatrix(rows, columns);
            auto [U, B, VT] = Bidiagonalize(matrix);
            CheckBidiag(matrix, U, B, VT);
        }
    }
}
} // namespace
