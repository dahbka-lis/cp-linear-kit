#include <gtest/gtest.h>

#include "../algorithms/hessenberg.h"
#include "../matrix_utils/checks.h"
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

template <MatrixType M, MatrixType F, MatrixType S>
void CheckHessenberg(const M &matrix, const F &U, const S &H) {
    using T = typename M::ElemType;

    EXPECT_TRUE(IsUnitary(U));
    EXPECT_TRUE(IsHessenberg(H));
    EXPECT_TRUE(AreEqualMatrices(matrix, U * H * Matrix<T>::Conjugated(U)));
}

TEST(TEST_HESSENBERG, HessenbergClear) {
    using Matrix = Matrix<long double>;

    Matrix matrix;

    auto [H, U] = GetHessenbergForm(matrix);
    CheckHessenberg(matrix, U, H);
}

TEST(TEST_HESSENBERG, HessenbergSquare) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    auto [H, U] = GetHessenbergForm(matrix);
    CheckHessenberg(matrix, U, H);
}

TEST(TEST_HESSENBERG, HessenbergComplex) {
    using Matrix = Matrix<Complex<long double>>;

    Matrix matrix = {{{1, 2}, {3, 4}, {-1, 2}},
                     {{7, -3}, {-3, 2}, {-5, -3}},
                     {{-6, 3}, {0, -1}, {8, 9}}};

    auto [H, U] = GetHessenbergForm(matrix);
    CheckHessenberg(matrix, U, H);
}

TEST(TEST_HESSENBERG, HessenbergView) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    auto view = matrix.GetSubmatrix({1, -1}, {1, -1});

    auto [H, U] = GetHessenbergForm(view);
    CheckHessenberg(view, U, H);
}

TEST(TEST_HESSENBERG, Stress) {
    using Type = Complex<long double>;
    using MatrixGenerator = RandomMatrixGenerator<Type>;
    using Matrix = Matrix<Type>;

    const size_t it_count = 10u;

    for (int32_t seed = 1; seed < 10; ++seed) {
        MatrixGenerator gen(seed);

        for (size_t it = 0; it < it_count; ++it) {
            int32_t size = gen.GetRandomMatrixSize();

            auto matrix = gen.GetRandomMatrix(size, size);
            auto [H, U] = GetHessenbergForm(matrix);
            CheckHessenberg(matrix, U, H);
        }
    }
}
} // namespace
