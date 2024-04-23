#include <gtest/gtest.h>

#include "../algorithms/qr_algorithm.h"
#include "../matrix_utils/checks.h"
#include "helpers.h"

namespace {
template <typename T = long double>
using Complex = std::complex<T>;

template <typename T = long double>
using Matrix = matrix_lib::Matrix<T>;

using namespace matrix_lib::algorithms;
using namespace matrix_lib::utils;
using matrix_lib::tests::RandomMatrixGenerator;

template <MatrixType M, MatrixType F, MatrixType S>
void CheckSpectral(const M &matrix, const F &D, const S &Q) {
    using T = typename M::ElemType;

    auto eps = 1e-10l;

    EXPECT_TRUE(IsUnitary(Q, eps));
    EXPECT_TRUE(
        AreEqualMatrices(matrix, Q * D * Matrix<T>::Conjugated(Q), eps));
}

TEST(TEST_SPECTRAL, SpectralClear) {
    using Matrix = Matrix<long double>;

    Matrix matrix;

    auto [D, Q] = GetSpecDecomposition(matrix);
    CheckSpectral(matrix, D, Q);
}

TEST(TEST_SPECTRAL, SpectralReal) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}};

    auto [D, Q] = GetSpecDecomposition(matrix);
    CheckSpectral(matrix, D, Q);
}

TEST(TEST_SPECTRAL, SpectralComplex) {
    using Matrix = Matrix<Complex<long double>>;

    Matrix matrix = {{{1, 0}, {2, 2}, {3, 3}},
                     {{2, -2}, {5, 0}, {6, 6}},
                     {{3, -3}, {6, -6}, {9, 0}}};

    auto [D, Q] = GetSpecDecomposition(matrix);
    CheckSpectral(matrix, D, Q);
}

TEST(TEST_SPECTRAL, SpectralView) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {{1, 2, 3, 4}, {2, 5, 6, 7}, {3, 6, 5, 8}, {4, 7, 8, 9}};
    auto view = matrix.GetSubmatrix({1, -1}, {1, -1});

    auto [D, Q] = GetSpecDecomposition(view);
    CheckSpectral(view, D, Q);
}

TEST(TEST_SPECTRAL, Stress) {
    using Type = long double;
    using MatrixGenerator = RandomMatrixGenerator<Type>;
    using Matrix = Matrix<Type>;

    const size_t it_count = 1u;

    for (int32_t seed = 1; seed < 10; ++seed) {
        MatrixGenerator gen(seed);

        for (size_t it = 0; it < it_count; ++it) {
            int32_t size = gen.GetRandomMatrixSize();
            auto matrix = gen.GetRandomSymmetricMatrix(size);
            auto [D, Q] = GetSpecDecomposition(matrix);
            CheckSpectral(matrix, D, Q);
        }
    }
}
} // namespace
