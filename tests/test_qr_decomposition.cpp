#include <gtest/gtest.h>

#include "../algorithms/qr_decomposition.h"
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
void CheckQR(const M &matrix, const F &Q, const S &R) {
    EXPECT_TRUE(IsUnitary(Q));
    EXPECT_TRUE(IsUpperTriangular(R));
    EXPECT_TRUE(AreEqualMatrices(matrix, Q * R));
}

TEST(TEST_QR_DECOMPOSITION, HouseholderClear) {
    using Matrix = Matrix<long double>;

    Matrix matrix;

    auto [Q, R] = HouseholderQR(matrix);
    CheckQR(matrix, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, HouseholderSquare) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto [Q, R] = HouseholderQR(matrix);
    CheckQR(matrix, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, HouseholderRectangle) {
    using Matrix = Matrix<long double>;

    {
        Matrix matrix = {{5, 4, 3, 2, 1}, {9, 4, 8, 1, 3}};

        auto [Q, R] = HouseholderQR(matrix);
        CheckQR(matrix, Q, R);
    }
    {
        Matrix matrix = {{1}, {2}, {3}, {4}, {5}};

        auto [Q, R] = HouseholderQR(matrix);
        CheckQR(matrix, Q, R);
    }
}

TEST(TEST_QR_DECOMPOSITION, HouseholderComplex) {
    using Matrix = Matrix<Complex<long double>>;

    Matrix matrix = {{{2, 3}, {0, 4}, {1, 2}}, {{0, 3}, {3, 0}, {1, -3}}};

    auto [Q, R] = HouseholderQR(matrix);
    CheckQR(matrix, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, HouseholderView) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto view = matrix.GetSubmatrix({0, -1}, {1, -1});

    auto [Q, R] = HouseholderQR(view);
    CheckQR(view, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, GivensClear) {
    using Matrix = Matrix<long double>;

    Matrix matrix;

    auto [Q, R] = GivensQR(matrix);
    CheckQR(matrix, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, GivensSquare) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {{5, 8, 1}, {0, 1, 4}, {7, 3, 2}};

    auto [Q, R] = GivensQR(matrix);
    CheckQR(matrix, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, GivensRectangle) {
    using Matrix = Matrix<long double>;

    {
        Matrix matrix = {{6, 6, 3, 1, 4}};

        auto [Q, R] = GivensQR(matrix);
        CheckQR(matrix, Q, R);
    }
    {
        Matrix matrix = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}};

        auto [Q, R] = GivensQR(matrix);
        CheckQR(matrix, Q, R);
    }
}

TEST(TEST_QR_DECOMPOSITION, GivensComplex) {
    using Matrix = Matrix<Complex<long double>>;

    Matrix matrix = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};

    auto [Q, R] = GivensQR(matrix);
    CheckQR(matrix, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, GivensView) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {{5, 8, 1}, {0, 1, 4}, {7, 3, 2}};
    auto view = matrix.GetColumn(0);

    auto [Q, R] = GivensQR(view);
    CheckQR(view, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, HessenbergClear) {
    using Matrix = Matrix<long double>;

    Matrix matrix;

    auto [Q, R] = HessenbergQR(matrix);
    CheckQR(matrix, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, HessenbergSquare) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {{9, -3, 1, 3}, {2, 3, 0, -1}, {0, 1, 4, 3}, {0, 0, -4, 5}};

    auto [Q, R] = HessenbergQR(matrix);
    CheckQR(matrix, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, HessenbergRectangle) {
    using Matrix = Matrix<long double>;

    {
        Matrix matrix = {{1, -2, 3, 4}, {6, 5, 7, 8}};

        auto [Q, R] = HessenbergQR(matrix);
        CheckQR(matrix, Q, R);
    }
    {
        Matrix matrix = {{1, -3}, {9, 4}, {0, 5}, {0, 0}};

        auto [Q, R] = HessenbergQR(matrix);
        CheckQR(matrix, Q, R);
    }
}

TEST(TEST_QR_DECOMPOSITION, HessenbergComplex) {
    using Matrix = Matrix<Complex<long double>>;

    Matrix matrix = {{{1, 2}, {7, 3}, {0, 3}}, {{0, -4}, {1, 1}, {-5, -5}}};

    auto [Q, R] = HessenbergQR(matrix);
    CheckQR(matrix, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, HessenbergView) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {{9, -3, 1, 3}, {2, 3, 0, -1}, {0, 1, 4, 3}, {0, 0, -4, 5}};
    auto view = matrix.GetSubmatrix({0, -1}, {0, 2});

    auto [Q, R] = HessenbergQR(view);
    CheckQR(view, Q, R);
}

TEST(TEST_QR_DECOMPOSITION, Stress) {
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

            auto id = it % 2;
            if (id == 0 || rows >= 70) {
                auto [Q, R] = HouseholderQR(matrix);
                CheckQR(matrix, Q, R);
            } else if (id == 1) {
                auto [Q, R] = GivensQR(matrix);
                CheckQR(matrix, Q, R);
            } else {
                matrix.ApplyForEach([](auto &val, auto i, auto j) {
                    if (i > j + 1)
                        val = Complex<long double>(0);
                });

                auto [Q, R] = HessenbergQR(matrix);
                CheckQR(matrix, Q, R);
            }
        }
    }
}
} // namespace
