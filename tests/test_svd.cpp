#include <gtest/gtest.h>

#include "../src/algorithms/svd.h"
#include "helpers.h"

namespace {
template <typename T = long double>
using Complex = std::complex<T>;

template <typename T = long double>
using Matrix = LinearKit::Matrix<T>;
using IndexType = LinearKit::Details::Types::IndexType;

using namespace LinearKit::Algorithm;
using namespace LinearKit::Utils;
using namespace LinearKit::MatrixUtils;
using LinearKit::Tests::RandomGenerator;
using LinearKit::Utils::Details::IsFloatComplexT;

template <MatrixType M>
void CheckPositiveSingular(const M &S) {
    using T = typename M::ElemType;

    for (IndexType i = 0; i < S.Columns(); ++i) {
        long double sing_value;

        if constexpr (IsFloatComplexT<T>::value) {
            sing_value = static_cast<long double>(S(0, i).real());
        } else {
            sing_value = static_cast<long double>(S(0, i));
        }

        EXPECT_TRUE(sing_value >= 0.l);
    }
}

template <MatrixType M, MatrixType F, MatrixType L, MatrixType K>
void CheckSVD(const M &matrix, const F &U, const L &S, const K &VT) {
    using T = M::ElemType;
    auto eps = 1e-10l;

    CheckPositiveSingular(S);
    EXPECT_TRUE(IsUnitary(U, eps));
    EXPECT_TRUE(IsUnitary(VT, eps));

    auto S_full = Matrix<T>::Diagonal(S, U.Columns(), VT.Rows());
    EXPECT_TRUE(AreEqualMatrices(matrix, U * S_full * VT, eps));
}

RandomGenerator<long double> generator(91348);

TEST(TEST_SVD, SVDClear) {
    using Matrix = Matrix<long double>;

    Matrix matrix;

    auto [U, S, VT] = SVD(matrix);
    CheckSVD(matrix, U, S, VT);
}

TEST(TEST_SVD, SVDSquare) {
    using Matrix = Matrix<long double>;

    {
        Matrix matrix = {
            {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

        auto [U, S, VT] = SVD(matrix);
        CheckSVD(matrix, U, S, VT);
    }
    {
        for (int32_t i = 1; i <= 10; ++i) {
            auto matrix = generator.GetMatrix(i, i);
            auto [U, S, VT] = SVD(matrix);
            CheckSVD(matrix, U, S, VT);
        }
    }
}

TEST(TEST_SVD, SVDRectangle) {
    using Matrix = Matrix<long double>;

    {
        Matrix matrix = {{5, 4, 3, 2, 1}, {9, 4, 8, 1, 3}};

        auto [U, S, VT] = SVD(matrix);
        CheckSVD(matrix, U, S, VT);
    }
    {
        Matrix matrix = {{1}, {2}, {3}, {4}, {5}};

        auto [U, S, VT] = SVD(matrix);
        CheckSVD(matrix, U, S, VT);
    }
    {
        for (int32_t i = 1; i <= 10; ++i) {
            for (int32_t j = 1; j <= 10; ++j) {
                if (i == j)
                    continue;

                auto matrix = generator.GetMatrix(i, j);
                auto [U, S, VT] = SVD(matrix);
                CheckSVD(matrix, U, S, VT);
            }
        }
    }
}

TEST(TEST_SVD, SVDComplex) {
    using Matrix = Matrix<Complex<long double>>;

    Matrix matrix = {{{1, 0}, {2, 2}, {3, 3}},
                     {{2, 2}, {5, 0}, {6, 6}},
                     {{3, 3}, {6, 6}, {9, 0}}};

    auto [U, S, VT] = SVD(matrix);
    CheckSVD(matrix, U, S, VT);
}

TEST(TEST_SVD, SVDView) {
    using Matrix = Matrix<long double>;

    Matrix matrix = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    auto view = matrix.GetSubmatrix({1, -1}, {1, -1});

    auto [U, S, VT] = SVD(view);
    CheckSVD(view, U, S, VT);
}

TEST(TEST_SVD, Stress) {
    using Type = Complex<long double>;
    using MatrixGenerator = RandomGenerator<Type>;
    using Matrix = Matrix<Type>;

    const size_t it_count = 10u;

    for (int32_t seed = 1; seed < 10; ++seed) {
        MatrixGenerator gen(seed);

        for (size_t it = 0; it < it_count; ++it) {
            int32_t first_size = gen.GetMatrixSize();
            int32_t second_size = gen.GetMatrixSize();

            auto matrix = gen.GetMatrix(first_size, second_size);
            auto [U, S, VT] = SVD(matrix);
            CheckSVD(matrix, U, S, VT);
        }
    }
}
} // namespace
