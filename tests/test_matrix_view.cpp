#include <gtest/gtest.h>

#include "helpers.h"

namespace {
    template <typename T = long double>
    using Complex = std::complex<T>;

    template <typename T = long double>
    using Matrix = matrix_lib::Matrix<T>;

    using matrix_lib::tests::RandomMatrixGenerator;
    using matrix_lib::utils::IsEqualFloating;

    TEST(TEST_MATRIX_VIEW, BasicConstructors) {
    }

    TEST(TEST_MATRIX_VIEW, CopySemantics) {
    }

    TEST(TEST_MATRIX_VIEW, MoveSemantics) {
    }

    void CheckArithmeticSum() {
    }

    void CheckArithmeticDiff() {
    }

    void CheckArithmeticMulti() {
    }

    TEST(TEST_MATRIX_VIEW, Arithmetic) {
        CheckArithmeticSum();
        CheckArithmeticDiff();
        CheckArithmeticMulti();
    }

    TEST(TEST_MATRIX_VIEW, Transpose) {
    }

    TEST(TEST_MATRIX_VIEW, Conjugate) {
    }

    TEST(TEST_MATRIX_VIEW, ApplyToEach) {
    }

    TEST(TEST_MATRIX_VIEW, Normalize) {
    }

    TEST(TEST_MATRIX_VIEW, DiagonalMatrix) {
    }

    TEST(TEST_MATRIX_VIEW, Stress) {
    }
} // namespace
