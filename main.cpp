#include "types/matrix/matrix.h"
#include <iostream>

using namespace linalg_lib;

int main() {
    auto matrix = Matrix<>::Diag({1, 2, 3, 4, 5});
    std::cout << matrix << std::endl;
    return 0;
}
