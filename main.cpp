#include "types/matrix/matrix.h"
#include <iostream>

using namespace matrix_lib;

int main() {
    auto matrix = Matrix<>::Diagonal({1, 2, 3, 4, 5});
    std::cout << matrix / 3 << std::endl;
    return 0;
}
