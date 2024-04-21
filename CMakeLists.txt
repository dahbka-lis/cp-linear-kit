cmake_minimum_required(VERSION 3.26)
project(cp_linalg_lib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-p=${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(main main.cpp)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

file(GLOB TEST_FILES "tests/*.cpp")

add_executable(test_matrix tests/test_matrix.cpp)
add_executable(test_matrix_view tests/test_matrix_view.cpp)
add_executable(test_qr_decomposition tests/test_qr_decomposition.cpp)
add_executable(test_bidiag tests/test_bidiag.cpp)
add_executable(test_hessenberg tests/test_hessenberg.cpp)
add_executable(test_all ${TEST_FILES})

target_link_libraries(test_matrix PUBLIC gtest_main)
target_link_libraries(test_matrix_view PUBLIC gtest_main)
target_link_libraries(test_qr_decomposition PUBLIC gtest_main)
target_link_libraries(test_bidiag PUBLIC gtest_main)
target_link_libraries(test_hessenberg PUBLIC gtest_main)
target_link_libraries(test_all PUBLIC gtest_main)

include(GoogleTest)
gtest_discover_tests(test_matrix test_matrix_view test_qr_decomposition test_bidiag test_hessenberg)
