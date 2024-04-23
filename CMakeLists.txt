cmake_minimum_required(VERSION 3.26)
project(cp_linalg_lib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-p=${CMAKE_CURRENT_SOURCE_DIR}")

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

include(GoogleTest)
file(GLOB TEST_FILES "tests/*.cpp")

add_executable(test_all ${TEST_FILES})
target_link_libraries(test_all PRIVATE gtest_main)

foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE gtest_main)
    gtest_discover_tests(${TEST_NAME})
endforeach()

add_executable(main main.cpp)
