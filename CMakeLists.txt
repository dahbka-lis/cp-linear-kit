cmake_minimum_required(VERSION 3.26)
project(cp_linalg_lib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-p=${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(main main.cpp)
