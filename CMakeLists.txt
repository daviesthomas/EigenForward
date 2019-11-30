cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(
    mlp
    DESCRIPTION "A simple library for eigen inference from keras weights"
)

set(CMAKE_CXX_STANDARD 11)

find_package(HighFive 2.1 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(src)
add_subdirectory(apps)


