cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(eigneMLP)
set(CMAKE_CXX_STANDARD 11)

find_package(HighFive 2.1 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(eigenmlp main.cpp)
target_link_libraries(eigenmlp HighFive Eigen3::Eigen)