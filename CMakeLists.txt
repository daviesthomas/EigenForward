cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(
    eigenForward
    DESCRIPTION "A simple library for eigen inference from keras weights"
)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(HighFive_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/submodules)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (HighFive REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(submodules/HighFive)
add_subdirectory(src)   
add_subdirectory(apps)

