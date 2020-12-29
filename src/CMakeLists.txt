
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

file(GLOB EIGEN_FORWARD_INCLUDE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/eigenMLP/*.h")

add_library(eigenForward_lib layer.cpp model.cpp ${EIGEN_FORWARD_INCLUDE})

target_include_directories(eigenForward_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/eigenMLP)

find_package(HighFive REQUIRED)

target_link_libraries(eigenForward_lib Eigen3::Eigen HighFive)
