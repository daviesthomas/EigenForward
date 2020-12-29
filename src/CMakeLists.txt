
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/eigenMLP/*.h")

add_library(eigenForward_lib layer.cpp model.cpp ${HEADER_LIST})

target_include_directories(eigenForward_lib PUBLIC ../include/eigenMLP)

find_package(HighFive REQUIRED)

target_link_libraries(eigenForward_lib Eigen3::Eigen HighFive)

source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})