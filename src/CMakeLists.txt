

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/eigenMLP/*.h")

add_library(eigenMLP_lib layer.cpp model.cpp ${HEADER_LIST})

target_include_directories(eigenMLP_lib PUBLIC ../include/eigenMLP)

target_link_libraries(eigenMLP_lib Eigen3::Eigen HighFive)

source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})