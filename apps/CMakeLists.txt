add_executable(app main.cpp)
target_link_libraries(app Eigen3::Eigen eigenMLP_lib)