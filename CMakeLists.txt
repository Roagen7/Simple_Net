cmake_minimum_required(VERSION 3.22)
project(networks)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(include/eigen-3.4.0)
add_executable(networks src/main.cpp src/SimpleNN.cpp src/Simple_NN.h)
target_link_libraries (networks Eigen3::Eigen)