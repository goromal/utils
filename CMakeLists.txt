cmake_minimum_required (VERSION 2.8.11)
project (utils)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

add_library(utils INTERFACE)
target_include_directories(utils
  INTERFACE
  include
)
