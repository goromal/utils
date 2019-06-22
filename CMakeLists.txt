cmake_minimum_required (VERSION 2.8.11)
project (utils)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
  include
  ${YAML_CPP_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_library(utils INTERFACE)
target_include_directories(utils
  INTERFACE
  include
  ${YAML_CPP_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)
target_link_libraries(utils ${YAML_CPP_LIBRARIES})
