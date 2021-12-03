# ESSENTIAL MATRIX BENCHMARK

# minimum version of cmake
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD "17")
if(NOT CMAKE_CXX_STANDARD_REQUIRED)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# release
set(CMAKE_BUILD_TYPE "Release")

# faster evaluation
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native" CACHE STRING "" FORCE)

# faster evaluation of math
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math" CACHE STRING "" FORCE)

# faster evaluation of Eigen
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx -mfma")

# build in bin folder
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
SET(CMAKE_INSTALL_RUNTIME_DESTINATION bin)

# set the project
project( TWO_VIEW_MINIMAL_SOLVER_PIPELINE )

# add packages
find_package(Eigen3 REQUIRED)
include_directories( ${EIGEN3_INCLUDE_DIRS} )
include_directories(src)

# add subdirectories
add_subdirectory(src)
