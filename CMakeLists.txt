cmake_minimum_required(VERSION 3.2)
project(PlanarMotionSIFT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

SET (CMAKE_BUILD_TYPE "RELEASE")

if (CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
endif(CMAKE_COMPILER_IS_GNUCXX)

find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

include_directories(
    src/include
)

set(SRCS
    src/solvers/solver_choi_kim_2018.cpp
    src/solvers/solver_guan_cvpr_2020_cs.cpp
    src/solvers/solver_guan_cvpr_2020_ls.cpp
    src/solvers/solver_valtonen_ornhag_eccv_2024.cpp
)

# Benchmark
add_executable(benchmark benchmark/benchmark.cpp ${SRCS})
target_link_libraries(benchmark PRIVATE Eigen3::Eigen)

