cmake_minimum_required(VERSION 3.12)
project(myGalerkin)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES gasdin2d.cpp CSR.cpp)
add_executable(myGalerkin gasdin2d.cpp gasdin2d.h CSR.cpp CSR.h targetver.h)

include_directories("/home/h0las/University/Cuda/myGalerkin/")
include_directories("/home/h0las/University/Cuda/AMGX_portable/base/include/")
target_link_libraries(myGalerkin "/home/h0las/University/Cuda/AMGX_portable/build/libamgxsh.so")