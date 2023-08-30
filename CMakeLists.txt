cmake_minimum_required(VERSION 3.20)
project(OP_RBF)

set(CMAKE_CXX_STANDARD 14)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-mfma -mavx2 -g ${CMAKE_CXX_FLAGS}")
    message(STATUS "optional: -mfma")
endif(CMAKE_COMPILER_IS_GNUCXX)

include_directories(.)

add_executable(OP_RBF
        rbf.hpp
        RBFilter_SSE2.cpp
        RBFilter_SSE2.h
        RBFilter_AVX2.cpp
        RBFilter_AVX2.h
        RBFilterPlain.cpp
        RBFilterPlain.h
        RecursiveBilateralFilter.cpp
        stb_image.h
        stb_image_write.h
        stdafx.cpp)

target_link_libraries(OP_RBF pthread)
