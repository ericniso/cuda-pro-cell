# Minimum version for CUDA
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# CUDA required
project(procell LANGUAGES CXX CUDA)

# Use c++11 standard
set(CMAKE_CXX_STANDARD 11)

# Binaries directory
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Update sources directory
SET(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}
)

# Source files
add_executable(
    procell
    ${PROJECT_SOURCE_DIR}/main.cpp
    ${PROJECT_SOURCE_DIR}/data/data_structures.h
)
