# Minimum version for CUDA
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# CUDA required
project(procell LANGUAGES CXX CUDA)

# Binaries directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Update sources directory
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}
)

# Source files
add_executable(
    procell
    ${PROJECT_SOURCE_DIR}/main.cu
    ${PROJECT_SOURCE_DIR}/simulation/cell.h
    ${PROJECT_SOURCE_DIR}/simulation/cell.cu
    ${PROJECT_SOURCE_DIR}/simulation/cells_population.h
    ${PROJECT_SOURCE_DIR}/simulation/cells_population.cu
    ${PROJECT_SOURCE_DIR}/utils/util.h
    ${PROJECT_SOURCE_DIR}/utils/util.cu
)

# C++11
target_compile_features(procell PUBLIC cxx_std_11)

# Enable separable compilation
set_target_properties(procell PROPERTIES CUDA_SEPARABLE_COMPILATION ON)