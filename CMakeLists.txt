# Minimum version for CUDA
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# CUDA required
project(procell LANGUAGES C CXX CUDA)

# Root dir
set(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR})

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
    ${PROJECT_SOURCE_DIR}/simulation/data_types.h
    ${PROJECT_SOURCE_DIR}/simulation/cell.h
    ${PROJECT_SOURCE_DIR}/simulation/cell.cu
    ${PROJECT_SOURCE_DIR}/simulation/cells_population.h
    ${PROJECT_SOURCE_DIR}/simulation/cells_population.cu
    ${PROJECT_SOURCE_DIR}/simulation/proliferation.h
    ${PROJECT_SOURCE_DIR}/simulation/proliferation.cu
    ${PROJECT_SOURCE_DIR}/simulation/simulator.h
    ${PROJECT_SOURCE_DIR}/simulation/simulator.cu
    ${PROJECT_SOURCE_DIR}/utils/util.h
    ${PROJECT_SOURCE_DIR}/utils/util.cu
    ${PROJECT_SOURCE_DIR}/io/parser.h
    ${PROJECT_SOURCE_DIR}/io/parser.cu
    ${PROJECT_SOURCE_DIR}/io/cmdargs.h
    ${PROJECT_SOURCE_DIR}/io/cmdargs.cpp
)

# C++98
target_compile_features(procell PUBLIC cxx_std_98)

# Enable separable compilation
set_target_properties(procell PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
