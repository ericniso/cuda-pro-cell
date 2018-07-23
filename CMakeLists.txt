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

# Gengetopt files directory
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/cmdline)

# Gengetopt
add_custom_command(
    OUTPUT ${PROJECT_SOURCE_DIR}/cmdline/cmdline.c ${PROJECT_SOURCE_DIR}/cmdline/cmdline.h
    DEPENDS ${PROJECT_ROOT_DIR}/conf/ggo.conf
    COMMAND gengetopt -i ${PROJECT_ROOT_DIR}/conf/ggo.conf --output-dir=${PROJECT_SOURCE_DIR}/cmdline/
)


# Source files
add_executable(
    procell
    ${PROJECT_SOURCE_DIR}/main.cu
    ${PROJECT_SOURCE_DIR}/cmdline/cmdline.h
    ${PROJECT_SOURCE_DIR}/cmdline/cmdline.c
    ${PROJECT_SOURCE_DIR}/simulation/data_types.h
    ${PROJECT_SOURCE_DIR}/simulation/cell.h
    ${PROJECT_SOURCE_DIR}/simulation/cell.cu
    ${PROJECT_SOURCE_DIR}/simulation/cells_population.h
    ${PROJECT_SOURCE_DIR}/simulation/cells_population.cu
    ${PROJECT_SOURCE_DIR}/simulation/proliferation.h
    ${PROJECT_SOURCE_DIR}/simulation/proliferation.cu
    ${PROJECT_SOURCE_DIR}/utils/util.h
    ${PROJECT_SOURCE_DIR}/utils/util.cu
    ${PROJECT_SOURCE_DIR}/io/parser.h
    ${PROJECT_SOURCE_DIR}/io/parser.cu
)

# C++11
target_compile_features(procell PUBLIC cxx_std_11)

# Enable separable compilation
set_target_properties(procell PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
