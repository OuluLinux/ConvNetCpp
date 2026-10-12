cmake_minimum_required(VERSION 3.18)
project(ConvNetCpp_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 61)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ConvNet/Cuda
    ${CUDA_INCLUDE_DIRS}
)

# Core CUDA Library
add_library(convnet_cuda STATIC
    src/ConvNet/Cuda/CudaMemoryPool.cu
    src/ConvNet/Cuda/CudaKernels.cu
    src/ConvNet/Cuda/CpuBaseline.cpp
    src/ConvNet/Cuda/BatchInference.cpp
)

set_target_properties(convnet_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Benchmarks executable
add_executable(gpu_benchmark src/ConvNet/Cuda/GpuBenchmark.cpp)
target_link_libraries(gpu_benchmark PRIVATE convnet_cuda pthread)

# Evaluator executable
add_executable(gpu_evaluator src/ConvNet/Cuda/GpuEvaluator.cpp)
target_link_libraries(gpu_evaluator PRIVATE convnet_cuda pthread)

# Edge case test executable
add_executable(edge_case_tests src/ConvNet/Cuda/EdgeCaseTestSuite.cpp)
target_link_libraries(edge_case_tests PRIVATE convnet_cuda pthread)
