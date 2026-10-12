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

# Benchmarks & Evaluators
add_executable(gpu_benchmark src/ConvNet/Cuda/GpuBenchmark.cpp)
target_link_libraries(gpu_benchmark PRIVATE convnet_cuda pthread)

add_executable(gpu_evaluator src/ConvNet/Cuda/GpuEvaluator.cpp)
target_link_libraries(gpu_evaluator PRIVATE convnet_cuda pthread)

add_executable(edge_case_tests src/ConvNet/Cuda/EdgeCaseTestSuite.cpp)
target_link_libraries(edge_case_tests PRIVATE convnet_cuda pthread)

# CUDA Tutorials
add_executable(tutorial03_cuda tutorial/ConvNet03_CudaBasics/ConvNet03_CudaBasics.cpp)
target_link_libraries(tutorial03_cuda PRIVATE convnet_cuda pthread)

add_executable(tutorial04_cuda tutorial/ConvNet04_CudaKernels/ConvNet04_CudaKernels.cpp)
target_link_libraries(tutorial04_cuda PRIVATE convnet_cuda pthread)

add_executable(tutorial05_cuda tutorial/ConvNet05_CudaBatchInference/ConvNet05_CudaBatchInference.cpp)
target_link_libraries(tutorial05_cuda PRIVATE convnet_cuda pthread)

add_executable(tutorial06_cuda tutorial/ConvNet06_CudaConv2D/ConvNet06_CudaConv2D.cpp)
target_link_libraries(tutorial06_cuda PRIVATE convnet_cuda pthread)

# Classify2D CUDA
add_library(classify2d_cuda_lib STATIC
    examples/Classify2D_Cuda/Classify2DCuda.cu
)
target_include_directories(classify2d_cuda_lib PUBLIC examples/Classify2D_Cuda)

add_executable(classify2d_cuda examples/Classify2D_Cuda/main.cpp)
target_include_directories(classify2d_cuda PUBLIC examples/Classify2D_Cuda)
target_link_libraries(classify2d_cuda PRIVATE classify2d_cuda_lib convnet_cuda pthread)
