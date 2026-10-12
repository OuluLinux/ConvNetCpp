# Makefile for ConvNetCpp CUDA backend, benchmarks, tutorials, and Classify2D CUDA
NVCC := /opt/cuda/bin/nvcc
CXX := g++-13
CUDA_PATH := /opt/cuda
ARCH := sm_61

NVCCFLAGS := -ccbin $(CXX) -arch=$(ARCH) -O3 --compiler-options "-Wall -O3" -std=c++17 -I$(CUDA_PATH)/include -Isrc/ConvNet/Cuda -Isrc
CXXFLAGS := -O3 -Wall -Wextra -std=c++17 -I$(CUDA_PATH)/include -Isrc/ConvNet/Cuda -Isrc
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -Wl,-rpath,$(CUDA_PATH)/lib64 -lpthread

BIN_DIR := bin
BUILD_DIR := build/obj

OBJS := \
	$(BUILD_DIR)/CudaMemoryPool.o \
	$(BUILD_DIR)/CudaKernels.o \
	$(BUILD_DIR)/CpuBaseline.o \
	$(BUILD_DIR)/BatchInference.o

LIB_TARGET := $(BUILD_DIR)/libconvnet_cuda.a
BENCH_TARGET := $(BIN_DIR)/gpu_benchmark
EVAL_TARGET := $(BIN_DIR)/gpu_evaluator
EDGE_TARGET := $(BIN_DIR)/edge_case_tests

# New CUDA tutorials and Classify2D
TUT03_TARGET := $(BIN_DIR)/tutorial03_cuda
TUT04_TARGET := $(BIN_DIR)/tutorial04_cuda
TUT05_TARGET := $(BIN_DIR)/tutorial05_cuda
TUT06_TARGET := $(BIN_DIR)/tutorial06_cuda
CLASSIFY2D_TARGET := $(BIN_DIR)/classify2d_cuda

.PHONY: all clean test bench eval tutorials classify2d directories

all: directories $(LIB_TARGET) $(BENCH_TARGET) $(EVAL_TARGET) $(EDGE_TARGET) tutorials $(CLASSIFY2D_TARGET)

directories:
	@mkdir -p $(BIN_DIR) $(BUILD_DIR)

$(BUILD_DIR)/CudaMemoryPool.o: src/ConvNet/Cuda/CudaMemoryPool.cu src/ConvNet/Cuda/CudaMemoryPool.h src/ConvNet/Cuda/CudaUtils.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/CudaKernels.o: src/ConvNet/Cuda/CudaKernels.cu src/ConvNet/Cuda/CudaKernels.h src/ConvNet/Cuda/CudaUtils.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/CpuBaseline.o: src/ConvNet/Cuda/CpuBaseline.cpp src/ConvNet/Cuda/CpuBaseline.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/BatchInference.o: src/ConvNet/Cuda/BatchInference.cpp src/ConvNet/Cuda/BatchInference.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIB_TARGET): $(OBJS)
	ar rcs $@ $(OBJS)

$(BENCH_TARGET): src/ConvNet/Cuda/GpuBenchmark.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

$(EVAL_TARGET): src/ConvNet/Cuda/GpuEvaluator.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

$(EDGE_TARGET): src/ConvNet/Cuda/EdgeCaseTestSuite.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

# Tutorials
$(TUT03_TARGET): tutorial/ConvNet03_CudaBasics/ConvNet03_CudaBasics.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

$(TUT04_TARGET): tutorial/ConvNet04_CudaKernels/ConvNet04_CudaKernels.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

$(TUT05_TARGET): tutorial/ConvNet05_CudaBatchInference/ConvNet05_CudaBatchInference.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

$(TUT06_TARGET): tutorial/ConvNet06_CudaConv2D/ConvNet06_CudaConv2D.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< $(LIB_TARGET) $(LDFLAGS) -o $@

tutorials: $(TUT03_TARGET) $(TUT04_TARGET) $(TUT05_TARGET) $(TUT06_TARGET)

# Classify2D CUDA
$(BUILD_DIR)/Classify2DCuda.o: examples/Classify2D_Cuda/Classify2DCuda.cu examples/Classify2D_Cuda/Classify2DCuda.h
	$(NVCC) $(NVCCFLAGS) -Iexamples/Classify2D_Cuda -c $< -o $@

$(CLASSIFY2D_TARGET): examples/Classify2D_Cuda/main.cpp $(BUILD_DIR)/Classify2DCuda.o $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -Iexamples/Classify2D_Cuda $< $(BUILD_DIR)/Classify2DCuda.o $(LIB_TARGET) $(LDFLAGS) -o $@

classify2d: $(CLASSIFY2D_TARGET)
	@$(CLASSIFY2D_TARGET)

test: $(EDGE_TARGET) $(EVAL_TARGET)
	@echo "Running Edge Case Test Suite..."
	@$(EDGE_TARGET)
	@echo "Running Automated Evaluator..."
	@$(EVAL_TARGET)

eval: $(EVAL_TARGET)
	@$(EVAL_TARGET)

bench: $(BENCH_TARGET)
	@$(BENCH_TARGET)

clean:
	rm -rf build/obj $(BENCH_TARGET) $(EVAL_TARGET) $(EDGE_TARGET) \
		$(TUT03_TARGET) $(TUT04_TARGET) $(TUT05_TARGET) $(TUT06_TARGET) $(CLASSIFY2D_TARGET) \
		benchmark_results.json benchmark_results.md
