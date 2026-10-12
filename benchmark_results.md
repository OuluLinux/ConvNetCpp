# ConvNetCpp GPU Performance Benchmark Report

**Hardware:** NVIDIA GeForce GTX 1060 6GB (Compute 6.1, 10 SMs, 192.2 GB/s Bandwidth, 4.57 TFLOPS FP32)

## 1. Matrix Multiplication (GEMM)

| Matrix Size | CPU Baseline (ms) | Naive GPU (ms) | Tiled GPU (ms) | Fused GPU (ms) | FP16 GPU (ms) | Tiled GFLOPS | Speedup vs CPU |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 128x128 | 3.04 | 0.022 | 0.01 | 0.01 | 0.01 | 380.7 | **275.6x** |
| 256x256 | 23.64 | 0.141 | 0.07 | 0.07 | 0.08 | 464.0 | **327.0x** |
| 512x512 | 283.82 | 1.110 | 0.53 | 0.53 | 0.59 | 508.3 | **537.4x** |
| 1024x1024 | 2270.59 | 10.58 | 4.46 | 4.03 | 4.07 | 481.9 | **509.6x** |
| 2048x2048 | 18164.70 | N/A | 28.74 | 28.71 | 30.15 | 597.8 | **632.1x** |

## 2. 2D Convolution (Conv2D)

| Layer Name | CPU Time (ms) | GPU Time (ms) | Throughput (GFLOPS) | Throughput (images/s) | Speedup |
|:---|:---|:---|:---|:---|:---|
| CIFAR-10 Conv1 (3->32, 32x32) | 74.50 | 0.33 | 172.2 | 97306.6 | **226.5x** |
| Mid-layer Conv2 (32->64, 32x32) | 773.76 | 3.04 | 198.7 | 5263.6 | **254.5x** |
| Deep Feature Conv3 (64->128, 16x16) | 344.59 | 1.52 | 198.3 | 5253.5 | **226.3x** |

## 3. GPU Memory Pooling Efficiency

| Block Size | Iterations | Raw `cudaMalloc` (ms) | `CudaMemoryPool` (ms) | Allocation Speedup |
|:---|:---|:---|:---|:---|
| 64 KB | 500 | 57.38 | 0.13 | **427.7x** |
| 1024 KB | 500 | 2.76 | 0.13 | **21.2x** |
| 16384 KB | 500 | 225.83 | 0.13 | **1747.1x** |
