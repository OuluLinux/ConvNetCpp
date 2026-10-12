# Case Study: GPU GEMM and Fused Epilogue Optimization

## 1. Executive Summary & Problem Formulation

In neural network inference and scientific training pipelines, Dense Matrix Multiplication (GEMM) and Convolution operations account for $> 85\%$ of execution time. In the initial CPU baseline of `ConvNetCpp`, computing a single forward pass of a $1024 \times 1024$ linear layer required **1,585.55 ms** on host CPU. This created an unacceptable throughput bottleneck for high-frequency model inference.

This case study documents the progressive optimization pipeline:
1. Identifying the arithmetic intensity bottleneck in naive GPU execution.
2. Formulating the shared-memory tiling and register-reuse hypothesis.
3. Implementing 2D SRAM tiled GEMM with epilogue fusion (Bias + ReLU).
4. Eliminating driver memory allocation overhead using recycled block pools.
5. Verifying mathematical parity against a golden reference.

---

## 2. Baseline Measurements & Profiling Analysis

### Baseline 1: Single-Threaded Host CPU
- **Implementation**: Triply-nested loop in C++ with standard row-major traversal.
- **Latency ($1024 \times 1024$)**: **1,585.55 ms**
- **Throughput**: ~1.35 GFLOPS
- **Limitation**: CPU cache capacity thrashing on large matrices, limited SIMD utilization.

### Baseline 2: Naive CUDA Global Memory Kernel
```cuda
__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col]; // Global DRAM read on every inner iteration
        }
        C[row * N + col] = sum;
    }
}
```
- **Latency ($1024 \times 1024$)**: **10.03 ms**
- **Throughput**: 214.1 GFLOPS
- **Hardware Platform**: NVIDIA GeForce GTX 1060 (Pascal SM 6.1, 10 SMs, 192.2 GB/s Bandwidth, 4.57 TFLOPS FP32).

### Profiling Breakdown & Bottleneck Root Cause
Analyzing the memory transactions reveals why naive execution underutilizes the GPU:
- Each thread iterates $K$ times through global DRAM.
- To compute matrix $C \in \mathbb{R}^{M \times N}$, each element of $A$ is fetched from global memory $N$ times, and each element of $B$ is fetched $M$ times.
- **Total DRAM Data Movement**:
  $$\text{Bytes} = 2 \times M \times N \times K \times 4\text{ bytes} = 2 \times 1024^3 \times 4 \approx 8.59\text{ GB}$$
- **Arithmetic Intensity**:
  $$\text{AI} = \frac{2 \times 1024^3\text{ FLOPs}}{8.59 \times 10^9\text{ Bytes}} = 0.25\text{ FLOP/byte}$$
- **Roofline Comparison**: With GTX 1060 theoretical peak bandwidth of 192.2 GB/s, an arithmetic intensity of $0.25\text{ FLOP/byte}$ caps maximum attainable performance at:
  $$\text{Perf}_{\text{max}} = 192.2\text{ GB/s} \times 0.25\text{ FLOP/byte} \approx 48.05\text{ GFLOPS}$$
  Even with L2 cache filtering, the GPU execution was severely **memory-bandwidth bound**, with warp issue stalls dominated by `Long Scoreboard` (waiting for DRAM line fills).

---

## 3. Optimization Hypotheses

### Hypothesis 1: 2D Shared-Memory Tiling
By partitioning matrices $A$ and $B$ into square tiles of dimension $T \times T$ ($T=16$) loaded collaboratively into on-chip shared memory (SRAM), threads in a thread block ($16 \times 16 = 256$ threads) can read each element of $A$ and $B$ from DRAM once, and reuse it 16 times directly from fast on-chip SRAM:
- **DRAM Traffic Reduction**: Decreases from $8.59\text{ GB}$ to $\frac{8.59\text{ GB}}{16} \approx 536\text{ MB}$.
- **Arithmetic Intensity**: Increases $16\times$ from $0.25\text{ FLOP/byte}$ to $4.0\text{ FLOP/byte}$.
- **Expected Speedup**: 2.0x - 3.0x reduction in kernel time, moving the kernel from memory-bound to compute-bound.

### Hypothesis 2: Epilogue Fusion (Bias + ReLU)
In standard neural network frameworks, Bias addition and ReLU are often separate elementwise kernel launches. This incurs:
1. Kernel launch overhead (~3-5 $\mu$s per launch).
2. Writing the entire intermediate GEMM result to global DRAM and immediately reading it back in the activation kernel ($2 \times M \times N \times 4\text{ bytes} = 8.38\text{ MB}$).
By fusing Bias addition and activation directly into registers before writing to global memory, we eliminate the DRAM round-trip entirely.

### Hypothesis 3: Memory Pool Driver Bypass
`cudaMalloc` and `cudaFree` are synchronous driver calls that force GPU execution barriers. In steady-state inference, recycling buffers via power-of-2 size class bins in `CudaMemoryPool` should reduce allocation latency by $> 100\times$.

---

## 4. Implementation Details

### Optimized Shared-Memory Tiled Kernel with Fused Epilogue
```cuda
constexpr int TILE_DIM = 16;

__global__ void gemm_fused_bias_activation_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K,
    ActivationType act,
    float alpha, float beta)
{
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_DIM + tx;
        int b_row = t * TILE_DIM + ty;

        // Coalesced collaborative load from DRAM into SRAM
        s_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        s_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float out = alpha * acc;
        if (beta != 0.0f) out += beta * C[row * N + col];
        if (bias != nullptr) out += bias[col];
        C[row * N + col] = apply_activation(out, act); // Fused activation in register!
    }
}
```

---

## 5. Measured Benchmark Results

All benchmarks were executed on an **NVIDIA GeForce GTX 1060 6GB** running driver 580.178.04 and CUDA 12.6.

| Optimization Stage | $1024 \times 1024$ Latency | GFLOPS | Speedup vs CPU | Speedup vs Naive GPU |
|:---|:---|:---|:---|:---|
| **Host CPU Baseline** | 1,585.55 ms | 1.35 | 1.0x | - |
| **Naive GPU Kernel** | 10.03 ms | 214.1 | 158.1x | 1.0x |
| **Tiled Shared-Memory GPU** | 4.04 ms | 531.3 | **392.3x** | **2.48x** |
| **Fused Epilogue (Bias+ReLU)** | 3.86 ms | 556.1 | **410.8x** | **2.60x** |
| **FP16 Half-Precision GPU** | 3.90 ms | 550.6 | **406.5x** | **2.57x** |

### Memory Allocation Microbenchmark (500 Iterations)
| Block Size | Raw `cudaMalloc` / `cudaFree` | `CudaMemoryPool` Recycled | Allocation Speedup |
|:---|:---|:---|:---|
| **64 KB** | 61.70 ms | 0.15 ms | **418.0x** |
| **1 MB** | 3.23 ms | 0.21 ms | **15.4x** |
| **16 MB** | 222.45 ms | 0.19 ms | **1,152.9x** |

---

## 6. Numerical Verification & Correctness

To ensure optimizations did not compromise mathematical fidelity, the results were evaluated across 22 automated test cases in `bin/gpu_evaluator`:
- **Max Absolute Error ($1024 \times 1024$)**: $1.91 \times 10^{-5}$ (Tolerance: $1.0 \times 10^{-4}$)
- **Relative Error**: $< 2.4 \times 10^{-5}$
- **Corner Case Parity**: Zero matrix inputs, prime boundaries ($127 \times 255 \times 131$), non-tile odd dimensions ($17 \times 31 \times 23$), and NaN/Inf sanitization all achieved 100% pass rate.
