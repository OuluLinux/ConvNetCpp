#include <cuda_runtime.h>
#include <cstdio>

// Task: Implement fused GEMM with Bias and ReLU: C = ReLU(A * B + bias)
// A: [M, K], B: [K, N], bias: [N], C: [M, N]
// All matrices in row-major layout.

__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    // TODO: Implement high-performance tiled shared-memory GEMM with fused bias and ReLU.
    // Ensure boundary handling for non-tile dimensions.
}

extern "C" void launch_fused_gemm_bias_relu(
    const float* d_A,
    const float* d_B,
    const float* d_bias,
    float* d_C,
    int M, int N, int K,
    cudaStream_t stream)
{
    // TODO: Configure launch grid and block dimensions, then launch kernel.
}
