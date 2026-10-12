#include <cuda_runtime.h>

// BUG: Naive single-thread memory reads with inverted thread-to-matrix mapping
// Leads to 100% uncoalesced global DRAM transactions, bottlenecking memory bus and missing perf target.
__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Inverted thread mapping: threadIdx.x iterates down rows instead of across adjacent columns
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        if (bias != nullptr) sum += bias[col];
        C[row * N + col] = (sum > 0.0f) ? sum : 0.0f;
    }
}

extern "C" void launch_fused_gemm_bias_relu(
    const float* d_A, const float* d_B, const float* d_bias, float* d_C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    fused_gemm_bias_relu_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_bias, d_C, M, N, K);
}
