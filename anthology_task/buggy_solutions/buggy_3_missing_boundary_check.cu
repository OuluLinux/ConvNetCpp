#include <cuda_runtime.h>

constexpr int TILE_DIM = 16;

// BUG: Missing boundary guards during tile load and output write!
// Assumes M, N, K are exact multiples of 16. On odd shapes (e.g. 17x31), triggers out-of-bounds memory corruption.
__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;
    int num_tiles = K / TILE_DIM; // BUG: Drops remaining elements when K % 16 != 0!

    for (int t = 0; t < num_tiles; ++t) {
        // BUG: No bounds checks against M, K, N when loading tiles!
        s_A[ty][tx] = A[row * K + (t * TILE_DIM + tx)];
        s_B[ty][tx] = B[(t * TILE_DIM + ty) * N + col];

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    // BUG: No bounds check against row < M && col < N before writing!
    if (bias != nullptr) acc += bias[col];
    C[row * N + col] = (acc > 0.0f) ? acc : 0.0f;
}

extern "C" void launch_fused_gemm_bias_relu(
    const float* d_A, const float* d_B, const float* d_bias, float* d_C,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    fused_gemm_bias_relu_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_bias, d_C, M, N, K);
}
