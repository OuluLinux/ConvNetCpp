#include "CudaKernels.h"
#include <cmath>

namespace ConvNet {
namespace Cuda {

constexpr int TILE_DIM = 16;

__device__ __forceinline__ float apply_activation(float x, ActivationType act) {
    switch (act) {
        case ActivationType::None:
            return x;
        case ActivationType::ReLU:
            return fmaxf(0.0f, x);
        case ActivationType::LeakyReLU:
            return x > 0.0f ? x : 0.01f * x;
        case ActivationType::Sigmoid:
            return 1.0f / (1.0f + __expf(-x));
        case ActivationType::Tanh:
            return tanhf(x);
        case ActivationType::GELU:
            return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        default:
            return x;
    }
}

__device__ __forceinline__ half apply_activation_half(half x, ActivationType act) {
    float f = __half2float(x);
    float out = apply_activation(f, act);
    return __float2half(out);
}

// -------------------------------------------------------------
// Naive GEMM
// -------------------------------------------------------------
__global__ void gemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        if (beta == 0.0f) {
            C[row * N + col] = alpha * sum;
        } else {
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
    }
}

// -------------------------------------------------------------
// Tiled Shared-Memory GEMM
// -------------------------------------------------------------
__global__ void gemm_tiled_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
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
        if (beta != 0.0f) {
            out += beta * C[row * N + col];
        }
        C[row * N + col] = out;
    }
}

// -------------------------------------------------------------
// Fused Tiled GEMM + Bias + Activation
// -------------------------------------------------------------
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
        if (beta != 0.0f) {
            out += beta * C[row * N + col];
        }
        if (bias != nullptr) {
            out += bias[col];
        }
        C[row * N + col] = apply_activation(out, act);
    }
}

// -------------------------------------------------------------
// FP16 Tiled GEMM
// -------------------------------------------------------------
__global__ void gemm_fp16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    const half* __restrict__ bias,
    half* __restrict__ C,
    int M, int N, int K,
    ActivationType act)
{
    __shared__ half s_A[TILE_DIM][TILE_DIM];
    __shared__ half s_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE_DIM + tx;
        int b_row = t * TILE_DIM + ty;

        s_A[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : __float2half(0.0f);
        s_B[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : __float2half(0.0f);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += __half2float(s_A[ty][k]) * __half2float(s_B[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (bias != nullptr) {
            acc += __half2float(bias[col]);
        }
        float activated = apply_activation(acc, act);
        C[row * N + col] = __float2half(activated);
    }
}

// -------------------------------------------------------------
// 2D Convolution Forward
// -------------------------------------------------------------
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ filters,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_channels, int kernel_h, int kernel_w,
    int out_h, int out_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    ActivationType act)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int b_oc = blockIdx.z;

    int oc = b_oc % out_channels;
    int b = b_oc / out_channels;

    if (ox >= out_w || oy >= out_h || b >= batch_size) {
        return;
    }

    float acc = (bias != nullptr) ? bias[oc] : 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int fy = 0; fy < kernel_h; ++fy) {
            int iy = oy * stride_h - pad_h + fy;
            for (int fx = 0; fx < kernel_w; ++fx) {
                int ix = ox * stride_w - pad_w + fx;

                if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                    int in_idx = ((b * in_channels + ic) * in_h + iy) * in_w + ix;
                    int f_idx = ((oc * in_channels + ic) * kernel_h + fy) * kernel_w + fx;
                    acc += input[in_idx] * filters[f_idx];
                }
            }
        }
    }

    int out_idx = ((b * out_channels + oc) * out_h + oy) * out_w + ox;
    output[out_idx] = apply_activation(acc, act);
}

// -------------------------------------------------------------
// Softmax Forward (Numerically Stable Row-wise)
// -------------------------------------------------------------
__global__ void softmax_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int num_classes)
{
    int b = blockIdx.x;
    if (b >= batch_size) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float shared_buf[];

    const float* row_in = input + b * num_classes;
    float* row_out = output + b * num_classes;

    // 1. Find max value across row
    float local_max = -1e30f;
    for (int i = tid; i < num_classes; i += block_size) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    shared_buf[tid] = local_max;
    __syncthreads();

    // Reduction for max
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] = fmaxf(shared_buf[tid], shared_buf[tid + s]);
        }
        __syncthreads();
    }
    float row_max = shared_buf[0];
    __syncthreads();

    // 2. Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += block_size) {
        float exp_val = __expf(row_in[i] - row_max);
        row_out[i] = exp_val;
        local_sum += exp_val;
    }
    shared_buf[tid] = local_sum;
    __syncthreads();

    // Reduction for sum
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }
    float row_sum = shared_buf[0];
    float inv_sum = 1.0f / (row_sum + 1e-12f);
    __syncthreads();

    // 3. Normalize probabilities
    for (int i = tid; i < num_classes; i += block_size) {
        row_out[i] *= inv_sum;
    }
}

// -------------------------------------------------------------
// Elementwise Activations and Conversions
// -------------------------------------------------------------
__global__ void activation_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int count,
    ActivationType act)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = apply_activation(input[idx], act);
    }
}

__global__ void float_to_half_kernel(
    const float* __restrict__ in,
    half* __restrict__ out,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void half_to_float_kernel(
    const half* __restrict__ in,
    float* __restrict__ out,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = __half2float(in[idx]);
    }
}

// -------------------------------------------------------------
// Host Launch Functions
// -------------------------------------------------------------
void LaunchGemmNaive(
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_KERNEL_CHECK();
}

void LaunchGemmTiledShared(
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_tiled_shared_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_KERNEL_CHECK();
}

void LaunchGemmFused(
    const float* d_A, const float* d_B, const float* d_bias, float* d_C,
    int M, int N, int K,
    ActivationType act,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_fused_bias_activation_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_bias, d_C, M, N, K, act, alpha, beta);
    CUDA_KERNEL_CHECK();
}

void LaunchGemmFp16(
    const half* d_A, const half* d_B, const half* d_bias, half* d_C,
    int M, int N, int K,
    ActivationType act,
    cudaStream_t stream)
{
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_fp16_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_bias, d_C, M, N, K, act);
    CUDA_KERNEL_CHECK();
}

void LaunchConv2D(
    const float* d_input, const float* d_filters, const float* d_bias, float* d_output,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    ActivationType act,
    cudaStream_t stream)
{
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        batch_size * out_channels
    );

    conv2d_forward_kernel<<<grid, block, 0, stream>>>(
        d_input, d_filters, d_bias, d_output,
        batch_size, in_channels, in_h, in_w,
        out_channels, kernel_h, kernel_w,
        out_h, out_w,
        pad_h, pad_w, stride_h, stride_w,
        act
    );
    CUDA_KERNEL_CHECK();
}

void LaunchSoftmax(
    const float* d_input, float* d_output,
    int batch_size, int num_classes,
    cudaStream_t stream)
{
    int block_size = 256;
    if (num_classes < 256) {
        block_size = 64;
        while (block_size < num_classes && block_size < 256) block_size *= 2;
    }
    dim3 grid(batch_size);
    dim3 block(block_size);
    size_t shared_mem = block_size * sizeof(float);

    softmax_forward_kernel<<<grid, block, shared_mem, stream>>>(d_input, d_output, batch_size, num_classes);
    CUDA_KERNEL_CHECK();
}

void LaunchActivation(
    const float* d_input, float* d_output,
    int count,
    ActivationType act,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (count + block - 1) / block;
    activation_kernel<<<grid, block, 0, stream>>>(d_input, d_output, count, act);
    CUDA_KERNEL_CHECK();
}

void LaunchFloatToHalf(const float* d_in, half* d_out, int count, cudaStream_t stream) {
    int block = 256;
    int grid = (count + block - 1) / block;
    float_to_half_kernel<<<grid, block, 0, stream>>>(d_in, d_out, count);
    CUDA_KERNEL_CHECK();
}

void LaunchHalfToFloat(const half* d_in, float* d_out, int count, cudaStream_t stream) {
    int block = 256;
    int grid = (count + block - 1) / block;
    half_to_float_kernel<<<grid, block, 0, stream>>>(d_in, d_out, count);
    CUDA_KERNEL_CHECK();
}

} // namespace Cuda
} // namespace ConvNet
