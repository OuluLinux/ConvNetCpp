#ifndef CONVNET_CUDA_KERNELS_H
#define CONVNET_CUDA_KERNELS_H

#include "CudaUtils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace ConvNet {
namespace Cuda {

// Matrix Multiplication (GEMM): C = alpha * (A * B) + beta * C
// A is M x K (row-major), B is K x N (row-major), C is M x N (row-major)
void LaunchGemmNaive(
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f,
    cudaStream_t stream = 0
);

void LaunchGemmTiledShared(
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f,
    cudaStream_t stream = 0
);

// Fused GEMM + Bias + Activation:
// C = Activation(alpha * (A * B) + beta * C + bias)
void LaunchGemmFused(
    const float* d_A, const float* d_B, const float* d_bias, float* d_C,
    int M, int N, int K,
    ActivationType act = ActivationType::ReLU,
    float alpha = 1.0f, float beta = 0.0f,
    cudaStream_t stream = 0
);

// Half-precision FP16 GEMM
void LaunchGemmFp16(
    const half* d_A, const half* d_B, const half* d_bias, half* d_C,
    int M, int N, int K,
    ActivationType act = ActivationType::None,
    cudaStream_t stream = 0
);

// 2D Spatial Convolution:
// Input: [B, C_in, H_in, W_in]
// Filters: [C_out, C_in, K_h, K_w]
// Bias: [C_out] (optional)
// Output: [B, C_out, H_out, W_out]
void LaunchConv2D(
    const float* d_input, const float* d_filters, const float* d_bias, float* d_output,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    ActivationType act = ActivationType::ReLU,
    cudaStream_t stream = 0
);

// Numerically stable row-wise Softmax:
// Input/Output: [batch_size, num_classes]
void LaunchSoftmax(
    const float* d_input, float* d_output,
    int batch_size, int num_classes,
    cudaStream_t stream = 0
);

// Elementwise Activation
void LaunchActivation(
    const float* d_input, float* d_output,
    int count,
    ActivationType act,
    cudaStream_t stream = 0
);

// Precision conversion helpers
void LaunchFloatToHalf(const float* d_in, half* d_out, int count, cudaStream_t stream = 0);
void LaunchHalfToFloat(const half* d_in, float* d_out, int count, cudaStream_t stream = 0);

} // namespace Cuda
} // namespace ConvNet

#endif // CONVNET_CUDA_KERNELS_H
