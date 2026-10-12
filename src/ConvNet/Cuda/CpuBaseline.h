#ifndef CONVNET_CPU_BASELINE_H
#define CONVNET_CPU_BASELINE_H

#include "CudaUtils.h"
#include <vector>

namespace ConvNet {
namespace Cuda {

// Golden CPU Reference implementations for numerical correctness verification

void CpuGemm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f
);

void CpuGemmFused(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K,
    ActivationType act = ActivationType::ReLU,
    float alpha = 1.0f, float beta = 0.0f
);

void CpuConv2D(
    const float* input, const float* filters, const float* bias, float* output,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    ActivationType act = ActivationType::ReLU
);

void CpuSoftmax(
    const float* input, float* output,
    int batch_size, int num_classes
);

void CpuActivation(
    const float* input, float* output,
    int count,
    ActivationType act
);

float CpuApplyActivation(float x, ActivationType act);

} // namespace Cuda
} // namespace ConvNet

#endif // CONVNET_CPU_BASELINE_H
