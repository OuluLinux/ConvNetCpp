#include "CpuBaseline.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace ConvNet {
namespace Cuda {

float CpuApplyActivation(float x, ActivationType act) {
    switch (act) {
        case ActivationType::None:
            return x;
        case ActivationType::ReLU:
            return std::max(0.0f, x);
        case ActivationType::LeakyReLU:
            return x > 0.0f ? x : 0.01f * x;
        case ActivationType::Sigmoid:
            return 1.0f / (1.0f + std::exp(-x));
        case ActivationType::Tanh:
            return std::tanh(x);
        case ActivationType::GELU:
            return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        default:
            return x;
    }
}

void CpuGemm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            if (beta == 0.0f) {
                C[i * N + j] = alpha * sum;
            } else {
                C[i * N + j] = alpha * sum + beta * C[i * N + j];
            }
        }
    }
}

void CpuGemmFused(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K,
    ActivationType act,
    float alpha, float beta)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            float out = alpha * sum;
            if (beta != 0.0f) {
                out += beta * C[i * N + j];
            }
            if (bias != nullptr) {
                out += bias[j];
            }
            C[i * N + j] = CpuApplyActivation(out, act);
        }
    }
}

void CpuConv2D(
    const float* input, const float* filters, const float* bias, float* output,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    ActivationType act)
{
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            float b_val = (bias != nullptr) ? bias[oc] : 0.0f;

            for (int oy = 0; oy < out_h; ++oy) {
                for (int ox = 0; ox < out_w; ++ox) {
                    float sum = b_val;

                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int fy = 0; fy < kernel_h; ++fy) {
                            int iy = oy * stride_h - pad_h + fy;
                            for (int fx = 0; fx < kernel_w; ++fx) {
                                int ix = ox * stride_w - pad_w + fx;

                                if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                                    int in_idx = ((b * in_channels + ic) * in_h + iy) * in_w + ix;
                                    int f_idx = ((oc * in_channels + ic) * kernel_h + fy) * kernel_w + fx;
                                    sum += input[in_idx] * filters[f_idx];
                                }
                            }
                        }
                    }

                    int out_idx = ((b * out_channels + oc) * out_h + oy) * out_w + ox;
                    output[out_idx] = CpuApplyActivation(sum, act);
                }
            }
        }
    }
}

void CpuSoftmax(
    const float* input, float* output,
    int batch_size, int num_classes)
{
    for (int b = 0; b < batch_size; ++b) {
        const float* row_in = input + b * num_classes;
        float* row_out = output + b * num_classes;

        // 1. Max
        float max_val = row_in[0];
        for (int c = 1; c < num_classes; ++c) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }

        // 2. Exp and sum
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float e = std::exp(row_in[c] - max_val);
            row_out[c] = e;
            sum_exp += e;
        }

        // 3. Normalize
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        for (int c = 0; c < num_classes; ++c) {
            row_out[c] *= inv_sum;
        }
    }
}

void CpuActivation(
    const float* input, float* output,
    int count,
    ActivationType act)
{
    for (int i = 0; i < count; ++i) {
        output[i] = CpuApplyActivation(input[i], act);
    }
}

} // namespace Cuda
} // namespace ConvNet
