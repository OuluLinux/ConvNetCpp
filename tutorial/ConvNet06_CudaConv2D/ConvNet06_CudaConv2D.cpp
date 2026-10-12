#include <ConvNet/Cuda/CudaUtils.h>
#include <ConvNet/Cuda/CudaKernels.h>
#include <ConvNet/Cuda/CpuBaseline.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>

/*
================================================================================
ConvNetCpp Tutorial 06: 2D Spatial Convolution & Image Processing on GPU
================================================================================

This tutorial demonstrates:
1. Setting up multi-channel 4D image tensors [Batch, Channel, Height, Width]
2. Handcrafted 2D spatial convolution with custom strides and zero-padding
3. Fused bias addition and ReLU activation in convolutional layers
4. High-throughput image processing (images/sec) compared to CPU baseline
5. Mathematical parity check against golden CPU convolution reference
*/

int main() {
    using namespace ConvNet::Cuda;

    std::cout << "========================================================================\n"
              << "       ConvNetCpp Tutorial 06: 2D Spatial Convolution on GPU            \n"
              << "========================================================================\n\n";

    // 1. Layer Hyperparameters
    const int batch_size = 16;
    const int in_channels = 3;   // RGB
    const int in_h = 32, in_w = 32;
    const int out_channels = 32; // 32 feature maps
    const int kernel_h = 3, kernel_w = 3;
    const int pad_h = 1, pad_w = 1;
    const int stride_h = 1, stride_w = 1;

    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    std::cout << "[Step 1] Convolution Layer Topology:\n"
              << "  Input Shape:   [" << batch_size << ", " << in_channels << ", " << in_h << ", " << in_w << "]\n"
              << "  Filter Shape:  [" << out_channels << ", " << in_channels << ", " << kernel_h << ", " << kernel_w << "]\n"
              << "  Padding:       [" << pad_h << ", " << pad_w << "], Stride: [" << stride_h << ", " << stride_w << "]\n"
              << "  Output Shape:  [" << batch_size << ", " << out_channels << ", " << out_h << ", " << out_w << "]\n\n";

    size_t in_elements = batch_size * in_channels * in_h * in_w;
    size_t f_elements = out_channels * in_channels * kernel_h * kernel_w;
    size_t out_elements = batch_size * out_channels * out_h * out_w;
    double total_flops = 2.0 * batch_size * out_channels * out_h * out_w * (in_channels * kernel_h * kernel_w);

    // Host buffers
    std::vector<float> h_input(in_elements);
    std::vector<float> h_filters(f_elements);
    std::vector<float> h_bias(out_channels, 0.05f);
    std::vector<float> h_out_cpu(out_elements, 0.0f);
    std::vector<float> h_out_gpu(out_elements, 0.0f);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < in_elements; ++i) h_input[i] = dist(rng);
    for (size_t i = 0; i < f_elements; ++i) h_filters[i] = dist(rng);

    // 2. CPU Convolution
    std::cout << "[Step 2] Executing Golden CPU 2D Convolution Reference...\n";
    auto t0_cpu = std::chrono::high_resolution_clock::now();
    CpuConv2D(h_input.data(), h_filters.data(), h_bias.data(), h_out_cpu.data(),
              batch_size, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w,
              pad_h, pad_w, stride_h, stride_w, ActivationType::ReLU);
    auto t1_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1_cpu - t0_cpu).count();
    std::cout << "  CPU Latency: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n\n";

    // 3. GPU Convolution
    std::cout << "[Step 3] Executing Custom CUDA Conv2D Kernel...\n";
    CudaDeviceBuffer<float> d_input(in_elements);
    CudaDeviceBuffer<float> d_filters(f_elements);
    CudaDeviceBuffer<float> d_bias(out_channels);
    CudaDeviceBuffer<float> d_output(out_elements);

    d_input.CopyFromHost(h_input.data(), in_elements);
    d_filters.CopyFromHost(h_filters.data(), f_elements);
    d_bias.CopyFromHost(h_bias.data(), out_channels);

    // Warm-up
    LaunchConv2D(d_input.Data(), d_filters.Data(), d_bias.Data(), d_output.Data(),
                 batch_size, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w,
                 pad_h, pad_w, stride_h, stride_w, ActivationType::ReLU);

    CudaEvent ev_start, ev_stop;
    int iters = 50;
    ev_start.Record();
    for (int i = 0; i < iters; ++i) {
        LaunchConv2D(d_input.Data(), d_filters.Data(), d_bias.Data(), d_output.Data(),
                     batch_size, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w,
                     pad_h, pad_w, stride_h, stride_w, ActivationType::ReLU);
    }
    ev_stop.Record();
    float gpu_ms = CudaEvent::ElapsedTimeMs(ev_start, ev_stop) / iters;
    d_output.CopyToHost(h_out_gpu.data(), out_elements);

    double gflops = (total_flops / (gpu_ms * 1e-3)) / 1e9;
    double images_per_sec = batch_size / (gpu_ms * 1e-3);
    double speedup = cpu_ms / gpu_ms;

    std::cout << "  GPU Latency:   " << std::fixed << std::setprecision(3) << gpu_ms << " ms\n"
              << "  Throughput:    " << std::setprecision(1) << images_per_sec << " images/sec (" << gflops << " GFLOPS)\n"
              << "  Speedup vs CPU:" << std::setprecision(1) << speedup << "x\n\n";

    // 4. Parity Check
    float max_diff = 0.0f;
    for (size_t i = 0; i < out_elements; ++i) {
        float diff = std::abs(h_out_cpu[i] - h_out_gpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    std::cout << "[Step 4] Verifying Numerical Parity against Golden Reference:\n"
              << "  Max Absolute Difference: " << std::scientific << max_diff << "\n";
    if (max_diff < 1e-4f) {
        std::cout << "  \033[32m[PASS] GPU Convolution matches CPU reference with mathematical parity!\033[0m\n\n";
    } else {
        std::cout << "  \033[31m[FAIL] Parity check breached tolerance threshold!\033[0m\n\n";
    }

    std::cout << "[TUTORIAL 06 COMPLETE]\n";
    return 0;
}
