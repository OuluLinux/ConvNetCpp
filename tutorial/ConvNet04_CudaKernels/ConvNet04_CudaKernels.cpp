#include <ConvNet/Cuda/CudaUtils.h>
#include <ConvNet/Cuda/CudaKernels.h>
#include <ConvNet/Cuda/CpuBaseline.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>

/*
================================================================================
ConvNetCpp Tutorial 04: Custom CUDA Kernels & Epilogue Fusion
================================================================================

This tutorial demonstrates:
1. Difference between Naive DRAM GEMM and Tiled Shared-Memory GEMM
2. The concept of 2D on-chip SRAM tiling (reducing global memory traffic 16x)
3. Fusing Bias addition and activation functions into register accumulators
4. Numerical parity validation against Golden CPU baseline
*/

int main() {
    using namespace ConvNet::Cuda;

    std::cout << "========================================================================\n"
              << "       ConvNetCpp Tutorial 04: Custom CUDA Kernels & Epilogue Fusion    \n"
              << "========================================================================\n\n";

    const int M = 512, N = 512, K = 512;
    std::cout << "[Step 1] Configuring Matrix Problem: C = ReLU(A * B + bias)\n"
              << "  Shape: [" << M << "x" << K << "] * [" << K << "x" << N << "] + [" << N << "]\n\n";

    // Host initialization
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_bias(N);
    std::vector<float> h_C_cpu(M * N, 0.0f);
    std::vector<float> h_C_naive(M * N, 0.0f);
    std::vector<float> h_C_tiled(M * N, 0.0f);
    std::vector<float> h_C_fused(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = ((i % 17) - 8) * 0.05f;
    for (int i = 0; i < K * N; ++i) h_B[i] = ((i % 13) - 6) * 0.05f;
    for (int i = 0; i < N; ++i) h_bias[i] = 0.1f;

    // Golden CPU Reference
    std::cout << "[Step 2] Executing Golden CPU Reference...\n";
    auto t0_cpu = std::chrono::high_resolution_clock::now();
    CpuGemmFused(h_A.data(), h_B.data(), h_bias.data(), h_C_cpu.data(), M, N, K, ActivationType::ReLU);
    auto t1_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1_cpu - t0_cpu).count();
    std::cout << "  CPU Execution Time: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n\n";

    // Allocate GPU Device Buffers
    CudaDeviceBuffer<float> d_A(M * K);
    CudaDeviceBuffer<float> d_B(K * N);
    CudaDeviceBuffer<float> d_bias(N);
    CudaDeviceBuffer<float> d_C(M * N);

    d_A.CopyFromHost(h_A.data(), M * K);
    d_B.CopyFromHost(h_B.data(), K * N);
    d_bias.CopyFromHost(h_bias.data(), N);

    CudaEvent ev_start, ev_stop;

    // 1. Naive GEMM
    std::cout << "[Step 3] Executing Naive GPU Kernel (Global Memory Reads)...\n";
    ev_start.Record();
    LaunchGemmNaive(d_A.Data(), d_B.Data(), d_C.Data(), M, N, K);
    ev_stop.Record();
    float naive_ms = CudaEvent::ElapsedTimeMs(ev_start, ev_stop);
    d_C.CopyToHost(h_C_naive.data(), M * N);
    std::cout << "  Naive GPU Latency:  " << naive_ms << " ms ("
              << (cpu_ms / naive_ms) << "x vs CPU)\n";

    // 2. Tiled Shared-Memory GEMM
    std::cout << "[Step 4] Executing Tiled Shared-Memory GEMM (16x16 SRAM Tiles)...\n";
    ev_start.Record();
    LaunchGemmTiledShared(d_A.Data(), d_B.Data(), d_C.Data(), M, N, K);
    ev_stop.Record();
    float tiled_ms = CudaEvent::ElapsedTimeMs(ev_start, ev_stop);
    d_C.CopyToHost(h_C_tiled.data(), M * N);
    std::cout << "  Tiled GPU Latency:  " << tiled_ms << " ms ("
              << (cpu_ms / tiled_ms) << "x vs CPU, " << (naive_ms / tiled_ms) << "x vs Naive GPU)\n";

    // 3. Fused GEMM + Bias + ReLU
    std::cout << "[Step 5] Executing Fused GEMM + Bias + ReLU Epilogue...\n";
    ev_start.Record();
    LaunchGemmFused(d_A.Data(), d_B.Data(), d_bias.Data(), d_C.Data(), M, N, K, ActivationType::ReLU);
    ev_stop.Record();
    float fused_ms = CudaEvent::ElapsedTimeMs(ev_start, ev_stop);
    d_C.CopyToHost(h_C_fused.data(), M * N);
    std::cout << "  Fused GPU Latency:  " << fused_ms << " ms ("
              << (cpu_ms / fused_ms) << "x vs CPU)\n\n";

    // Correctness Verification against Golden Reference
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(h_C_cpu[i] - h_C_fused[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "[Step 6] Numerical Parity Verification:\n"
              << "  Max Absolute Difference vs CPU Reference: " << std::scientific << max_diff << "\n";
    if (max_diff < 1e-4f) {
        std::cout << "  \033[32m[PASS] Output matches golden CPU reference within tolerance!\033[0m\n\n";
    } else {
        std::cout << "  \033[31m[FAIL] Parity check failed!\033[0m\n\n";
    }

    std::cout << "[TUTORIAL 04 COMPLETE]\n";
    return 0;
}
