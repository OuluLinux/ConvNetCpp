#include <ConvNet/Cuda/CudaUtils.h>
#include <ConvNet/Cuda/CudaMemoryPool.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

/*
================================================================================
ConvNetCpp Tutorial 03: CUDA Hardware Discovery, Memory Management & Streams
================================================================================

This tutorial demonstrates fundamental CUDA systems engineering in ConvNetCpp:
1. Querying GPU architectural parameters (SM count, clock, peak TFLOPS, bandwidth)
2. RAII memory management using CudaDeviceBuffer and CudaPinnedBuffer
3. Measuring PCIe DMA throughput (Pageable vs Pinned Host Memory)
4. Asynchronous operations with CudaStream and precise timing with CudaEvent
5. Eliminating driver synchronization stalls with CudaMemoryPool
*/

int main() {
    using namespace ConvNet::Cuda;

    std::cout << "========================================================================\n"
              << "       ConvNetCpp Tutorial 03: CUDA Architecture & Memory Management    \n"
              << "========================================================================\n\n";

    // 1. Device Discovery
    DeviceProperties dev = QueryDeviceProperties(0);
    std::cout << "[Step 1] Detected GPU Hardware:\n"
              << "  Device Name:         " << dev.name << "\n"
              << "  Compute Capability:  SM " << dev.major << "." << dev.minor << "\n"
              << "  Multiprocessors:     " << dev.multi_processor_count << " SMs\n"
              << "  Max Threads / Block: " << dev.max_threads_per_block << "\n"
              << "  Global Memory:       " << (dev.total_memory / (1024 * 1024)) << " MB\n"
              << "  Peak Bandwidth:      " << std::fixed << std::setprecision(1) << dev.memory_bandwidth_gbps << " GB/s\n"
              << "  Peak FP32 Compute:   " << std::fixed << std::setprecision(2) << dev.peak_fp32_tflops << " TFLOPS\n\n";

    // 2. Memory Transfer Benchmark: Pinned vs Pageable Host Memory
    const size_t num_elements = 16 * 1024 * 1024; // 16M floats = 64 MB
    const size_t num_bytes = num_elements * sizeof(float);
    std::cout << "[Step 2] Measuring PCIe DMA Transfer Rates (Payload: 64 MB)...\n";

    CudaDeviceBuffer<float> d_buf(num_elements);

    // Pageable host memory
    std::vector<float> h_pageable(num_elements, 1.23f);
    CudaEvent ev_start, ev_stop;

    ev_start.Record();
    d_buf.CopyFromHost(h_pageable.data(), num_elements);
    ev_stop.Record();
    float pageable_ms = CudaEvent::ElapsedTimeMs(ev_start, ev_stop);
    double pageable_gbps = (num_bytes / (pageable_ms * 1e-3)) / 1e9;

    std::cout << "  Pageable Host Memory: " << std::fixed << std::setprecision(2)
              << pageable_ms << " ms  (" << pageable_gbps << " GB/s)\n";

    // Pinned host memory (cudaHostAlloc)
    CudaPinnedBuffer<float> h_pinned(num_elements);
    for (size_t i = 0; i < num_elements; ++i) h_pinned[i] = 1.23f;

    CudaStream stream;
    ev_start.Record(stream);
    d_buf.CopyFromHost(h_pinned.Data(), num_elements, stream);
    ev_stop.Record(stream);
    stream.Synchronize();

    float pinned_ms = CudaEvent::ElapsedTimeMs(ev_start, ev_stop);
    double pinned_gbps = (num_bytes / (pinned_ms * 1e-3)) / 1e9;

    std::cout << "  Pinned Host Memory:   " << std::fixed << std::setprecision(2)
              << pinned_ms << " ms  (" << pinned_gbps << " GB/s)\n"
              << "  Speedup from Pinning: " << (pinned_gbps / pageable_gbps) << "x\n\n";

    // 3. Memory Pooling vs Driver Allocation
    std::cout << "[Step 3] Memory Allocation Latency (500 iterations of 4 MB buffers):\n";
    size_t alloc_size = 4 * 1024 * 1024;
    int iters = 500;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        void* ptr = nullptr;
        cudaMalloc(&ptr, alloc_size);
        cudaFree(ptr);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double raw_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto& pool = CudaMemoryPool::Instance();
    pool.Clear();

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        void* ptr = pool.Allocate(alloc_size);
        pool.Deallocate(ptr);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double pool_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "  Raw cudaMalloc/Free:  " << raw_ms << " ms\n"
              << "  CudaMemoryPool:       " << pool_ms << " ms\n"
              << "  Allocation Speedup:   " << (raw_ms / std::max(0.001, pool_ms)) << "x\n\n";

    std::cout << pool.DumpStatistics() << "\n";
    std::cout << "[TUTORIAL 03 COMPLETE]\n";
    return 0;
}
