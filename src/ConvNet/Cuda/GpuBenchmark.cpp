#include "CudaUtils.h"
#include "CudaKernels.h"
#include "CudaMemoryPool.h"
#include "CpuBaseline.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>

namespace ConvNet {
namespace Cuda {

struct GemmBenchResult {
    int M, N, K;
    double cpu_time_ms;
    double gpu_naive_ms;
    double gpu_tiled_ms;
    double gpu_fused_ms;
    double gpu_fp16_ms;
    double warmup_ms;
    double steady_state_ms;
    double gflops_tiled;
    double speedup_tiled_vs_cpu;
    double speedup_tiled_vs_naive;
};

struct ConvBenchResult {
    std::string name;
    int B, Cin, H, W, Cout, K, S, P;
    double cpu_time_ms;
    double gpu_time_ms;
    double gflops;
    double images_per_sec;
    double speedup;
};

struct AllocBenchResult {
    size_t alloc_size_bytes;
    int iterations;
    double raw_cuda_malloc_ms;
    double pool_malloc_ms;
    double speedup;
};

class BenchmarkRunner {
public:
    BenchmarkRunner() : device_(QueryDeviceProperties(0)) {}

    void Run() {
        PrintDeviceHeader();

        RunGemmBenchmarks();
        RunConvBenchmarks();
        RunMemoryPoolBenchmarks();

        WriteMarkdownReport("benchmark_results.md");
        WriteJsonReport("benchmark_results.json");

        std::cout << "\nBenchmark complete. Reports saved to 'benchmark_results.md' and 'benchmark_results.json'.\n";
    }

private:
    DeviceProperties device_;
    std::vector<GemmBenchResult> gemm_results_;
    std::vector<ConvBenchResult> conv_results_;
    std::vector<AllocBenchResult> alloc_results_;

    void PrintDeviceHeader() {
        std::cout << "\n========================================================================================\n"
                  << "                      ConvNetCpp GPU Performance Benchmark Suite                        \n"
                  << "========================================================================================\n"
                  << "Hardware Platform:     " << device_.name << "\n"
                  << "Compute Capability:    SM " << device_.major << "." << device_.minor << "\n"
                  << "Multiprocessors (SMs): " << device_.multi_processor_count << "\n"
                  << "Max Threads / Block:   " << device_.max_threads_per_block << "\n"
                  << "Global Memory:         " << (device_.total_memory / (1024 * 1024)) << " MB\n"
                  << "Theoretical Bandwidth: " << std::fixed << std::setprecision(1) << device_.memory_bandwidth_gbps << " GB/s\n"
                  << "Peak FP32 Compute:     " << std::fixed << std::setprecision(2) << device_.peak_fp32_tflops << " TFLOPS\n"
                  << "========================================================================================\n\n";
    }

    void RunGemmBenchmarks() {
        std::cout << "--- 1. Matrix Multiplication (GEMM) Benchmark (FP32 & FP16) ---\n";
        std::cout << std::left
                  << std::setw(12) << "Size (NxN)"
                  << std::setw(14) << "CPU (ms)"
                  << std::setw(14) << "Naive GPU(ms)"
                  << std::setw(14) << "Tiled GPU(ms)"
                  << std::setw(14) << "Fused (ms)"
                  << std::setw(14) << "FP16 (ms)"
                  << std::setw(12) << "GFLOPS"
                  << std::setw(10) << "Speedup"
                  << "\n";
        std::cout << std::string(104, '-') << "\n";

        std::vector<int> sizes = {128, 256, 512, 1024, 2048};

        for (int sz : sizes) {
            GemmBenchResult res;
            res.M = res.N = res.K = sz;
            size_t elements = static_cast<size_t>(sz) * sz;
            double ops = 2.0 * sz * sz * sz;

            std::vector<float> h_A(elements, 0.01f);
            std::vector<float> h_B(elements, 0.02f);
            std::vector<float> h_bias(sz, 0.05f);
            std::vector<float> h_C(elements, 0.0f);

            // CPU measurement
            if (sz <= 512) {
                auto t0 = std::chrono::high_resolution_clock::now();
                CpuGemm(h_A.data(), h_B.data(), h_C.data(), sz, sz, sz);
                auto t1 = std::chrono::high_resolution_clock::now();
                res.cpu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            } else {
                // Extrapolate O(N^3) CPU time for sizes > 512 to avoid multi-minute waits in benchmark
                double base_cpu = gemm_results_.back().cpu_time_ms;
                res.cpu_time_ms = base_cpu * 8.0; // 2x dimension is 8x work
            }

            // GPU Buffers
            CudaDeviceBuffer<float> d_A(elements);
            CudaDeviceBuffer<float> d_B(elements);
            CudaDeviceBuffer<float> d_bias(sz);
            CudaDeviceBuffer<float> d_C(elements);

            d_A.CopyFromHost(h_A.data(), elements);
            d_B.CopyFromHost(h_B.data(), elements);
            d_bias.CopyFromHost(h_bias.data(), sz);

            CudaEvent start, stop;

            // Warm-up run
            start.Record();
            LaunchGemmTiledShared(d_A.Data(), d_B.Data(), d_C.Data(), sz, sz, sz);
            stop.Record();
            res.warmup_ms = CudaEvent::ElapsedTimeMs(start, stop);

            // Naive GPU (only for sz <= 1024)
            if (sz <= 1024) {
                int iters = (sz <= 512) ? 10 : 3;
                start.Record();
                for (int i = 0; i < iters; ++i) {
                    LaunchGemmNaive(d_A.Data(), d_B.Data(), d_C.Data(), sz, sz, sz);
                }
                stop.Record();
                res.gpu_naive_ms = CudaEvent::ElapsedTimeMs(start, stop) / iters;
            } else {
                res.gpu_naive_ms = -1.0;
            }

            // Tiled Shared Memory GPU
            int iters = (sz <= 512) ? 50 : 20;
            start.Record();
            for (int i = 0; i < iters; ++i) {
                LaunchGemmTiledShared(d_A.Data(), d_B.Data(), d_C.Data(), sz, sz, sz);
            }
            stop.Record();
            res.gpu_tiled_ms = CudaEvent::ElapsedTimeMs(start, stop) / iters;
            res.steady_state_ms = res.gpu_tiled_ms;

            // Fused GEMM (GEMM + Bias + ReLU)
            start.Record();
            for (int i = 0; i < iters; ++i) {
                LaunchGemmFused(d_A.Data(), d_B.Data(), d_bias.Data(), d_C.Data(), sz, sz, sz, ActivationType::ReLU);
            }
            stop.Record();
            res.gpu_fused_ms = CudaEvent::ElapsedTimeMs(start, stop) / iters;

            // FP16 GEMM
            CudaDeviceBuffer<half> d_A_h(elements);
            CudaDeviceBuffer<half> d_B_h(elements);
            CudaDeviceBuffer<half> d_bias_h(sz);
            CudaDeviceBuffer<half> d_C_h(elements);

            LaunchFloatToHalf(d_A.Data(), d_A_h.Data(), elements);
            LaunchFloatToHalf(d_B.Data(), d_B_h.Data(), elements);
            LaunchFloatToHalf(d_bias.Data(), d_bias_h.Data(), sz);

            start.Record();
            for (int i = 0; i < iters; ++i) {
                LaunchGemmFp16(d_A_h.Data(), d_B_h.Data(), d_bias_h.Data(), d_C_h.Data(), sz, sz, sz, ActivationType::None);
            }
            stop.Record();
            res.gpu_fp16_ms = CudaEvent::ElapsedTimeMs(start, stop) / iters;

            // Metrics
            res.gflops_tiled = (ops / (res.gpu_tiled_ms * 1e-3)) / 1e9;
            res.speedup_tiled_vs_cpu = res.cpu_time_ms / res.gpu_tiled_ms;
            res.speedup_tiled_vs_naive = (res.gpu_naive_ms > 0) ? (res.gpu_naive_ms / res.gpu_tiled_ms) : 0.0;

            gemm_results_.push_back(res);

            std::cout << std::left
                      << std::setw(12) << (std::to_string(sz) + "x" + std::to_string(sz))
                      << std::fixed << std::setprecision(2) << std::setw(14) << res.cpu_time_ms
                      << std::setw(14) << (res.gpu_naive_ms > 0 ? std::to_string(res.gpu_naive_ms).substr(0, 5) : "N/A")
                      << std::setw(14) << res.gpu_tiled_ms
                      << std::setw(14) << res.gpu_fused_ms
                      << std::setw(14) << res.gpu_fp16_ms
                      << std::setw(12) << res.gflops_tiled
                      << "\033[32m" << std::setw(10) << (std::to_string(res.speedup_tiled_vs_cpu).substr(0, 4) + "x") << "\033[0m"
                      << "\n";
        }
        std::cout << "\n";
    }

    void RunConvBenchmarks() {
        std::cout << "--- 2. 2D Convolution (Conv2D) Benchmark ---\n";
        std::cout << std::left
                  << std::setw(30) << "Layer Configuration"
                  << std::setw(14) << "CPU (ms)"
                  << std::setw(14) << "GPU (ms)"
                  << std::setw(14) << "GFLOPS"
                  << std::setw(18) << "Images / sec"
                  << std::setw(10) << "Speedup"
                  << "\n";
        std::cout << std::string(100, '-') << "\n";

        struct ConvConfig {
            std::string name;
            int B, Cin, H, W, Cout, K, S, P;
        };

        std::vector<ConvConfig> configs = {
            {"CIFAR-10 Conv1 (3->32, 32x32)", 32, 3, 32, 32, 32, 3, 1, 1},
            {"Mid-layer Conv2 (32->64, 32x32)", 16, 32, 32, 32, 64, 3, 1, 1},
            {"Deep Feature Conv3 (64->128, 16x16)", 8, 64, 16, 16, 128, 3, 1, 1}
        };

        for (const auto& cfg : configs) {
            ConvBenchResult res;
            res.name = cfg.name;
            res.B = cfg.B; res.Cin = cfg.Cin; res.H = cfg.H; res.W = cfg.W;
            res.Cout = cfg.Cout; res.K = cfg.K; res.S = cfg.S; res.P = cfg.P;

            int out_h = (cfg.H + 2 * cfg.P - cfg.K) / cfg.S + 1;
            int out_w = (cfg.W + 2 * cfg.P - cfg.K) / cfg.S + 1;

            size_t in_size = cfg.B * cfg.Cin * cfg.H * cfg.W;
            size_t f_size = cfg.Cout * cfg.Cin * cfg.K * cfg.K;
            size_t out_size = cfg.B * cfg.Cout * out_h * out_w;

            double ops = 2.0 * cfg.B * cfg.Cout * out_h * out_w * (cfg.Cin * cfg.K * cfg.K);

            std::vector<float> h_in(in_size, 0.05f);
            std::vector<float> h_f(f_size, 0.01f);
            std::vector<float> h_b(cfg.Cout, 0.0f);
            std::vector<float> h_out_cpu(out_size, 0.0f);

            // CPU sample (batch 1 extrapolated to B)
            auto t0 = std::chrono::high_resolution_clock::now();
            CpuConv2D(h_in.data(), h_f.data(), h_b.data(), h_out_cpu.data(),
                      1, cfg.Cin, cfg.H, cfg.W, cfg.Cout, cfg.K, cfg.K, cfg.P, cfg.P, cfg.S, cfg.S);
            auto t1 = std::chrono::high_resolution_clock::now();
            double b1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            res.cpu_time_ms = b1_ms * cfg.B;

            // GPU
            CudaDeviceBuffer<float> d_in(in_size);
            CudaDeviceBuffer<float> d_f(f_size);
            CudaDeviceBuffer<float> d_b(cfg.Cout);
            CudaDeviceBuffer<float> d_out(out_size);

            d_in.CopyFromHost(h_in.data(), in_size);
            d_f.CopyFromHost(h_f.data(), f_size);
            d_b.CopyFromHost(h_b.data(), cfg.Cout);

            CudaEvent start, stop;
            // Warm-up
            LaunchConv2D(d_in.Data(), d_f.Data(), d_b.Data(), d_out.Data(),
                         cfg.B, cfg.Cin, cfg.H, cfg.W, cfg.Cout, cfg.K, cfg.K,
                         cfg.P, cfg.P, cfg.S, cfg.S, ActivationType::ReLU);

            int iters = 20;
            start.Record();
            for (int i = 0; i < iters; ++i) {
                LaunchConv2D(d_in.Data(), d_f.Data(), d_b.Data(), d_out.Data(),
                             cfg.B, cfg.Cin, cfg.H, cfg.W, cfg.Cout, cfg.K, cfg.K,
                             cfg.P, cfg.P, cfg.S, cfg.S, ActivationType::ReLU);
            }
            stop.Record();
            res.gpu_time_ms = CudaEvent::ElapsedTimeMs(start, stop) / iters;

            res.gflops = (ops / (res.gpu_time_ms * 1e-3)) / 1e9;
            res.images_per_sec = (cfg.B / (res.gpu_time_ms * 1e-3));
            res.speedup = res.cpu_time_ms / res.gpu_time_ms;

            conv_results_.push_back(res);

            std::cout << std::left
                      << std::setw(30) << res.name
                      << std::fixed << std::setprecision(2) << std::setw(14) << res.cpu_time_ms
                      << std::setw(14) << res.gpu_time_ms
                      << std::setw(14) << res.gflops
                      << std::setw(18) << res.images_per_sec
                      << "\033[32m" << std::setw(10) << (std::to_string(res.speedup).substr(0, 4) + "x") << "\033[0m"
                      << "\n";
        }
        std::cout << "\n";
    }

    void RunMemoryPoolBenchmarks() {
        std::cout << "--- 3. MemoryPool vs Raw Driver Allocation Benchmark ---\n";
        std::cout << std::left
                  << std::setw(16) << "Block Size"
                  << std::setw(16) << "Iterations"
                  << std::setw(22) << "Raw cudaMalloc (ms)"
                  << std::setw(22) << "CudaMemoryPool (ms)"
                  << std::setw(10) << "Speedup"
                  << "\n";
        std::cout << std::string(86, '-') << "\n";

        std::vector<size_t> sizes = {64 * 1024, 1024 * 1024, 16 * 1024 * 1024};
        int iters = 500;

        for (size_t sz : sizes) {
            AllocBenchResult res;
            res.alloc_size_bytes = sz;
            res.iterations = iters;

            // 1. Raw cudaMalloc / cudaFree
            auto t0_raw = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) {
                void* ptr = nullptr;
                cudaMalloc(&ptr, sz);
                cudaFree(ptr);
            }
            auto t1_raw = std::chrono::high_resolution_clock::now();
            res.raw_cuda_malloc_ms = std::chrono::duration<double, std::milli>(t1_raw - t0_raw).count();

            // 2. CudaMemoryPool Allocate / Deallocate
            auto& pool = CudaMemoryPool::Instance();
            pool.Clear();

            auto t0_pool = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) {
                void* ptr = pool.Allocate(sz);
                pool.Deallocate(ptr);
            }
            auto t1_pool = std::chrono::high_resolution_clock::now();
            res.pool_malloc_ms = std::chrono::duration<double, std::milli>(t1_pool - t0_pool).count();

            res.speedup = res.raw_cuda_malloc_ms / std::max(0.001, res.pool_malloc_ms);
            alloc_results_.push_back(res);

            std::cout << std::left
                      << std::setw(16) << (std::to_string(sz / 1024) + " KB")
                      << std::setw(16) << iters
                      << std::fixed << std::setprecision(2) << std::setw(22) << res.raw_cuda_malloc_ms
                      << std::setw(22) << res.pool_malloc_ms
                      << "\033[32m" << std::setw(10) << (std::to_string(res.speedup).substr(0, 4) + "x") << "\033[0m"
                      << "\n";
        }
        std::cout << "\n";
    }

    void WriteMarkdownReport(const std::string& path) {
        std::ofstream ofs(path);
        ofs << "# ConvNetCpp GPU Performance Benchmark Report\n\n";
        ofs << "**Hardware:** " << device_.name << " (Compute " << device_.major << "." << device_.minor << ", "
            << device_.multi_processor_count << " SMs, " << std::fixed << std::setprecision(1) << device_.memory_bandwidth_gbps << " GB/s Bandwidth, "
            << std::fixed << std::setprecision(2) << device_.peak_fp32_tflops << " TFLOPS FP32)\n\n";

        ofs << "## 1. Matrix Multiplication (GEMM)\n\n";
        ofs << "| Matrix Size | CPU Baseline (ms) | Naive GPU (ms) | Tiled GPU (ms) | Fused GPU (ms) | FP16 GPU (ms) | Tiled GFLOPS | Speedup vs CPU |\n";
        ofs << "|:---|:---|:---|:---|:---|:---|:---|:---|\n";
        for (const auto& r : gemm_results_) {
            ofs << "| " << r.M << "x" << r.N << " | "
                << std::fixed << std::setprecision(2) << r.cpu_time_ms << " | "
                << (r.gpu_naive_ms > 0 ? std::to_string(r.gpu_naive_ms).substr(0, 5) : "N/A") << " | "
                << r.gpu_tiled_ms << " | "
                << r.gpu_fused_ms << " | "
                << r.gpu_fp16_ms << " | "
                << std::setprecision(1) << r.gflops_tiled << " | **"
                << std::setprecision(1) << r.speedup_tiled_vs_cpu << "x** |\n";
        }

        ofs << "\n## 2. 2D Convolution (Conv2D)\n\n";
        ofs << "| Layer Name | CPU Time (ms) | GPU Time (ms) | Throughput (GFLOPS) | Throughput (images/s) | Speedup |\n";
        ofs << "|:---|:---|:---|:---|:---|:---|\n";
        for (const auto& r : conv_results_) {
            ofs << "| " << r.name << " | "
                << std::fixed << std::setprecision(2) << r.cpu_time_ms << " | "
                << r.gpu_time_ms << " | "
                << std::setprecision(1) << r.gflops << " | "
                << std::setprecision(1) << r.images_per_sec << " | **"
                << std::setprecision(1) << r.speedup << "x** |\n";
        }

        ofs << "\n## 3. GPU Memory Pooling Efficiency\n\n";
        ofs << "| Block Size | Iterations | Raw `cudaMalloc` (ms) | `CudaMemoryPool` (ms) | Allocation Speedup |\n";
        ofs << "|:---|:---|:---|:---|:---|\n";
        for (const auto& r : alloc_results_) {
            ofs << "| " << (r.alloc_size_bytes / 1024) << " KB | " << r.iterations << " | "
                << std::fixed << std::setprecision(2) << r.raw_cuda_malloc_ms << " | "
                << r.pool_malloc_ms << " | **"
                << std::setprecision(1) << r.speedup << "x** |\n";
        }
    }

    void WriteJsonReport(const std::string& path) {
        std::ofstream ofs(path);
        ofs << "{\n";
        ofs << "  \"device\": {\n"
            << "    \"name\": \"" << device_.name << "\",\n"
            << "    \"compute\": \"" << device_.major << "." << device_.minor << "\",\n"
            << "    \"peak_fp32_tflops\": " << device_.peak_fp32_tflops << ",\n"
            << "    \"bandwidth_gbps\": " << device_.memory_bandwidth_gbps << "\n"
            << "  },\n";

        ofs << "  \"gemm_benchmarks\": [\n";
        for (size_t i = 0; i < gemm_results_.size(); ++i) {
            const auto& r = gemm_results_[i];
            ofs << "    {\n"
                << "      \"size\": " << r.M << ",\n"
                << "      \"cpu_ms\": " << r.cpu_time_ms << ",\n"
                << "      \"gpu_naive_ms\": " << r.gpu_naive_ms << ",\n"
                << "      \"gpu_tiled_ms\": " << r.gpu_tiled_ms << ",\n"
                << "      \"gpu_fused_ms\": " << r.gpu_fused_ms << ",\n"
                << "      \"gpu_fp16_ms\": " << r.gpu_fp16_ms << ",\n"
                << "      \"gflops\": " << r.gflops_tiled << ",\n"
                << "      \"speedup_vs_cpu\": " << r.speedup_tiled_vs_cpu << "\n"
                << "    }" << (i + 1 < gemm_results_.size() ? "," : "") << "\n";
        }
        ofs << "  ],\n";

        ofs << "  \"conv_benchmarks\": [\n";
        for (size_t i = 0; i < conv_results_.size(); ++i) {
            const auto& r = conv_results_[i];
            ofs << "    {\n"
                << "      \"name\": \"" << r.name << "\",\n"
                << "      \"cpu_ms\": " << r.cpu_time_ms << ",\n"
                << "      \"gpu_ms\": " << r.gpu_time_ms << ",\n"
                << "      \"gflops\": " << r.gflops << ",\n"
                << "      \"images_per_sec\": " << r.images_per_sec << ",\n"
                << "      \"speedup\": " << r.speedup << "\n"
                << "    }" << (i + 1 < conv_results_.size() ? "," : "") << "\n";
        }
        ofs << "  ],\n";

        ofs << "  \"memory_benchmarks\": [\n";
        for (size_t i = 0; i < alloc_results_.size(); ++i) {
            const auto& r = alloc_results_[i];
            ofs << "    {\n"
                << "      \"size_bytes\": " << r.alloc_size_bytes << ",\n"
                << "      \"raw_cuda_malloc_ms\": " << r.raw_cuda_malloc_ms << ",\n"
                << "      \"pool_malloc_ms\": " << r.pool_malloc_ms << ",\n"
                << "      \"speedup\": " << r.speedup << "\n"
                << "    }" << (i + 1 < alloc_results_.size() ? "," : "") << "\n";
        }
        ofs << "  ]\n";
        ofs << "}\n";
    }
};

} // namespace Cuda
} // namespace ConvNet

int main() {
    try {
        ConvNet::Cuda::BenchmarkRunner runner;
        runner.Run();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Benchmark failed: " << ex.what() << "\n";
        return 1;
    }
}
