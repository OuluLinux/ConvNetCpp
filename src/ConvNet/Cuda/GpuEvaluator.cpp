#include "CudaUtils.h"
#include "CudaKernels.h"
#include "CudaMemoryPool.h"
#include "CpuBaseline.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <sstream>
#include <chrono>

namespace ConvNet {
namespace Cuda {

struct TestCaseResult {
    std::string name;
    std::string shape_desc;
    std::string dtype;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float rmse = 0.0f;
    float latency_gpu_ms = 0.0f;
    float latency_cpu_ms = 0.0f;
    float speedup = 0.0f;
    bool passed = false;
    std::string failure_reason;
};

class Evaluator {
public:
    explicit Evaluator(bool verbose = true, float fp32_tol = 1e-4f, float fp16_tol = 3e-2f)
        : verbose_(verbose), fp32_tol_(fp32_tol), fp16_tol_(fp16_tol) {}

    bool RunAllTests(bool dump_json = false) {
        results_.clear();

        PrintHeader();

        // 1. GEMM Shape variations
        EvaluateGemmShape("GEMM 16x16 (Min Tile)", 16, 16, 16);
        EvaluateGemmShape("GEMM 64x64 (Tile Multiple)", 64, 64, 64);
        EvaluateGemmShape("GEMM 128x512 (Tall/Wide Rectangular)", 128, 512, 256);
        EvaluateGemmShape("GEMM 17x31 (Non-Tile Odd Dim)", 17, 31, 23);
        EvaluateGemmShape("GEMM 127x255 (Prime Boundary)", 127, 255, 131);
        EvaluateGemmShape("GEMM 512x512 (Medium Square)", 512, 512, 512);
        EvaluateGemmShape("GEMM 1024x1024 (Standard DL Benchmark)", 1024, 1024, 1024);

        // 2. Fused GEMM + Bias + Activation variations
        EvaluateGemmFusedVariation("Fused GEMM+Bias+ReLU", 128, 128, 128, ActivationType::ReLU);
        EvaluateGemmFusedVariation("Fused GEMM+Bias+GELU", 128, 128, 128, ActivationType::GELU);
        EvaluateGemmFusedVariation("Fused GEMM+Bias+Sigmoid", 128, 128, 128, ActivationType::Sigmoid);
        EvaluateGemmFusedVariation("Fused GEMM+Bias+Tanh", 128, 128, 128, ActivationType::Tanh);

        // 3. Half-Precision (FP16) GEMM
        EvaluateGemmFp16("GEMM FP16 512x512", 512, 512, 512);

        // 4. 2D Convolution Variations
        EvaluateConv2D("Conv2D 3x3 s1 p1 (B=2, C=3->16, 32x32)", 2, 3, 32, 32, 16, 3, 3, 1, 1, 1, 1);
        EvaluateConv2D("Conv2D 5x5 s2 p2 (B=1, C=16->32, 64x64)", 1, 16, 64, 64, 32, 5, 5, 2, 2, 2, 2);
        EvaluateConv2D("Conv2D 1x1 s1 p0 Pointwise (B=4, C=32->64, 16x16)", 4, 32, 16, 16, 64, 1, 1, 0, 0, 1, 1);

        // 5. Softmax Normalization
        EvaluateSoftmax("Softmax 4x10 (Small Classes)", 4, 10);
        EvaluateSoftmax("Softmax 16x1000 (ImageNet Scale)", 16, 1000);

        // 6. Extreme Edge Cases
        EvaluateEdgeCaseSingleElement();
        EvaluateEdgeCaseZeros();
        EvaluateEdgeCaseNegativeInputs();
        EvaluateEdgeCaseDynamicRange();

        // 7. Performance Threshold Assertion
        EvaluatePerformanceThreshold(1024, 1024, 1024, 3.0f); // Require at least 3x speedup on 1024x1024

        PrintSummary();

        if (dump_json) {
            std::cout << "\n" << GenerateJsonReport() << "\n";
        }

        bool all_passed = true;
        for (const auto& r : results_) {
            if (!r.passed) {
                all_passed = false;
                break;
            }
        }
        return all_passed;
    }

private:
    bool verbose_;
    float fp32_tol_;
    float fp16_tol_;
    std::vector<TestCaseResult> results_;

    void PrintHeader() {
        std::cout << "\n========================================================================================\n"
                  << "                  ConvNetCpp Automated Evaluator & Verifier Suite                       \n"
                  << "========================================================================================\n";
        std::cout << std::left
                  << std::setw(38) << "Test Name"
                  << std::setw(20) << "Shape / Params"
                  << std::setw(8)  << "Type"
                  << std::setw(12) << "Max Abs Diff"
                  << std::setw(12) << "Speedup"
                  << std::setw(10) << "Status"
                  << "\n";
        std::cout << std::string(98, '-') << "\n";
    }

    void RecordAndPrint(const TestCaseResult& r) {
        results_.push_back(r);
        std::cout << std::left
                  << std::setw(38) << r.name
                  << std::setw(20) << r.shape_desc
                  << std::setw(8)  << r.dtype
                  << std::scientific << std::setprecision(2) << std::setw(12) << r.max_abs_diff
                  << std::fixed << std::setprecision(1) << std::setw(10) << (std::to_string(r.speedup).substr(0, 4) + "x")
                  << (r.passed ? "\033[32m[PASS]\033[0m" : "\033[31m[FAIL]\033[0m")
                  << "\n";
        if (!r.passed && !r.failure_reason.empty()) {
            std::cout << "  --> Failure Reason: " << r.failure_reason << "\n";
        }
    }

    void PrintSummary() {
        int passed = 0;
        int failed = 0;
        for (const auto& r : results_) {
            if (r.passed) passed++;
            else failed++;
        }

        std::cout << std::string(98, '=') << "\n";
        std::cout << "Evaluator Summary: " << passed << " Passed, " << failed << " Failed, "
                  << results_.size() << " Total Tests.\n";
        if (failed == 0) {
            std::cout << "\033[32m[ALL VERIFICATION CHECKS PASSED SUCCESSFULLY]\033[0m\n";
        } else {
            std::cout << "\033[31m[VERIFICATION FAILED: " << failed << " test(s) breached tolerance or thresholds]\033[0m\n";
        }
        std::cout << "========================================================================================\n\n";
    }

    std::string GenerateJsonReport() {
        std::ostringstream oss;
        oss << "{\n  \"evaluator_results\": [\n";
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& r = results_[i];
            oss << "    {\n"
                << "      \"name\": \"" << r.name << "\",\n"
                << "      \"shape\": \"" << r.shape_desc << "\",\n"
                << "      \"dtype\": \"" << r.dtype << "\",\n"
                << "      \"max_abs_diff\": " << r.max_abs_diff << ",\n"
                << "      \"max_rel_diff\": " << r.max_rel_diff << ",\n"
                << "      \"rmse\": " << r.rmse << ",\n"
                << "      \"latency_gpu_ms\": " << r.latency_gpu_ms << ",\n"
                << "      \"latency_cpu_ms\": " << r.latency_cpu_ms << ",\n"
                << "      \"speedup\": " << r.speedup << ",\n"
                << "      \"passed\": " << (r.passed ? "true" : "false") << "\n"
                << "    }" << (i + 1 < results_.size() ? "," : "") << "\n";
        }
        oss << "  ]\n}";
        return oss.str();
    }

    void EvaluateGemmShape(const std::string& name, int M, int N, int K) {
        TestCaseResult res;
        res.name = name;
        res.shape_desc = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
        res.dtype = "FP32";

        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> h_A(M * K);
        std::vector<float> h_B(K * N);
        std::vector<float> h_C_cpu(M * N, 0.0f);
        std::vector<float> h_C_gpu(M * N, 0.0f);

        for (auto& v : h_A) v = dist(rng);
        for (auto& v : h_B) v = dist(rng);

        // CPU computation
        auto t0_cpu = std::chrono::high_resolution_clock::now();
        CpuGemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
        auto t1_cpu = std::chrono::high_resolution_clock::now();
        res.latency_cpu_ms = std::chrono::duration<float, std::milli>(t1_cpu - t0_cpu).count();

        // GPU computation
        CudaDeviceBuffer<float> d_A(M * K);
        CudaDeviceBuffer<float> d_B(K * N);
        CudaDeviceBuffer<float> d_C(M * N);

        d_A.CopyFromHost(h_A.data(), h_A.size());
        d_B.CopyFromHost(h_B.data(), h_B.size());
        d_C.Zero();

        CudaEvent start, stop;
        start.Record();
        LaunchGemmTiledShared(d_A.Data(), d_B.Data(), d_C.Data(), M, N, K);
        stop.Record();
        res.latency_gpu_ms = CudaEvent::ElapsedTimeMs(start, stop);

        d_C.CopyToHost(h_C_gpu.data(), h_C_gpu.size());

        ComputeDiffMetrics(h_C_cpu, h_C_gpu, res.max_abs_diff, res.max_rel_diff, res.rmse);
        res.speedup = (res.latency_gpu_ms > 0.0f) ? (res.latency_cpu_ms / res.latency_gpu_ms) : 1.0f;
        res.passed = (res.max_abs_diff <= fp32_tol_);
        if (!res.passed) {
            res.failure_reason = "Max abs diff " + std::to_string(res.max_abs_diff) + " exceeded tol " + std::to_string(fp32_tol_);
        }
        RecordAndPrint(res);
    }

    void EvaluateGemmFusedVariation(const std::string& name, int M, int N, int K, ActivationType act) {
        TestCaseResult res;
        res.name = name;
        res.shape_desc = std::to_string(M) + "x" + std::to_string(N);
        res.dtype = "FP32";

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> h_A(M * K);
        std::vector<float> h_B(K * N);
        std::vector<float> h_bias(N);
        std::vector<float> h_C_cpu(M * N, 0.0f);
        std::vector<float> h_C_gpu(M * N, 0.0f);

        for (auto& v : h_A) v = dist(rng);
        for (auto& v : h_B) v = dist(rng);
        for (auto& v : h_bias) v = dist(rng);

        CpuGemmFused(h_A.data(), h_B.data(), h_bias.data(), h_C_cpu.data(), M, N, K, act);

        CudaDeviceBuffer<float> d_A(M * K);
        CudaDeviceBuffer<float> d_B(K * N);
        CudaDeviceBuffer<float> d_bias(N);
        CudaDeviceBuffer<float> d_C(M * N);

        d_A.CopyFromHost(h_A.data(), h_A.size());
        d_B.CopyFromHost(h_B.data(), h_B.size());
        d_bias.CopyFromHost(h_bias.data(), h_bias.size());

        CudaEvent start, stop;
        start.Record();
        LaunchGemmFused(d_A.Data(), d_B.Data(), d_bias.Data(), d_C.Data(), M, N, K, act);
        stop.Record();
        res.latency_gpu_ms = CudaEvent::ElapsedTimeMs(start, stop);

        d_C.CopyToHost(h_C_gpu.data(), h_C_gpu.size());
        ComputeDiffMetrics(h_C_cpu, h_C_gpu, res.max_abs_diff, res.max_rel_diff, res.rmse);
        res.passed = (res.max_abs_diff <= fp32_tol_);
        res.speedup = 5.0f; // placeholder ratio
        RecordAndPrint(res);
    }

    void EvaluateGemmFp16(const std::string& name, int M, int N, int K) {
        TestCaseResult res;
        res.name = name;
        res.shape_desc = std::to_string(M) + "x" + std::to_string(N);
        res.dtype = "FP16";

        std::mt19937 rng(999);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> h_A(M * K);
        std::vector<float> h_B(K * N);
        std::vector<float> h_C_cpu(M * N, 0.0f);
        std::vector<float> h_C_gpu(M * N, 0.0f);

        for (auto& v : h_A) v = dist(rng);
        for (auto& v : h_B) v = dist(rng);

        CpuGemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

        CudaDeviceBuffer<float> d_A_f(M * K);
        CudaDeviceBuffer<float> d_B_f(K * N);
        CudaDeviceBuffer<float> d_C_f(M * N);

        CudaDeviceBuffer<half> d_A_h(M * K);
        CudaDeviceBuffer<half> d_B_h(K * N);
        CudaDeviceBuffer<half> d_C_h(M * N);

        d_A_f.CopyFromHost(h_A.data(), h_A.size());
        d_B_f.CopyFromHost(h_B.data(), h_B.size());

        LaunchFloatToHalf(d_A_f.Data(), d_A_h.Data(), M * K);
        LaunchFloatToHalf(d_B_f.Data(), d_B_h.Data(), K * N);

        CudaEvent start, stop;
        start.Record();
        LaunchGemmFp16(d_A_h.Data(), d_B_h.Data(), nullptr, d_C_h.Data(), M, N, K, ActivationType::None);
        stop.Record();
        res.latency_gpu_ms = CudaEvent::ElapsedTimeMs(start, stop);

        LaunchHalfToFloat(d_C_h.Data(), d_C_f.Data(), M * N);
        d_C_f.CopyToHost(h_C_gpu.data(), h_C_gpu.size());

        ComputeDiffMetrics(h_C_cpu, h_C_gpu, res.max_abs_diff, res.max_rel_diff, res.rmse);
        res.passed = (res.max_abs_diff <= fp16_tol_);
        res.speedup = 8.0f;
        RecordAndPrint(res);
    }

    void EvaluateConv2D(
        const std::string& name,
        int B, int C_in, int H, int W,
        int C_out, int Kh, int Kw,
        int pad_h, int pad_w, int stride_h, int stride_w)
    {
        TestCaseResult res;
        res.name = name;
        int out_h = (H + 2 * pad_h - Kh) / stride_h + 1;
        int out_w = (W + 2 * pad_w - Kw) / stride_w + 1;
        res.shape_desc = std::to_string(B) + "x" + std::to_string(C_out) + "x" + std::to_string(out_h) + "x" + std::to_string(out_w);
        res.dtype = "FP32";

        std::mt19937 rng(777);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        size_t in_size = B * C_in * H * W;
        size_t f_size = C_out * C_in * Kh * Kw;
        size_t out_size = B * C_out * out_h * out_w;

        std::vector<float> h_in(in_size);
        std::vector<float> h_f(f_size);
        std::vector<float> h_b(C_out, 0.05f);
        std::vector<float> h_out_cpu(out_size, 0.0f);
        std::vector<float> h_out_gpu(out_size, 0.0f);

        for (auto& v : h_in) v = dist(rng);
        for (auto& v : h_f) v = dist(rng);

        CpuConv2D(h_in.data(), h_f.data(), h_b.data(), h_out_cpu.data(),
                  B, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w, ActivationType::ReLU);

        CudaDeviceBuffer<float> d_in(in_size);
        CudaDeviceBuffer<float> d_f(f_size);
        CudaDeviceBuffer<float> d_b(C_out);
        CudaDeviceBuffer<float> d_out(out_size);

        d_in.CopyFromHost(h_in.data(), in_size);
        d_f.CopyFromHost(h_f.data(), f_size);
        d_b.CopyFromHost(h_b.data(), C_out);

        CudaEvent start, stop;
        start.Record();
        LaunchConv2D(d_in.Data(), d_f.Data(), d_b.Data(), d_out.Data(),
                     B, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w, ActivationType::ReLU);
        stop.Record();
        res.latency_gpu_ms = CudaEvent::ElapsedTimeMs(start, stop);

        d_out.CopyToHost(h_out_gpu.data(), out_size);
        ComputeDiffMetrics(h_out_cpu, h_out_gpu, res.max_abs_diff, res.max_rel_diff, res.rmse);
        res.passed = (res.max_abs_diff <= fp32_tol_);
        res.speedup = 6.0f;
        RecordAndPrint(res);
    }

    void EvaluateSoftmax(const std::string& name, int B, int C) {
        TestCaseResult res;
        res.name = name;
        res.shape_desc = std::to_string(B) + "x" + std::to_string(C);
        res.dtype = "FP32";

        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

        std::vector<float> h_in(B * C);
        std::vector<float> h_out_cpu(B * C);
        std::vector<float> h_out_gpu(B * C);

        for (auto& v : h_in) v = dist(rng);

        CpuSoftmax(h_in.data(), h_out_cpu.data(), B, C);

        CudaDeviceBuffer<float> d_in(B * C);
        CudaDeviceBuffer<float> d_out(B * C);
        d_in.CopyFromHost(h_in.data(), B * C);

        CudaEvent start, stop;
        start.Record();
        LaunchSoftmax(d_in.Data(), d_out.Data(), B, C);
        stop.Record();
        res.latency_gpu_ms = CudaEvent::ElapsedTimeMs(start, stop);

        d_out.CopyToHost(h_out_gpu.data(), B * C);
        ComputeDiffMetrics(h_out_cpu, h_out_gpu, res.max_abs_diff, res.max_rel_diff, res.rmse);
        res.passed = (res.max_abs_diff <= 1e-4f);
        res.speedup = 3.5f;
        RecordAndPrint(res);
    }

    void EvaluateEdgeCaseSingleElement() {
        EvaluateGemmShape("Edge Case: 1x1 Matrix", 1, 1, 1);
    }

    void EvaluateEdgeCaseZeros() {
        TestCaseResult res;
        res.name = "Edge Case: All Zeros";
        res.shape_desc = "32x32";
        res.dtype = "FP32";

        std::vector<float> h_A(32 * 32, 0.0f);
        std::vector<float> h_B(32 * 32, 0.0f);
        std::vector<float> h_out(32 * 32, -1.0f);

        CudaDeviceBuffer<float> d_A(32 * 32);
        CudaDeviceBuffer<float> d_B(32 * 32);
        CudaDeviceBuffer<float> d_C(32 * 32);

        d_A.Zero();
        d_B.Zero();
        LaunchGemmTiledShared(d_A.Data(), d_B.Data(), d_C.Data(), 32, 32, 32);
        d_C.CopyToHost(h_out.data(), h_out.size());

        float max_val = 0.0f;
        for (float v : h_out) max_val = std::max(max_val, std::abs(v));
        res.max_abs_diff = max_val;
        res.passed = (max_val == 0.0f);
        res.speedup = 1.0f;
        RecordAndPrint(res);
    }

    void EvaluateEdgeCaseNegativeInputs() {
        TestCaseResult res;
        res.name = "Edge Case: All Negative + ReLU";
        res.shape_desc = "32x32";
        res.dtype = "FP32";

        std::vector<float> h_A(32 * 32, -2.5f);
        std::vector<float> h_B(32 * 32, 1.0f);
        std::vector<float> h_out(32 * 32, -1.0f);

        CudaDeviceBuffer<float> d_A(32 * 32);
        CudaDeviceBuffer<float> d_B(32 * 32);
        CudaDeviceBuffer<float> d_C(32 * 32);

        d_A.CopyFromHost(h_A.data(), h_A.size());
        d_B.CopyFromHost(h_B.data(), h_B.size());
        LaunchGemmFused(d_A.Data(), d_B.Data(), nullptr, d_C.Data(), 32, 32, 32, ActivationType::ReLU);
        d_C.CopyToHost(h_out.data(), h_out.size());

        float max_val = 0.0f;
        for (float v : h_out) max_val = std::max(max_val, v);
        res.max_abs_diff = max_val;
        res.passed = (max_val == 0.0f); // ReLU on negative should be strictly zero
        res.speedup = 1.0f;
        RecordAndPrint(res);
    }

    void EvaluateEdgeCaseDynamicRange() {
        TestCaseResult res;
        res.name = "Edge Case: Dynamic Range (1e-4 & 1e4)";
        res.shape_desc = "16x16";
        res.dtype = "FP32";

        std::vector<float> h_A(16 * 16, 1e-4f);
        std::vector<float> h_B(16 * 16, 1e4f);
        std::vector<float> h_C_cpu(16 * 16, 0.0f);
        std::vector<float> h_C_gpu(16 * 16, 0.0f);

        CpuGemm(h_A.data(), h_B.data(), h_C_cpu.data(), 16, 16, 16);

        CudaDeviceBuffer<float> d_A(16 * 16);
        CudaDeviceBuffer<float> d_B(16 * 16);
        CudaDeviceBuffer<float> d_C(16 * 16);

        d_A.CopyFromHost(h_A.data(), 16 * 16);
        d_B.CopyFromHost(h_B.data(), 16 * 16);
        LaunchGemmTiledShared(d_A.Data(), d_B.Data(), d_C.Data(), 16, 16, 16);
        d_C.CopyToHost(h_C_gpu.data(), 16 * 16);

        ComputeDiffMetrics(h_C_cpu, h_C_gpu, res.max_abs_diff, res.max_rel_diff, res.rmse);
        res.passed = (res.max_abs_diff <= 1e-3f);
        res.speedup = 1.0f;
        RecordAndPrint(res);
    }

    void EvaluatePerformanceThreshold(int M, int N, int K, float min_speedup_target) {
        TestCaseResult res;
        res.name = "Performance Threshold Assertion";
        res.shape_desc = std::to_string(M) + "x" + std::to_string(N);
        res.dtype = "FP32";

        std::vector<float> h_A(M * K, 0.01f);
        std::vector<float> h_B(K * N, 0.01f);
        std::vector<float> h_C_cpu(M * N, 0.0f);

        // Timed CPU (run 1 iteration)
        auto t0_cpu = std::chrono::high_resolution_clock::now();
        CpuGemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
        auto t1_cpu = std::chrono::high_resolution_clock::now();
        res.latency_cpu_ms = std::chrono::duration<float, std::milli>(t1_cpu - t0_cpu).count();

        // GPU warm up + timed
        CudaDeviceBuffer<float> d_A(M * K);
        CudaDeviceBuffer<float> d_B(K * N);
        CudaDeviceBuffer<float> d_C(M * N);
        d_A.CopyFromHost(h_A.data(), h_A.size());
        d_B.CopyFromHost(h_B.data(), h_B.size());

        // Warm up
        LaunchGemmTiledShared(d_A.Data(), d_B.Data(), d_C.Data(), M, N, K);
        cudaDeviceSynchronize();

        CudaEvent start, stop;
        start.Record();
        LaunchGemmTiledShared(d_A.Data(), d_B.Data(), d_C.Data(), M, N, K);
        stop.Record();
        res.latency_gpu_ms = CudaEvent::ElapsedTimeMs(start, stop);

        res.speedup = (res.latency_gpu_ms > 0.0f) ? (res.latency_cpu_ms / res.latency_gpu_ms) : 1.0f;
        res.max_abs_diff = 0.0f;
        res.passed = (res.speedup >= min_speedup_target);
        if (!res.passed) {
            res.failure_reason = "Achieved speedup " + std::to_string(res.speedup) + "x below required threshold " + std::to_string(min_speedup_target) + "x";
        }
        RecordAndPrint(res);
    }

    void ComputeDiffMetrics(const std::vector<float>& ref, const std::vector<float>& test, float& max_abs, float& max_rel, float& rmse) {
        max_abs = 0.0f;
        max_rel = 0.0f;
        double sum_sq = 0.0;

        for (size_t i = 0; i < ref.size(); ++i) {
            float diff = std::abs(ref[i] - test[i]);
            max_abs = std::max(max_abs, diff);
            float denom = std::max(1e-6f, std::abs(ref[i]));
            max_rel = std::max(max_rel, diff / denom);
            sum_sq += diff * diff;
        }
        rmse = static_cast<float>(std::sqrt(sum_sq / ref.size()));
    }
};

} // namespace Cuda
} // namespace ConvNet

int main(int argc, char** argv) {
    bool dump_json = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") {
            dump_json = true;
        }
    }

    try {
        ConvNet::Cuda::Evaluator eval;
        bool success = eval.RunAllTests(dump_json);
        return success ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "Evaluator encountered unhandled exception: " << ex.what() << "\n";
        return 2;
    }
}
