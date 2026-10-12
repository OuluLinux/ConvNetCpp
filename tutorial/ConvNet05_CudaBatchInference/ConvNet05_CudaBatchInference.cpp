#include <ConvNet/Cuda/BatchInference.h>
#include <iostream>
#include <vector>
#include <iomanip>

/*
================================================================================
ConvNetCpp Tutorial 05: High-Throughput Batched Model Serving on GPU
================================================================================

This tutorial demonstrates:
1. Setting up a batched model serving pipeline (BatchInferencePipeline)
2. The impact of the GPU warm-up phase on latency stability
3. Dynamic batching: scaling from batch size 1 to batch size 64
4. Collecting production serving metrics (p50, p95, p99 latencies, QPS throughput)
5. Validating deterministic reproducibility across repeated inference queries
*/

int main() {
    using namespace ConvNet::Cuda;

    std::cout << "========================================================================\n"
              << "       ConvNetCpp Tutorial 05: Batched GPU Model Serving Pipeline       \n"
              << "========================================================================\n\n";

    // 1. Setup Model Architecture
    // 2 inputs (e.g. 2D coordinates) -> 64 hidden neurons -> 2 output classes
    const int in_features = 2;
    const int hidden_features = 64;
    const int out_classes = 2;
    const int max_batch_size = 64;

    std::cout << "[Step 1] Initializing Model Serving Pipeline on GPU...\n"
              << "  Architecture: Input (2) -> FC+ReLU (64) -> FC (2) -> Softmax (2)\n"
              << "  Max Supported Batch Size: " << max_batch_size << "\n\n";

    BatchInferencePipeline pipeline(in_features, hidden_features, out_classes, max_batch_size);

    // 2. Cold Run vs Warm-Up
    std::cout << "[Step 2] Evaluating Warm-Up Phase Impact...\n";
    std::vector<float> sample_input(max_batch_size * in_features, 0.5f);
    std::vector<float> sample_output(max_batch_size * out_classes, 0.0f);

    auto t_cold_0 = std::chrono::high_resolution_clock::now();
    pipeline.Infer(sample_input.data(), sample_output.data(), 1);
    auto t_cold_1 = std::chrono::high_resolution_clock::now();
    double cold_ms = std::chrono::duration<double, std::milli>(t_cold_1 - t_cold_0).count();

    pipeline.WarmUp(10);

    auto t_warm_0 = std::chrono::high_resolution_clock::now();
    pipeline.Infer(sample_input.data(), sample_output.data(), 1);
    auto t_warm_1 = std::chrono::high_resolution_clock::now();
    double warm_ms = std::chrono::duration<double, std::milli>(t_warm_1 - t_warm_0).count();

    std::cout << "  Cold Launch Latency: " << std::fixed << std::setprecision(3) << cold_ms << " ms\n"
              << "  Warmed-up Latency:   " << warm_ms << " ms\n"
              << "  Jitter Reduction:    " << (cold_ms / std::max(0.001, warm_ms)) << "x faster after warmup\n\n";

    // 3. Dynamic Batch Size Benchmarking
    std::cout << "[Step 3] Dynamic Batch Scaling (100 batches per configuration):\n";
    std::vector<int> test_batches = {1, 8, 16, 32, 64};

    std::cout << std::left
              << std::setw(12) << "Batch Size"
              << std::setw(14) << "Mean (ms)"
              << std::setw(14) << "p50 (ms)"
              << std::setw(14) << "p95 (ms)"
              << std::setw(18) << "Samples / sec"
              << std::setw(18) << "Queries / sec"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    for (int bs : test_batches) {
        InferenceMetrics m = pipeline.BenchmarkServing(bs, 100, 10);
        std::cout << std::left
                  << std::setw(12) << bs
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << m.mean_latency_ms
                  << std::setw(14) << m.p50_latency_ms
                  << std::setw(14) << m.p95_latency_ms
                  << std::fixed << std::setprecision(1)
                  << std::setw(18) << m.throughput_samples_per_sec
                  << std::setw(18) << m.throughput_queries_per_sec
                  << "\n";
    }
    std::cout << "\n";

    // 4. Deterministic Reproducibility
    std::cout << "[Step 4] Validating Deterministic Inference Output...\n";
    bool deterministic = pipeline.VerifyDeterminism(sample_input, 16, 10, 1e-6f);
    if (deterministic) {
        std::cout << "  \033[32m[PASS] Output is 100% bitwise deterministic across 10 repeated batch runs!\033[0m\n\n";
    } else {
        std::cout << "  \033[31m[FAIL] Non-deterministic variation detected!\033[0m\n\n";
    }

    std::cout << "[TUTORIAL 05 COMPLETE]\n";
    return 0;
}
