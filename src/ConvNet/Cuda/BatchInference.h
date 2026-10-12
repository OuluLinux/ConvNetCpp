#ifndef CONVNET_BATCH_INFERENCE_H
#define CONVNET_BATCH_INFERENCE_H

#include "CudaUtils.h"
#include "CudaKernels.h"
#include <vector>
#include <string>
#include <chrono>

namespace ConvNet {
namespace Cuda {

struct InferenceMetrics {
    int total_queries = 0;
    int total_samples = 0;
    double total_time_ms = 0.0;
    double min_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    double mean_latency_ms = 0.0;
    double p50_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double p99_latency_ms = 0.0;
    double throughput_samples_per_sec = 0.0;
    double throughput_queries_per_sec = 0.0;

    std::string ToJson() const;
    std::string ToMarkdown() const;
};

// High-performance batched model serving pipeline
class BatchInferencePipeline {
public:
    BatchInferencePipeline(int in_features, int hidden_features, int out_classes, int max_batch_size = 64);
    ~BatchInferencePipeline();

    // Initialize weights with deterministic seed
    void InitializeRandomWeights(unsigned int seed = 42);

    // Warm-up runs to stabilize GPU clocks and prime memory
    void WarmUp(int iterations = 10);

    // Synchronous inference for a single batch
    void Infer(
        const float* h_input,
        float* h_output,
        int batch_size
    );

    // Asynchronous multi-stream inference for high concurrency
    void InferStream(
        int stream_idx,
        const float* h_input,
        float* h_output,
        int batch_size
    );

    // Run benchmark over N batches and collect percentiles & throughput
    InferenceMetrics BenchmarkServing(
        int batch_size,
        int num_batches,
        int warmup_batches = 10
    );

    // Verify deterministic reproducibility
    bool VerifyDeterminism(
        const std::vector<float>& sample_input,
        int batch_size,
        int repetitions = 5,
        float tolerance = 1e-6f
    );

    int InFeatures() const { return in_features_; }
    int HiddenFeatures() const { return hidden_features_; }
    int OutClasses() const { return out_classes_; }
    int MaxBatchSize() const { return max_batch_size_; }

private:
    int in_features_;
    int hidden_features_;
    int out_classes_;
    int max_batch_size_;

    // Device weights
    CudaDeviceBuffer<float> d_w1_;      // [in_features, hidden_features]
    CudaDeviceBuffer<float> d_b1_;      // [hidden_features]
    CudaDeviceBuffer<float> d_w2_;      // [hidden_features, out_classes]
    CudaDeviceBuffer<float> d_b2_;      // [out_classes]

    // Device intermediate buffers
    CudaDeviceBuffer<float> d_input_;   // [max_batch_size, in_features]
    CudaDeviceBuffer<float> d_hidden_;  // [max_batch_size, hidden_features]
    CudaDeviceBuffer<float> d_logits_;  // [max_batch_size, out_classes]
    CudaDeviceBuffer<float> d_output_;  // [max_batch_size, out_classes]

    // Multi-stream support
    static constexpr int NUM_STREAMS = 4;
    std::vector<std::unique_ptr<CudaStream>> streams_;

    // Latency recording history
    std::vector<double> recorded_latencies_ms_;
};

} // namespace Cuda
} // namespace ConvNet

#endif // CONVNET_BATCH_INFERENCE_H
