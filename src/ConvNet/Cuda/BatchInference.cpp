#include "BatchInference.h"
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace ConvNet {
namespace Cuda {

BatchInferencePipeline::BatchInferencePipeline(int in_features, int hidden_features, int out_classes, int max_batch_size)
    : in_features_(in_features),
      hidden_features_(hidden_features),
      out_classes_(out_classes),
      max_batch_size_(max_batch_size),
      d_w1_(in_features * hidden_features),
      d_b1_(hidden_features),
      d_w2_(hidden_features * out_classes),
      d_b2_(out_classes),
      d_input_(max_batch_size * in_features),
      d_hidden_(max_batch_size * hidden_features),
      d_logits_(max_batch_size * out_classes),
      d_output_(max_batch_size * out_classes)
{
    for (int i = 0; i < NUM_STREAMS; ++i) {
        streams_.push_back(std::make_unique<CudaStream>());
    }
    InitializeRandomWeights(42);
}

BatchInferencePipeline::~BatchInferencePipeline() = default;

void BatchInferencePipeline::InitializeRandomWeights(unsigned int seed) {
    std::mt19937 rng(seed);
    float std_w1 = std::sqrt(2.0f / in_features_);
    float std_w2 = std::sqrt(2.0f / hidden_features_);
    std::normal_distribution<float> dist_w1(0.0f, std_w1);
    std::normal_distribution<float> dist_w2(0.0f, std_w2);

    std::vector<float> h_w1(in_features_ * hidden_features_);
    for (auto& val : h_w1) val = dist_w1(rng);

    std::vector<float> h_b1(hidden_features_, 0.01f);

    std::vector<float> h_w2(hidden_features_ * out_classes_);
    for (auto& val : h_w2) val = dist_w2(rng);

    std::vector<float> h_b2(out_classes_, 0.0f);

    d_w1_.CopyFromHost(h_w1.data(), h_w1.size());
    d_b1_.CopyFromHost(h_b1.data(), h_b1.size());
    d_w2_.CopyFromHost(h_w2.data(), h_w2.size());
    d_b2_.CopyFromHost(h_b2.data(), h_b2.size());
}

void BatchInferencePipeline::WarmUp(int iterations) {
    std::vector<float> dummy_in(max_batch_size_ * in_features_, 0.5f);
    std::vector<float> dummy_out(max_batch_size_ * out_classes_, 0.0f);

    for (int i = 0; i < iterations; ++i) {
        Infer(dummy_in.data(), dummy_out.data(), max_batch_size_);
    }
}

void BatchInferencePipeline::Infer(
    const float* h_input,
    float* h_output,
    int batch_size)
{
    InferStream(0, h_input, h_output, batch_size);
    streams_[0]->Synchronize();
}

void BatchInferencePipeline::InferStream(
    int stream_idx,
    const float* h_input,
    float* h_output,
    int batch_size)
{
    if (batch_size > max_batch_size_) {
        throw std::invalid_argument("Batch size exceeds configured max_batch_size");
    }

    cudaStream_t stream = streams_[stream_idx % NUM_STREAMS]->Get();

    // 1. Asynchronously copy batch input to GPU
    d_input_.CopyFromHost(h_input, batch_size * in_features_, stream);

    // 2. Layer 1: GEMM + Fused Bias + ReLU activation
    // [batch_size, in_features] * [in_features, hidden_features] + bias1 -> [batch_size, hidden_features]
    LaunchGemmFused(
        d_input_.Data(),
        d_w1_.Data(),
        d_b1_.Data(),
        d_hidden_.Data(),
        batch_size,
        hidden_features_,
        in_features_,
        ActivationType::ReLU,
        1.0f,
        0.0f,
        stream
    );

    // 3. Layer 2: GEMM + Fused Bias (Linear output before softmax)
    // [batch_size, hidden_features] * [hidden_features, out_classes] + bias2 -> [batch_size, out_classes]
    LaunchGemmFused(
        d_hidden_.Data(),
        d_w2_.Data(),
        d_b2_.Data(),
        d_logits_.Data(),
        batch_size,
        out_classes_,
        hidden_features_,
        ActivationType::None,
        1.0f,
        0.0f,
        stream
    );

    // 4. Layer 3: Softmax probability normalization
    LaunchSoftmax(
        d_logits_.Data(),
        d_output_.Data(),
        batch_size,
        out_classes_,
        stream
    );

    // 5. Asynchronously copy output probabilities back to host
    d_output_.CopyToHost(h_output, batch_size * out_classes_, stream);
}

InferenceMetrics BatchInferencePipeline::BenchmarkServing(
    int batch_size,
    int num_batches,
    int warmup_batches)
{
    if (batch_size > max_batch_size_) {
        throw std::invalid_argument("Benchmark batch_size exceeds max_batch_size");
    }

    std::vector<float> h_input(batch_size * in_features_, 0.25f);
    std::vector<float> h_output(batch_size * out_classes_);

    // 1. Warm-up
    for (int i = 0; i < warmup_batches; ++i) {
        Infer(h_input.data(), h_output.data(), batch_size);
    }

    // 2. Timed steady-state benchmark runs
    std::vector<double> latencies;
    latencies.reserve(num_batches);

    CudaEvent start_event;
    CudaEvent stop_event;

    auto t_start_wall = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_batches; ++i) {
        start_event.Record(streams_[0]->Get());
        InferStream(0, h_input.data(), h_output.data(), batch_size);
        stop_event.Record(streams_[0]->Get());
        stop_event.Synchronize();

        float ms = CudaEvent::ElapsedTimeMs(start_event, stop_event);
        latencies.push_back(ms);
    }

    auto t_stop_wall = std::chrono::high_resolution_clock::now();
    double total_wall_ms = std::chrono::duration<double, std::milli>(t_stop_wall - t_start_wall).count();

    std::sort(latencies.begin(), latencies.end());

    InferenceMetrics m;
    m.total_queries = num_batches;
    m.total_samples = num_batches * batch_size;
    m.total_time_ms = total_wall_ms;
    m.min_latency_ms = latencies.front();
    m.max_latency_ms = latencies.back();

    double sum = 0.0;
    for (double v : latencies) sum += v;
    m.mean_latency_ms = sum / latencies.size();

    size_t idx_50 = static_cast<size_t>(latencies.size() * 0.50);
    size_t idx_95 = static_cast<size_t>(latencies.size() * 0.95);
    size_t idx_99 = static_cast<size_t>(latencies.size() * 0.99);

    m.p50_latency_ms = latencies[std::min(idx_50, latencies.size() - 1)];
    m.p95_latency_ms = latencies[std::min(idx_95, latencies.size() - 1)];
    m.p99_latency_ms = latencies[std::min(idx_99, latencies.size() - 1)];

    double total_sec = total_wall_ms / 1000.0;
    m.throughput_queries_per_sec = (total_sec > 0.0) ? (num_batches / total_sec) : 0.0;
    m.throughput_samples_per_sec = (total_sec > 0.0) ? (m.total_samples / total_sec) : 0.0;

    return m;
}

bool BatchInferencePipeline::VerifyDeterminism(
    const std::vector<float>& sample_input,
    int batch_size,
    int repetitions,
    float tolerance)
{
    if (sample_input.size() < static_cast<size_t>(batch_size * in_features_)) {
        return false;
    }

    std::vector<float> first_output(batch_size * out_classes_);
    Infer(sample_input.data(), first_output.data(), batch_size);

    std::vector<float> check_output(batch_size * out_classes_);
    for (int rep = 1; rep < repetitions; ++rep) {
        Infer(sample_input.data(), check_output.data(), batch_size);

        for (size_t i = 0; i < first_output.size(); ++i) {
            float diff = std::abs(first_output[i] - check_output[i]);
            if (diff > tolerance) {
                return false;
            }
        }
    }
    return true;
}

std::string InferenceMetrics::ToJson() const {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"total_queries\": " << total_queries << ",\n"
        << "  \"total_samples\": " << total_samples << ",\n"
        << "  \"total_time_ms\": " << std::fixed << std::setprecision(3) << total_time_ms << ",\n"
        << "  \"min_latency_ms\": " << min_latency_ms << ",\n"
        << "  \"mean_latency_ms\": " << mean_latency_ms << ",\n"
        << "  \"p50_latency_ms\": " << p50_latency_ms << ",\n"
        << "  \"p95_latency_ms\": " << p95_latency_ms << ",\n"
        << "  \"p99_latency_ms\": " << p99_latency_ms << ",\n"
        << "  \"max_latency_ms\": " << max_latency_ms << ",\n"
        << "  \"throughput_samples_per_sec\": " << throughput_samples_per_sec << ",\n"
        << "  \"throughput_queries_per_sec\": " << throughput_queries_per_sec << "\n"
        << "}";
    return oss.str();
}

std::string InferenceMetrics::ToMarkdown() const {
    std::ostringstream oss;
    oss << "| Metric | Value |\n"
        << "|:---|:---|\n"
        << "| Total Queries | " << total_queries << " |\n"
        << "| Total Samples | " << total_samples << " |\n"
        << "| Throughput | " << std::fixed << std::setprecision(1) << throughput_samples_per_sec << " samples/s (" << throughput_queries_per_sec << " queries/s) |\n"
        << "| Latency Mean | " << std::fixed << std::setprecision(3) << mean_latency_ms << " ms |\n"
        << "| Latency p50 | " << p50_latency_ms << " ms |\n"
        << "| Latency p95 | " << p95_latency_ms << " ms |\n"
        << "| Latency p99 | " << p99_latency_ms << " ms |\n"
        << "| Latency Min / Max | " << min_latency_ms << " / " << max_latency_ms << " ms |\n";
    return oss.str();
}

} // namespace Cuda
} // namespace ConvNet
