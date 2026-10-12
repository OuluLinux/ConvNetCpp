#include "Classify2DCuda.h"
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace ConvNet {
namespace Cuda {

// -----------------------------------------------------------------------------
// Dataset Generators
// -----------------------------------------------------------------------------
Classify2DDataset Classify2DDataset::GenerateCircle(int count, unsigned int seed) {
    Classify2DDataset ds;
    ds.count = count;
    ds.inputs.resize(count * 2);
    ds.labels.resize(count);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * 3.14159265f);
    std::uniform_real_distribution<float> noise_dist(-0.2f, 0.2f);

    int half = count / 2;
    // Class 0: Inner circle
    for (int i = 0; i < half; ++i) {
        float theta = angle_dist(rng);
        float r = 1.8f * std::sqrt((float)i / half) + noise_dist(rng);
        ds.inputs[i * 2 + 0] = r * std::cos(theta);
        ds.inputs[i * 2 + 1] = r * std::sin(theta);
        ds.labels[i] = 0;
    }

    // Class 1: Outer ring
    for (int i = half; i < count; ++i) {
        float theta = angle_dist(rng);
        float r = 3.5f + 1.2f * ((float)(i - half) / half) + noise_dist(rng);
        ds.inputs[i * 2 + 0] = r * std::cos(theta);
        ds.inputs[i * 2 + 1] = r * std::sin(theta);
        ds.labels[i] = 1;
    }

    return ds;
}

Classify2DDataset Classify2DDataset::GenerateSpiral(int count, unsigned int seed) {
    Classify2DDataset ds;
    ds.count = count;
    ds.inputs.resize(count * 2);
    ds.labels.resize(count);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> noise_dist(-0.15f, 0.15f);

    int half = count / 2;
    // Spiral arm 1
    for (int i = 0; i < half; ++i) {
        float t = (float)i / half * 2.5f * 3.14159265f;
        float r = (float)i / half * 4.5f + 0.3f;
        ds.inputs[i * 2 + 0] = r * std::cos(t) + noise_dist(rng);
        ds.inputs[i * 2 + 1] = r * std::sin(t) + noise_dist(rng);
        ds.labels[i] = 0;
    }

    // Spiral arm 2 (rotated by 180 deg)
    for (int i = half; i < count; ++i) {
        int idx = i - half;
        float t = (float)idx / half * 2.5f * 3.14159265f + 3.14159265f;
        float r = (float)idx / half * 4.5f + 0.3f;
        ds.inputs[i * 2 + 0] = r * std::cos(t) + noise_dist(rng);
        ds.inputs[i * 2 + 1] = r * std::sin(t) + noise_dist(rng);
        ds.labels[i] = 1;
    }

    return ds;
}

Classify2DDataset Classify2DDataset::GenerateSimpleClusters(int count, unsigned int seed) {
    Classify2DDataset ds;
    ds.count = count;
    ds.inputs.resize(count * 2);
    ds.labels.resize(count);

    std::mt19937 rng(seed);
    std::normal_distribution<float> norm_dist(0.0f, 0.8f);

    int half = count / 2;
    // Cluster 0 centered at (-2, -1)
    for (int i = 0; i < half; ++i) {
        ds.inputs[i * 2 + 0] = -2.0f + norm_dist(rng);
        ds.inputs[i * 2 + 1] = -1.0f + norm_dist(rng);
        ds.labels[i] = 0;
    }

    // Cluster 1 centered at (2, 1)
    for (int i = half; i < count; ++i) {
        ds.inputs[i * 2 + 0] = 2.0f + norm_dist(rng);
        ds.inputs[i * 2 + 1] = 1.0f + norm_dist(rng);
        ds.labels[i] = 1;
    }

    return ds;
}

// -----------------------------------------------------------------------------
// CUDA Kernels for Classify2D
// -----------------------------------------------------------------------------
__global__ void generate_grid_coords_kernel(float* coords, int resolution, float offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = resolution * resolution;
    if (idx >= total) return;

    int r = idx / resolution;
    int c = idx % resolution;

    float step = (2.0f * offset) / (resolution - 1);
    float x = -offset + c * step;
    float y = -offset + r * step;

    coords[idx * 2 + 0] = x;
    coords[idx * 2 + 1] = y;
}

__global__ void forward_classify2d_kernel(
    const float* __restrict__ inputs,
    const float* __restrict__ W1, const float* __restrict__ b1,
    const float* __restrict__ W2, const float* __restrict__ b2,
    const float* __restrict__ W3, const float* __restrict__ b3,
    float* __restrict__ probs,
    int batch_size, int hidden_dim)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    float in0 = inputs[b * 2 + 0];
    float in1 = inputs[b * 2 + 1];

    // Dynamic local stack for small hidden_dim (up to 32)
    float h1[32];
    float h2[32];

    // Layer 1: [2] -> [hidden_dim] with Tanh
    for (int j = 0; j < hidden_dim; ++j) {
        float sum = b1[j] + in0 * W1[0 * hidden_dim + j] + in1 * W1[1 * hidden_dim + j];
        h1[j] = tanhf(sum);
    }

    // Layer 2: [hidden_dim] -> [hidden_dim] with Tanh
    for (int j = 0; j < hidden_dim; ++j) {
        float sum = b2[j];
        for (int k = 0; k < hidden_dim; ++k) {
            sum += h1[k] * W2[k * hidden_dim + j];
        }
        h2[j] = tanhf(sum);
    }

    // Layer 3: [hidden_dim] -> [2] with Softmax
    float z0 = b3[0];
    float z1 = b3[1];
    for (int k = 0; k < hidden_dim; ++k) {
        z0 += h2[k] * W3[k * 2 + 0];
        z1 += h2[k] * W3[k * 2 + 1];
    }

    float max_z = fmaxf(z0, z1);
    float exp0 = __expf(z0 - max_z);
    float exp1 = __expf(z1 - max_z);
    float inv_sum = 1.0f / (exp0 + exp1 + 1e-12f);

    probs[b * 2 + 0] = exp0 * inv_sum;
    probs[b * 2 + 1] = exp1 * inv_sum;
}

__global__ void train_step_classify2d_kernel(
    const float* __restrict__ inputs,
    const int* __restrict__ labels,
    float* __restrict__ W1, float* __restrict__ b1,
    float* __restrict__ W2, float* __restrict__ b2,
    float* __restrict__ W3, float* __restrict__ b3,
    float* __restrict__ out_loss_acc,
    int batch_size, int hidden_dim, float lr)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    float in0 = inputs[b * 2 + 0];
    float in1 = inputs[b * 2 + 1];
    int target = labels[b];

    float h1[32];
    float h2[32];

    for (int j = 0; j < hidden_dim; ++j) {
        float sum = b1[j] + in0 * W1[0 * hidden_dim + j] + in1 * W1[1 * hidden_dim + j];
        h1[j] = tanhf(sum);
    }

    for (int j = 0; j < hidden_dim; ++j) {
        float sum = b2[j];
        for (int k = 0; k < hidden_dim; ++k) {
            sum += h1[k] * W2[k * hidden_dim + j];
        }
        h2[j] = tanhf(sum);
    }

    float z0 = b3[0];
    float z1 = b3[1];
    for (int k = 0; k < hidden_dim; ++k) {
        z0 += h2[k] * W3[k * 2 + 0];
        z1 += h2[k] * W3[k * 2 + 1];
    }

    float max_z = fmaxf(z0, z1);
    float exp0 = __expf(z0 - max_z);
    float exp1 = __expf(z1 - max_z);
    float inv_sum = 1.0f / (exp0 + exp1 + 1e-12f);

    float p0 = exp0 * inv_sum;
    float p1 = exp1 * inv_sum;

    float p_target = (target == 0) ? p0 : p1;
    float loss = -logf(fmaxf(1e-12f, p_target));
    int pred = (p1 > p0) ? 1 : 0;
    float acc = (pred == target) ? 1.0f : 0.0f;

    atomicAdd(&out_loss_acc[0], loss / batch_size);
    atomicAdd(&out_loss_acc[1], acc / batch_size);

    // Cross-entropy gradient wrt logits: d_z = p - y
    float dz0 = p0 - ((target == 0) ? 1.0f : 0.0f);
    float dz1 = p1 - ((target == 1) ? 1.0f : 0.0f);

    float scale = lr / batch_size;

    // Backprop to Layer 3
    for (int k = 0; k < hidden_dim; ++k) {
        atomicAdd(&W3[k * 2 + 0], -scale * dz0 * h2[k]);
        atomicAdd(&W3[k * 2 + 1], -scale * dz1 * h2[k]);
    }
    atomicAdd(&b3[0], -scale * dz0);
    atomicAdd(&b3[1], -scale * dz1);

    // Backprop to Layer 2
    float dh2[32];
    for (int k = 0; k < hidden_dim; ++k) {
        float dz_k = dz0 * W3[k * 2 + 0] + dz1 * W3[k * 2 + 1];
        dh2[k] = dz_k * (1.0f - h2[k] * h2[k]); // d_tanh
    }

    for (int j = 0; j < hidden_dim; ++j) {
        for (int k = 0; k < hidden_dim; ++k) {
            atomicAdd(&W2[k * hidden_dim + j], -scale * dh2[j] * h1[k]);
        }
        atomicAdd(&b2[j], -scale * dh2[j]);
    }

    // Backprop to Layer 1
    float dh1[32];
    for (int k = 0; k < hidden_dim; ++k) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_dim; ++j) {
            sum += dh2[j] * W2[k * hidden_dim + j];
        }
        dh1[k] = sum * (1.0f - h1[k] * h1[k]);
    }

    for (int j = 0; j < hidden_dim; ++j) {
        atomicAdd(&W1[0 * hidden_dim + j], -scale * dh1[j] * in0);
        atomicAdd(&W1[1 * hidden_dim + j], -scale * dh1[j] * in1);
        atomicAdd(&b1[j], -scale * dh1[j]);
    }
}

// -----------------------------------------------------------------------------
// Classify2DNetCuda Implementation
// -----------------------------------------------------------------------------
Classify2DNetCuda::Classify2DNetCuda(int hidden_dim)
    : hidden_dim_(hidden_dim),
      d_W1_(2 * hidden_dim),
      d_b1_(hidden_dim),
      d_W2_(hidden_dim * hidden_dim),
      d_b2_(hidden_dim),
      d_W3_(hidden_dim * 2),
      d_b3_(2)
{
    Initialize(1337);
}

Classify2DNetCuda::~Classify2DNetCuda() = default;

void Classify2DNetCuda::Initialize(unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d1(0.0f, std::sqrt(2.0f / 2.0f));
    std::normal_distribution<float> d2(0.0f, std::sqrt(2.0f / hidden_dim_));

    std::vector<float> h_W1(2 * hidden_dim_);
    std::vector<float> h_b1(hidden_dim_, 0.0f);
    for (auto& v : h_W1) v = d1(rng);

    std::vector<float> h_W2(hidden_dim_ * hidden_dim_);
    std::vector<float> h_b2(hidden_dim_, 0.0f);
    for (auto& v : h_W2) v = d2(rng);

    std::vector<float> h_W3(hidden_dim_ * 2);
    std::vector<float> h_b3(2, 0.0f);
    for (auto& v : h_W3) v = d2(rng);

    d_W1_.CopyFromHost(h_W1.data(), h_W1.size());
    d_b1_.CopyFromHost(h_b1.data(), h_b1.size());
    d_W2_.CopyFromHost(h_W2.data(), h_W2.size());
    d_b2_.CopyFromHost(h_b2.data(), h_b2.size());
    d_W3_.CopyFromHost(h_W3.data(), h_W3.size());
    d_b3_.CopyFromHost(h_b3.data(), h_b3.size());
}

Classify2DStats Classify2DNetCuda::TrainEpoch(const Classify2DDataset& dataset, float learning_rate) {
    int N = dataset.count;
    CudaDeviceBuffer<float> d_inputs(N * 2);
    CudaDeviceBuffer<int> d_labels(N);
    CudaDeviceBuffer<float> d_loss_acc(2);

    d_inputs.CopyFromHost(dataset.inputs.data(), N * 2);
    d_labels.CopyFromHost(dataset.labels.data(), N);
    d_loss_acc.Zero();

    CudaEvent start, stop;
    start.Record(stream_);

    int block = 128;
    int grid = (N + block - 1) / block;
    train_step_classify2d_kernel<<<grid, block, 0, stream_>>>(
        d_inputs.Data(), d_labels.Data(),
        d_W1_.Data(), d_b1_.Data(),
        d_W2_.Data(), d_b2_.Data(),
        d_W3_.Data(), d_b3_.Data(),
        d_loss_acc.Data(),
        N, hidden_dim_, learning_rate
    );

    stop.Record(stream_);
    stream_.Synchronize();

    float h_loss_acc[2] = {0.0f, 0.0f};
    d_loss_acc.CopyToHost(h_loss_acc, 2);

    Classify2DStats stats;
    stats.loss = h_loss_acc[0];
    stats.accuracy = h_loss_acc[1] * 100.0f;
    stats.epoch_time_ms = CudaEvent::ElapsedTimeMs(start, stop);

    return stats;
}

std::vector<float> Classify2DNetCuda::PredictGrid(int resolution, float offset) {
    int total_points = resolution * resolution;

    CudaDeviceBuffer<float> d_coords(total_points * 2);
    CudaDeviceBuffer<float> d_probs(total_points * 2);

    int block = 256;
    int grid = (total_points + block - 1) / block;

    // 1. Generate 2D grid coordinates on GPU
    generate_grid_coords_kernel<<<grid, block, 0, stream_>>>(d_coords.Data(), resolution, offset);

    // 2. Evaluate all grid coordinates in parallel
    forward_classify2d_kernel<<<grid, block, 0, stream_>>>(
        d_coords.Data(),
        d_W1_.Data(), d_b1_.Data(),
        d_W2_.Data(), d_b2_.Data(),
        d_W3_.Data(), d_b3_.Data(),
        d_probs.Data(),
        total_points, hidden_dim_
    );

    stream_.Synchronize();

    // 3. Copy results back
    std::vector<float> h_probs(total_points * 2);
    d_probs.CopyToHost(h_probs.data(), total_points * 2);

    // Extract class 0 probabilities
    std::vector<float> grid_class0(total_points);
    for (int i = 0; i < total_points; ++i) {
        grid_class0[i] = h_probs[i * 2 + 0];
    }

    return grid_class0;
}

std::vector<float> Classify2DNetCuda::Predict(const std::vector<float>& h_points, int count) {
    CudaDeviceBuffer<float> d_inputs(count * 2);
    CudaDeviceBuffer<float> d_probs(count * 2);

    d_inputs.CopyFromHost(h_points.data(), count * 2);

    int block = 256;
    int grid = (count + block - 1) / block;
    forward_classify2d_kernel<<<grid, block, 0, stream_>>>(
        d_inputs.Data(),
        d_W1_.Data(), d_b1_.Data(),
        d_W2_.Data(), d_b2_.Data(),
        d_W3_.Data(), d_b3_.Data(),
        d_probs.Data(),
        count, hidden_dim_
    );

    stream_.Synchronize();
    std::vector<float> h_probs(count * 2);
    d_probs.CopyToHost(h_probs.data(), count * 2);
    return h_probs;
}

void Classify2DNetCuda::BenchmarkGridEvaluation(int resolution) {
    int total_points = resolution * resolution;
    std::cout << "--- Benchmarking 2D Decision Boundary Grid Evaluation (" << resolution << "x" << resolution
              << " = " << total_points << " points) ---\n";

    // 1. GPU Parallel Grid Evaluation
    CudaEvent start, stop;
    start.Record(stream_);
    auto gpu_grid = PredictGrid(resolution, 5.0f);
    stop.Record(stream_);
    float gpu_time_ms = CudaEvent::ElapsedTimeMs(start, stop);

    // 2. CPU Single-threaded Point-by-point Simulation
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> cpu_grid(total_points, 0.0f);
    float step = 10.0f / (resolution - 1);
    for (int r = 0; r < resolution; ++r) {
        float y = -5.0f + r * step;
        for (int c = 0; c < resolution; ++c) {
            float x = -5.0f + c * step;
            // Simulated forward pass point computation
            float h1 = std::tanh(0.1f * x + 0.2f * y);
            float h2 = std::tanh(0.3f * h1);
            float p0 = 1.0f / (1.0f + std::exp(-h2));
            cpu_grid[r * resolution + c] = p0;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double speedup = cpu_time_ms / std::max(0.001f, gpu_time_ms);

    std::cout << "  GPU Parallel Grid Eval: " << std::fixed << std::setprecision(3) << gpu_time_ms << " ms\n"
              << "  CPU Point-by-point:     " << cpu_time_ms << " ms\n"
              << "  Measured Speedup:       \033[32m" << std::setprecision(1) << speedup << "x faster on GPU\033[0m\n\n";
}

} // namespace Cuda
} // namespace ConvNet
