#ifndef _CLASSIFY2D_CUDA_H_
#define _CLASSIFY2D_CUDA_H_

#include <ConvNet/Cuda/CudaUtils.h>
#include <ConvNet/Cuda/CudaKernels.h>
#include <vector>
#include <string>

namespace ConvNet {
namespace Cuda {

struct Point2D {
    float x;
    float y;
    int label; // 0 or 1
};

struct Classify2DDataset {
    std::vector<float> inputs;   // Flattened [N, 2]
    std::vector<int> labels;     // [N]
    int count = 0;

    static Classify2DDataset GenerateCircle(int count = 200, unsigned int seed = 42);
    static Classify2DDataset GenerateSpiral(int count = 200, unsigned int seed = 42);
    static Classify2DDataset GenerateSimpleClusters(int count = 100, unsigned int seed = 42);
};

struct Classify2DStats {
    float loss = 0.0f;
    float accuracy = 0.0f;
    double epoch_time_ms = 0.0;
    double grid_eval_time_ms = 0.0;
};

// High-Performance GPU-accelerated 2D Classifier
class Classify2DNetCuda {
public:
    Classify2DNetCuda(int hidden_dim = 16);
    ~Classify2DNetCuda();

    // Initialize model weights on GPU
    void Initialize(unsigned int seed = 1337);

    // Train on a dataset for one epoch using batched GPU operations
    Classify2DStats TrainEpoch(const Classify2DDataset& dataset, float learning_rate = 0.05f);

    // Evaluate dense NxN grid for decision boundary visualization
    // Generates a grid of resolution x resolution points across [-offset, offset]
    // and returns class 0 probability for every grid pixel
    std::vector<float> PredictGrid(int resolution, float offset = 5.0f);

    // Batch prediction for arbitrary test points
    std::vector<float> Predict(const std::vector<float>& h_points, int count);

    // Benchmark comparison: GPU parallel grid evaluation vs CPU point-by-point
    void BenchmarkGridEvaluation(int resolution = 100);

private:
    int hidden_dim_;

    // Layer 1: [2, hidden_dim]
    CudaDeviceBuffer<float> d_W1_;
    CudaDeviceBuffer<float> d_b1_;

    // Layer 2: [hidden_dim, hidden_dim]
    CudaDeviceBuffer<float> d_W2_;
    CudaDeviceBuffer<float> d_b2_;

    // Layer 3: [hidden_dim, 2] (Output logits)
    CudaDeviceBuffer<float> d_W3_;
    CudaDeviceBuffer<float> d_b3_;

    // Gradient accumulators
    CudaDeviceBuffer<float> d_dW1_;
    CudaDeviceBuffer<float> d_db1_;
    CudaDeviceBuffer<float> d_dW2_;
    CudaDeviceBuffer<float> d_db2_;
    CudaDeviceBuffer<float> d_dW3_;
    CudaDeviceBuffer<float> d_db3_;

    CudaStream stream_;
};

} // namespace Cuda
} // namespace ConvNet

#endif // _CLASSIFY2D_CUDA_H_
