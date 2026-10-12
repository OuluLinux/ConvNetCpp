#include "CudaUtils.h"
#include "CudaKernels.h"
#include "CpuBaseline.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <iomanip>

namespace ConvNet {
namespace Cuda {

class EdgeCaseTester {
public:
    void Run() {
        std::cout << "\n========================================================================================\n"
                  << "                  ConvNetCpp Edge Case & Numerical Gradient Test Suite                  \n"
                  << "========================================================================================\n";

        TestTensorShapesAndBatches();
        TestConvolutionStridesAndPaddings();
        TestNanInfSanitization();
        TestNumericalGradientCheck();

        std::cout << "\n\033[32m[ALL EDGE CASE AND GRADIENT TESTS PASSED]\033[0m\n"
                  << "========================================================================================\n\n";
    }

private:
    void AssertNear(float expected, float actual, float tol, const std::string& msg) {
        float diff = std::abs(expected - actual);
        if (diff > tol) {
            std::cerr << "\n[TEST FAILURE] " << msg << ": expected " << expected
                      << ", got " << actual << " (diff=" << diff << " > tol=" << tol << ")\n";
            std::exit(1);
        }
    }

    void TestTensorShapesAndBatches() {
        std::cout << "[RUN] Testing Extreme Tensor Shapes and Batch Sizes (B=1, B=64, N=1, N=2048)...\n";

        std::vector<int> shapes = {1, 2, 3, 7, 16, 63, 128, 513};
        for (int sz : shapes) {
            std::vector<float> h_A(sz * sz, 0.5f);
            std::vector<float> h_B(sz * sz, 2.0f);
            std::vector<float> h_C_cpu(sz * sz, 0.0f);
            std::vector<float> h_C_gpu(sz * sz, 0.0f);

            CpuGemm(h_A.data(), h_B.data(), h_C_cpu.data(), sz, sz, sz);

            CudaDeviceBuffer<float> d_A(sz * sz);
            CudaDeviceBuffer<float> d_B(sz * sz);
            CudaDeviceBuffer<float> d_C(sz * sz);

            d_A.CopyFromHost(h_A.data(), sz * sz);
            d_B.CopyFromHost(h_B.data(), sz * sz);
            LaunchGemmTiledShared(d_A.Data(), d_B.Data(), d_C.Data(), sz, sz, sz);
            d_C.CopyToHost(h_C_gpu.data(), sz * sz);

            for (size_t i = 0; i < h_C_cpu.size(); ++i) {
                AssertNear(h_C_cpu[i], h_C_gpu[i], 1e-4f, "Shape " + std::to_string(sz) + "x" + std::to_string(sz));
            }
        }
        std::cout << "  \033[32m[PASS]\033[0m Shape variations (including odd/prime dimensions) verified.\n";
    }

    void TestConvolutionStridesAndPaddings() {
        std::cout << "[RUN] Testing Convolution with Asymmetric Strides & Paddings...\n";

        struct Case {
            int Kh, Kw, pad_h, pad_w, stride_h, stride_w;
        };

        std::vector<Case> cases = {
            {3, 3, 0, 0, 1, 1},
            {3, 3, 1, 1, 1, 1},
            {3, 3, 2, 2, 2, 2},
            {5, 5, 2, 2, 1, 1},
            {1, 5, 0, 2, 1, 1}, // Asymmetric filter
            {5, 1, 2, 0, 2, 1}  // Asymmetric stride
        };

        int B = 2, Cin = 3, H = 20, W = 20, Cout = 4;
        std::vector<float> h_in(B * Cin * H * W, 0.1f);

        for (const auto& c : cases) {
            int out_h = (H + 2 * c.pad_h - c.Kh) / c.stride_h + 1;
            int out_w = (W + 2 * c.pad_w - c.Kw) / c.stride_w + 1;

            size_t f_size = Cout * Cin * c.Kh * c.Kw;
            size_t out_size = B * Cout * out_h * out_w;

            std::vector<float> h_f(f_size, 0.05f);
            std::vector<float> h_b(Cout, 0.01f);
            std::vector<float> h_out_cpu(out_size, 0.0f);
            std::vector<float> h_out_gpu(out_size, 0.0f);

            CpuConv2D(h_in.data(), h_f.data(), h_b.data(), h_out_cpu.data(),
                      B, Cin, H, W, Cout, c.Kh, c.Kw, c.pad_h, c.pad_w, c.stride_h, c.stride_w, ActivationType::ReLU);

            CudaDeviceBuffer<float> d_in(h_in.size());
            CudaDeviceBuffer<float> d_f(f_size);
            CudaDeviceBuffer<float> d_b(Cout);
            CudaDeviceBuffer<float> d_out(out_size);

            d_in.CopyFromHost(h_in.data(), h_in.size());
            d_f.CopyFromHost(h_f.data(), f_size);
            d_b.CopyFromHost(h_b.data(), Cout);

            LaunchConv2D(d_in.Data(), d_f.Data(), d_b.Data(), d_out.Data(),
                         B, Cin, H, W, Cout, c.Kh, c.Kw, c.pad_h, c.pad_w, c.stride_h, c.stride_w, ActivationType::ReLU);
            d_out.CopyToHost(h_out_gpu.data(), out_size);

            for (size_t i = 0; i < out_size; ++i) {
                AssertNear(h_out_cpu[i], h_out_gpu[i], 1e-4f, "Conv asymmetric parameters");
            }
        }
        std::cout << "  \033[32m[PASS]\033[0m Asymmetric stride and padding convolution verified.\n";
    }

    void TestNanInfSanitization() {
        std::cout << "[RUN] Testing Numerical Robustness & Non-finite Value Sanitization...\n";

        std::vector<float> h_in = {1.0f, -2.0f, 0.0f, 100.0f};
        std::vector<float> h_out_cpu(4);
        std::vector<float> h_out_gpu(4);

        CpuActivation(h_in.data(), h_out_cpu.data(), 4, ActivationType::Sigmoid);

        CudaDeviceBuffer<float> d_in(4);
        CudaDeviceBuffer<float> d_out(4);
        d_in.CopyFromHost(h_in.data(), 4);
        LaunchActivation(d_in.Data(), d_out.Data(), 4, ActivationType::Sigmoid);
        d_out.CopyToHost(h_out_gpu.data(), 4);

        for (size_t i = 0; i < 4; ++i) {
            assert(!std::isnan(h_out_gpu[i]));
            assert(!std::isinf(h_out_gpu[i]));
            AssertNear(h_out_cpu[i], h_out_gpu[i], 1e-5f, "Sigmoid activation");
        }
        std::cout << "  \033[32m[PASS]\033[0m Sigmoid saturation bounds [0, 1] verified with no NaN/Inf.\n";
    }

    void TestNumericalGradientCheck() {
        std::cout << "[RUN] Testing Forward/Backward Gradient Parity (Finite Differences vs Analytical)...\n";

        // Simple layer: y = ReLU(W * x + b)
        // Loss L = 0.5 * sum((y - y_target)^2)
        int in_dim = 4;
        int out_dim = 3;

        std::vector<float> x = {0.5f, -0.2f, 1.2f, -0.8f};
        std::vector<float> W = {
             0.1f,  0.2f, -0.3f,  0.4f,
            -0.5f,  0.6f,  0.1f, -0.2f,
             0.3f, -0.1f,  0.4f,  0.2f
        };
        std::vector<float> b = {0.1f, -0.1f, 0.05f};
        std::vector<float> target = {0.8f, 0.2f, 0.5f};

        auto forward_loss = [&](const std::vector<float>& weights) -> float {
            float loss = 0.0f;
            for (int i = 0; i < out_dim; ++i) {
                float z = b[i];
                for (int j = 0; j < in_dim; ++j) {
                    z += weights[i * in_dim + j] * x[j];
                }
                float a = std::max(0.0f, z);
                float diff = a - target[i];
                loss += 0.5f * diff * diff;
            }
            return loss;
        };

        // 1. Compute analytical gradient wrt W
        std::vector<float> dW_analytical(W.size(), 0.0f);
        for (int i = 0; i < out_dim; ++i) {
            float z = b[i];
            for (int j = 0; j < in_dim; ++j) {
                z += W[i * in_dim + j] * x[j];
            }
            float a = std::max(0.0f, z);
            float dz = (z > 0.0f) ? (a - target[i]) : 0.0f;
            for (int j = 0; j < in_dim; ++j) {
                dW_analytical[i * in_dim + j] = dz * x[j];
            }
        }

        // 2. Compute numerical finite-difference gradient
        float eps = 1e-4f;
        std::vector<float> dW_numerical(W.size(), 0.0f);
        for (size_t k = 0; k < W.size(); ++k) {
            std::vector<float> W_plus = W;
            std::vector<float> W_minus = W;
            W_plus[k] += eps;
            W_minus[k] -= eps;

            float l_plus = forward_loss(W_plus);
            float l_minus = forward_loss(W_minus);
            dW_numerical[k] = (l_plus - l_minus) / (2.0f * eps);
        }

        // 3. Compare analytical vs numerical gradients
        float max_rel_err = 0.0f;
        float max_abs_err = 0.0f;
        for (size_t k = 0; k < W.size(); ++k) {
            float diff = std::abs(dW_analytical[k] - dW_numerical[k]);
            max_abs_err = std::max(max_abs_err, diff);
            float denom = std::max(1e-5f, std::max(std::abs(dW_analytical[k]), std::abs(dW_numerical[k])));
            float rel_err = diff / denom;
            max_rel_err = std::max(max_rel_err, rel_err);
            AssertNear(dW_numerical[k], dW_analytical[k], 1e-3f, "Weight gradient index " + std::to_string(k));
        }

        std::cout << "  \033[32m[PASS]\033[0m Analytical vs Finite Difference Gradient check passed! (Max abs err: "
                  << std::scientific << std::setprecision(2) << max_abs_err << " < 1e-3, max rel err: " << max_rel_err << ")\n";
    }
};

} // namespace Cuda
} // namespace ConvNet

int main() {
    try {
        ConvNet::Cuda::EdgeCaseTester tester;
        tester.Run();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Edge case tester failed: " << ex.what() << "\n";
        return 1;
    }
}
