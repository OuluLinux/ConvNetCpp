#include "Classify2DCuda.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

int main(int argc, char** argv) {
    using namespace ConvNet::Cuda;

    std::string dataset_type = "spiral";
    int epochs = 100;
    int grid_res = 100;
    float lr = 0.2f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--circle") dataset_type = "circle";
        else if (arg == "--spiral") dataset_type = "spiral";
        else if (arg == "--clusters") dataset_type = "clusters";
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::atoi(argv[++i]);
        else if (arg == "--grid" && i + 1 < argc) grid_res = std::atoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::atof(argv[++i]);
    }

    std::cout << "========================================================================\n"
              << "       ConvNetCpp Classify2D - CUDA Accelerated 2D Neural Classifier    \n"
              << "========================================================================\n\n";

    // 1. Generate Dataset
    Classify2DDataset dataset;
    if (dataset_type == "circle") {
        dataset = Classify2DDataset::GenerateCircle(400);
        std::cout << "Dataset: Concentric Circles (400 samples)\n";
    } else if (dataset_type == "clusters") {
        dataset = Classify2DDataset::GenerateSimpleClusters(300);
        std::cout << "Dataset: Gaussian Clusters (300 samples)\n";
    } else {
        dataset = Classify2DDataset::GenerateSpiral(400);
        std::cout << "Dataset: Intertwined Archimedean Spirals (400 samples)\n";
    }

    // 2. Initialize Model
    int hidden_dim = 16;
    Classify2DNetCuda model(hidden_dim);
    std::cout << "Model Architecture: 2 Inputs -> FC+Tanh(" << hidden_dim
              << ") -> FC+Tanh(" << hidden_dim << ") -> Softmax(2 Classes)\n\n";

    // 3. Training Loop on GPU
    std::cout << "[Training on GPU: " << epochs << " epochs, learning rate = " << lr << "]\n";
    std::cout << std::left
              << std::setw(10) << "Epoch"
              << std::setw(16) << "Cross-Entropy Loss"
              << std::setw(16) << "Accuracy (%)"
              << std::setw(14) << "Time (ms)"
              << "\n";
    std::cout << std::string(56, '-') << "\n";

    for (int ep = 1; ep <= epochs; ++ep) {
        Classify2DStats stats = model.TrainEpoch(dataset, lr);

        if (ep == 1 || ep % 20 == 0 || ep == epochs) {
            std::cout << std::left
                      << std::setw(10) << ep
                      << std::fixed << std::setprecision(4) << std::setw(16) << stats.loss
                      << std::setprecision(1) << std::setw(16) << stats.accuracy
                      << std::setprecision(3) << std::setw(14) << stats.epoch_time_ms
                      << "\n";
        }
    }
    std::cout << "\n";

    // 4. Benchmark Decision Boundary Grid Generation
    model.BenchmarkGridEvaluation(grid_res);

    // 5. Render ASCII Decision Boundary Map
    std::cout << "Learned Decision Boundary (" << 30 << "x" << 15 << " ASCII projection):\n";
    int ascii_w = 40;
    int ascii_h = 20;
    auto grid = model.PredictGrid(ascii_w, 4.5f);

    std::cout << "+" << std::string(ascii_w, '-') << "+\n";
    for (int r = 0; r < ascii_h; ++r) {
        std::cout << "|";
        for (int c = 0; c < ascii_w; ++c) {
            float p0 = grid[r * ascii_w + c];
            if (p0 > 0.65f) std::cout << "\033[34m.\033[0m"; // Class 0 region
            else if (p0 < 0.35f) std::cout << "\033[31m#\033[0m"; // Class 1 region
            else std::cout << "\033[33m~\033[0m"; // Decision boundary transition
        }
        std::cout << "|\n";
    }
    std::cout << "+" << std::string(ascii_w, '-') << "+\n";
    std::cout << "Legend: \033[34m.\033[0m = Class 0, \033[31m#\033[0m = Class 1, \033[33m~\033[0m = Boundary\n\n";

    std::cout << "[CLASSIFY2D CUDA COMPLETE]\n";
    return 0;
}
