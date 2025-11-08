#ifndef _ConvNet_PerformanceTesting_h_
#define _ConvNet_PerformanceTesting_h_

#include "ConvNet.h"
#include "MemoryPool.h"
#include "RuntimeFlexibility.h"
#include "CrtpLayers.h"
#include <chrono>
#include <vector>
#include <string>
#include <map>

namespace ConvNet {

// Performance counter for tracking various metrics
class PerfCounter {
private:
    std::string name;
    std::vector<double> values;
    std::chrono::high_resolution_clock::time_point start_time;
    bool is_timing;
    
public:
    PerfCounter(const std::string& name) : name(name), is_timing(false) {}
    
    void Start() {
        start_time = std::chrono::high_resolution_clock::now();
        is_timing = true;
    }
    
    double Stop() {
        if (is_timing) {
            auto end_time = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            values.push_back(duration);
            is_timing = false;
            return duration;
        }
        return 0.0;
    }
    
    double GetAvg() const {
        if (values.empty()) return 0.0;
        double sum = 0.0;
        for (double v : values) sum += v;
        return sum / values.size();
    }
    
    double GetMin() const {
        if (values.empty()) return 0.0;
        double min_val = values[0];
        for (double v : values) min_val = std::min(min_val, v);
        return min_val;
    }
    
    double GetMax() const {
        if (values.empty()) return 0.0;
        double max_val = values[0];
        for (double v : values) max_val = std::max(max_val, v);
        return max_val;
    }
    
    double GetLast() const {
        if (values.empty()) return 0.0;
        return values.back();
    }
    
    size_t GetCount() const { return values.size(); }
    const std::string& GetName() const { return name; }
    void Reset() { values.clear(); }
};

// Benchmark result structure
struct BenchmarkResult {
    std::string name;
    double execution_time_ms;
    double memory_used_mb;
    int iterations;
    std::string config;
    
    BenchmarkResult(const std::string& n, double time_ms, double mem_mb, int iter, const std::string& cfg = "")
        : name(n), execution_time_ms(time_ms), memory_used_mb(mem_mb), iterations(iter), config(cfg) {}
};

// Performance benchmarking framework
class PerfBenchmark {
private:
    std::vector<BenchmarkResult> results;
    std::map<std::string, PerfCounter> counters;
    static PerfBenchmark* instance;
    
    PerfBenchmark() = default;
    
public:
    static PerfBenchmark& GetInstance() {
        if (!instance) {
            instance = new PerfBenchmark();
        }
        return *instance;
    }
    
    // Add a benchmark result
    void AddResult(const BenchmarkResult& result) {
        results.push_back(result);
    }
    
    // Get performance counter (create if doesn't exist)
    PerfCounter& GetCounter(const std::string& name) {
        return counters[name];
    }
    
    // Run a simple timing benchmark
    template<typename Func>
    double TimeBenchmark(const std::string& name, Func&& func, int iterations = 1) {
        PerfCounter& counter = GetCounter(name);
        double total_time = 0.0;
        
        for (int i = 0; i < iterations; i++) {
            counter.Start();
            func();
            total_time += counter.Stop();
        }
        
        // Add to results
        AddResult(BenchmarkResult(name, total_time, 0, iterations));
        return total_time;
    }
    
    // Benchmark memory usage (approximation)
    template<typename Func>
    double MemoryBenchmark(const std::string& name, Func&& func) {
        // This is a simplified memory benchmark - in a real implementation,
        // you'd track actual memory allocations through the pool
        size_t before_memory = GetMemoryPoolUsage();
        func();
        size_t after_memory = GetMemoryPoolUsage();
        
        double memory_used_mb = (after_memory - before_memory) / (1024.0 * 1024.0);
        
        // Find the corresponding time benchmark to add to
        for (auto& result : results) {
            if (result.name == name) {
                result.memory_used_mb = memory_used_mb;
                break;
            }
        }
        
        return memory_used_mb;
    }
    
    // Get memory usage from pools
    size_t GetMemoryPoolUsage() const {
        // In a real implementation, this would query actual memory usage
        // For now, we'll return an approximation
        return 0;
    }
    
    // Print benchmark results
    void PrintResults() const {
        printf("\n=== Performance Benchmark Results ===\n");
        for (const auto& result : results) {
            printf("%-30s | Time: %8.3f ms | Memory: %8.3f MB | Iterations: %d\n", 
                   result.name.c_str(), result.execution_time_ms, 
                   result.memory_used_mb, result.iterations);
        }
        printf("=====================================\n\n");
    }
    
    // Clear all results
    void ClearResults() { results.clear(); }
    
    // Get all results for programmatic access
    const std::vector<BenchmarkResult>& GetResults() const { return results; }
};

// Singleton instance
PerfBenchmark* PerfBenchmark::instance = nullptr;

// Macro for easy benchmarking
#define BENCHMARK(name, iterations, code) \
    do { \
        auto& benchmark = ConvNet::PerfBenchmark::GetInstance(); \
        benchmark.TimeBenchmark(name, [&]() { code; }, iterations); \
    } while(0)

// Network performance benchmark
class NetworkPerfBenchmark {
public:
    // Compare performance between old and new architectures
    static void CompareNetworkPerformance() {
        printf("Running Network Performance Comparison Benchmark...\n");
        
        // Create test networks of different sizes
        std::vector<std::tuple<int, int, int>> configs = {
            {10, 10, 8},   // Small network
            {20, 20, 16},  // Medium network  
            {32, 32, 32}   // Large network
        };
        
        for (const auto& [input_size, depth, filters] : configs) {
            std::string config_str = std::to_string(input_size) + "x" + std::to_string(input_size) + "x" + std::to_string(depth);
            
            // Test old architecture (existing ConvNet::Net with layers)
            ConvNet::Net old_net;
            old_net.AddLayer().Create<ConvLayer>(3, 3, filters);
            old_net.AddLayer().Create<ReluLayer>();
            old_net.AddLayer().Create<PoolLayer>(2, 2);
            old_net.AddLayer().Create<FullyConnLayer>(10);
            
            // Create test input
            Volume input;
            input.Init(input_size, input_size, depth);
            
            // Time old network forward pass
            BENCHMARK("OldNet_Fwd_" + config_str, 100, {
                old_net.Forward(input, false);
            });
            
            // Time old network backward pass  
            Vector<double> labels;
            labels.SetCount(10, 0.1);
            BENCHMARK("OldNet_Bwd_" + config_str, 100, {
                old_net.Backward(labels);
            });
            
            // Test new architecture (CRTP-based network)
            // We'll benchmark individual layer performance since full CRTP network 
            // implementation is for compile-time optimization
            ConvLayerCRTP conv_layer(3, 3, filters);
            conv_layer.Init(input_size, input_size, depth);
            
            ReluLayerCRTP relu_layer;
            relu_layer.Init(input_size, input_size, filters); // Approximate output size
            
            BENCHMARK("ConvLayerCRTP_Fwd_" + config_str, 100, {
                conv_layer.Forward(input, false);
            });
            
            BENCHMARK("ReluLayerCRTP_Fwd_" + config_str, 100, {
                Volume temp_output = conv_layer.GetOutput();
                relu_layer.Forward(temp_output, false);
            });
        }
    }
    
    // Memory usage comparison
    static void CompareMemoryUsage() {
        printf("Running Memory Usage Comparison Benchmark...\n");
        
        // Test memory pool efficiency
        BENCHMARK("MemoryPool_Allocation", 1000, {
            PoolMat mat(100, 100);
            mat.Init(100, 100, 0.1);
        });
        
        BENCHMARK("Standard_Vector_Allocation", 1000, {
            Volume vol;
            vol.Init(100, 100, 1);
        });
    }
    
    // Comprehensive benchmark suite
    static void RunAllBenchmarks() {
        printf("\n=== Starting Performance Benchmark Suite ===\n");
        
        CompareNetworkPerformance();
        CompareMemoryUsage();
        
        auto& benchmark = PerfBenchmark::GetInstance();
        benchmark.PrintResults();
    }
};

// Utility function to run benchmarks easily
inline void RunPerformanceBenchmarks() {
    NetworkPerfBenchmark::RunAllBenchmarks();
}

} // namespace ConvNet

#endif