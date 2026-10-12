#ifndef CONVNET_CUDA_MEMORY_POOL_H
#define CONVNET_CUDA_MEMORY_POOL_H

#include "CudaUtils.h"
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstddef>
#include <string>

namespace ConvNet {
namespace Cuda {

struct PoolStatistics {
    size_t total_alloc_requests = 0;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t current_active_allocations = 0;
    size_t current_active_bytes = 0;
    size_t current_pooled_bytes = 0;
    size_t peak_allocated_bytes = 0;
    size_t total_device_memory_allocated = 0;

    double HitRate() const {
        if (total_alloc_requests == 0) return 0.0;
        return (static_cast<double>(cache_hits) / total_alloc_requests) * 100.0;
    }
};

class CudaMemoryPool {
public:
    static CudaMemoryPool& Instance();

    CudaMemoryPool();
    ~CudaMemoryPool();

    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

    // Allocate memory on the GPU (or reuse from pool)
    void* Allocate(size_t size);

    // Return allocated memory back to the pool
    void Deallocate(void* ptr);

    // Free all currently idle cached blocks back to CUDA driver
    void Clear();

    // Pre-populate pool with blocks of given size
    void PreAllocate(size_t size, size_t count);

    // Get current statistics
    PoolStatistics GetStatistics() const;

    // Report summary string
    std::string DumpStatistics() const;

    // Check if there are active unreturned allocations (memory leaks)
    bool HasLeaks() const;

    // Copy operations
    static bool CopyToGPU(void* d_dst, const void* h_src, size_t size, cudaStream_t stream = 0);
    static bool CopyFromGPU(void* h_dst, const void* d_src, size_t size, cudaStream_t stream = 0);

private:
    struct BlockInfo {
        void* ptr = nullptr;
        size_t size = 0;
        bool in_use = false;
    };

    static size_t RoundUpPowerOfTwo(size_t size);

    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::vector<BlockInfo>> bin_pools_;
    std::unordered_map<void*, size_t> active_ptr_sizes_;
    PoolStatistics stats_;
};

} // namespace Cuda
} // namespace ConvNet

#endif // CONVNET_CUDA_MEMORY_POOL_H
