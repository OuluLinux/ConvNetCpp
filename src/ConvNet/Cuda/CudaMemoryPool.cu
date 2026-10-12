#include "CudaMemoryPool.h"
#include <iomanip>
#include <sstream>

namespace ConvNet {
namespace Cuda {

CudaMemoryPool& CudaMemoryPool::Instance() {
    static CudaMemoryPool pool;
    return pool;
}

CudaMemoryPool::CudaMemoryPool() = default;

CudaMemoryPool::~CudaMemoryPool() {
    Clear();
}

size_t CudaMemoryPool::RoundUpPowerOfTwo(size_t size) {
    if (size == 0) return 1024;
    size_t power = 1024; // 1 KB minimum bin
    while (power < size && power < (1ULL << 30)) { // Up to 1 GB
        power *= 2;
    }
    return (power >= size) ? power : size;
}

void* CudaMemoryPool::Allocate(size_t size) {
    if (size == 0) return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);
    stats_.total_alloc_requests++;

    size_t bin_size = RoundUpPowerOfTwo(size);
    auto& pool = bin_pools_[bin_size];

    // Check for an available block in this size bin
    for (auto& block : pool) {
        if (!block.in_use) {
            block.in_use = true;
            active_ptr_sizes_[block.ptr] = bin_size;
            stats_.cache_hits++;
            stats_.current_active_allocations++;
            stats_.current_active_bytes += bin_size;
            if (stats_.current_active_bytes > stats_.peak_allocated_bytes) {
                stats_.peak_allocated_bytes = stats_.current_active_bytes;
            }
            return block.ptr;
        }
    }

    // Cache miss: allocate new device memory
    stats_.cache_misses++;
    void* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, bin_size);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CudaMemoryPool: cudaMalloc failed for " << bin_size << " bytes: "
            << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }

    pool.push_back({d_ptr, bin_size, true});
    active_ptr_sizes_[d_ptr] = bin_size;
    stats_.current_active_allocations++;
    stats_.current_active_bytes += bin_size;
    stats_.current_pooled_bytes += bin_size;
    stats_.total_device_memory_allocated += bin_size;

    if (stats_.current_active_bytes > stats_.peak_allocated_bytes) {
        stats_.peak_allocated_bytes = stats_.current_active_bytes;
    }

    return d_ptr;
}

void CudaMemoryPool::Deallocate(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_ptr_sizes_.find(ptr);
    if (it == active_ptr_sizes_.end()) {
        // Unknown pointer, fallback to regular cudaFree
        cudaFree(ptr);
        return;
    }

    size_t bin_size = it->second;
    active_ptr_sizes_.erase(it);

    auto& pool = bin_pools_[bin_size];
    for (auto& block : pool) {
        if (block.ptr == ptr) {
            block.in_use = false;
            break;
        }
    }

    stats_.current_active_allocations--;
    stats_.current_active_bytes -= bin_size;
}

void CudaMemoryPool::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : bin_pools_) {
        for (auto& block : pair.second) {
            if (block.ptr) {
                cudaFree(block.ptr);
                block.ptr = nullptr;
            }
        }
    }
    bin_pools_.clear();
    active_ptr_sizes_.clear();
    stats_.current_active_allocations = 0;
    stats_.current_active_bytes = 0;
    stats_.current_pooled_bytes = 0;
}

void CudaMemoryPool::PreAllocate(size_t size, size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t bin_size = RoundUpPowerOfTwo(size);
    auto& pool = bin_pools_[bin_size];

    for (size_t i = 0; i < count; ++i) {
        void* d_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, bin_size));
        pool.push_back({d_ptr, bin_size, false});
        stats_.current_pooled_bytes += bin_size;
        stats_.total_device_memory_allocated += bin_size;
    }
}

PoolStatistics CudaMemoryPool::GetStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

std::string CudaMemoryPool::DumpStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << "=== CudaMemoryPool Statistics ===\n"
        << "Total requests: " << stats_.total_alloc_requests << "\n"
        << "Cache hits:     " << stats_.cache_hits << " (" << std::fixed << std::setprecision(1) << stats_.HitRate() << "% hit rate)\n"
        << "Cache misses:   " << stats_.cache_misses << "\n"
        << "Active blocks:  " << stats_.current_active_allocations << " (" << (stats_.current_active_bytes / 1024.0 / 1024.0) << " MB)\n"
        << "Pooled blocks:  " << (stats_.current_pooled_bytes / 1024.0 / 1024.0) << " MB total device memory\n"
        << "Peak memory:    " << (stats_.peak_allocated_bytes / 1024.0 / 1024.0) << " MB\n";
    return oss.str();
}

bool CudaMemoryPool::HasLeaks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !active_ptr_sizes_.empty();
}

bool CudaMemoryPool::CopyToGPU(void* d_dst, const void* h_src, size_t size, cudaStream_t stream) {
    if (!d_dst || !h_src || size == 0) return false;
    if (stream) {
        return cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }
    return cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool CudaMemoryPool::CopyFromGPU(void* h_dst, const void* d_src, size_t size, cudaStream_t stream) {
    if (!h_dst || !d_src || size == 0) return false;
    if (stream) {
        return cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    }
    return cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost) == cudaSuccess;
}

} // namespace Cuda
} // namespace ConvNet
