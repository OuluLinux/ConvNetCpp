#include "ConvNet.h"
#include "MemoryPool.h"
#include <cstdlib>
#include <cstring>

namespace ConvNet {

// Define static member
const std::vector<size_t> MemoryPool::DEFAULT_SIZES = {
    1024,       // 1KB
    2048,       // 2KB
    4096,       // 4KB
    8192,       // 8KB
    16384,      // 16KB
    32768,      // 32KB
    65536,      // 64KB
    131072,     // 128KB
    262144,     // 256KB
    524288,     // 512KB
    1048576,    // 1MB
    2097152,    // 2MB
    4194304,    // 4MB
    8388608,    // 8MB
    16777216    // 16MB
};

thread_local std::unique_ptr<MemoryPool> ThreadLocalMemoryPool::thread_pool = nullptr;

MemoryPool::MemoryPool() {
    // Pre-allocate default size pools
    for (size_t size : DEFAULT_SIZES) {
        // Start with a small number of blocks per size class
        PreAllocate(size, 4);
    }
}

MemoryPool::~MemoryPool() {
    Clear(); // Clean up all allocated memory
}

void* MemoryPool::AllocateFromPool(size_t size) {
    // Find the next power of 2 that's >= size
    size_t pool_size = 1024; // Start from 1KB
    while (pool_size < size && pool_size <= 16777216) { // Don't exceed 16MB
        pool_size *= 2;
    }
    
    if (pool_size < size) {
        // Size is too large, allocate directly
        void* ptr = malloc(size);
        if (ptr) {
            allocated_blocks[ptr] = size;
        }
        return ptr;
    }
    
    auto& pool = size_pools[pool_size];
    
    // Look for a free block
    for (auto& block : pool) {
        if (!block.used) {
            block.used = true;
            allocated_blocks[block.ptr] = pool_size;
            return block.ptr;
        }
    }
    
    // No free blocks, allocate a new one
    void* new_ptr = malloc(pool_size);
    if (new_ptr) {
        pool.emplace_back(new_ptr, pool_size);
        auto& block = pool.back();
        block.used = true;
        allocated_blocks[block.ptr] = pool_size;
        return block.ptr;
    }
    
    return nullptr; // Allocation failed
}

void MemoryPool::DeallocateToPool(void* ptr, size_t size) {
    // Find the appropriate pool size
    auto it = size_pools.find(size);
    if (it != size_pools.end()) {
        auto& pool = it->second;
        for (auto& block : pool) {
            if (block.ptr == ptr) {
                block.used = false;
                allocated_blocks.erase(ptr);
                return;
            }
        }
    }
    
    // If not found in pooled blocks, it was allocated directly, so free it
    if (allocated_blocks.count(ptr)) {
        free(ptr);
        allocated_blocks.erase(ptr);
    }
}

void* MemoryPool::Allocate(size_t size) {
    if (size == 0) return nullptr;
    
    lock.Enter();
    void* ptr = AllocateFromPool(size);
    lock.Leave();
    
    return ptr;
}

void MemoryPool::Deallocate(void* ptr) {
    if (!ptr) return;
    
    lock.Enter();
    auto it = allocated_blocks.find(ptr);
    if (it != allocated_blocks.end()) {
        size_t size = it->second;
        DeallocateToPool(ptr, size);
    }
    lock.Leave();
}

void MemoryPool::Clear() {
    lock.Enter();
    
    // Free all allocated blocks
    for (const auto& pair : allocated_blocks) {
        free(pair.first);
    }
    allocated_blocks.clear();
    
    // Free all pool blocks
    for (auto& size_pool : size_pools) {
        for (auto& block : size_pool.second) {
            if (block.ptr) {
                free(block.ptr);
            }
        }
    }
    size_pools.clear();
    
    lock.Leave();
}

size_t MemoryPool::GetAllocatedMemory() const {
    size_t total = 0;
    for (const auto& pair : allocated_blocks) {
        total += pair.second;
    }
    return total;
}

size_t MemoryPool::GetPooledMemory() const {
    size_t total = 0;
    for (const auto& size_pool : size_pools) {
        for (const auto& block : size_pool.second) {
            if (!allocated_blocks.count(block.ptr)) { // Not currently allocated
                total += block.size;
            }
        }
    }
    return total;
}

size_t MemoryPool::GetTotalMemoryUsed() const {
    return GetAllocatedMemory() + GetPooledMemory();
}

void MemoryPool::PreAllocate(size_t size, int count) {
    lock.Enter();
    auto& pool = size_pools[size];
    
    for (int i = 0; i < count; i++) {
        void* ptr = malloc(size);
        if (ptr) {
            pool.emplace_back(ptr, size);
        }
    }
    
    lock.Leave();
}

String MemoryPool::GetMemoryUsageInfo() const {
    String info;
    info << "Memory Pool Usage:\n";
    info << "Allocated Memory: " << GetAllocatedMemory() << " bytes\n";
    info << "Pooled Memory: " << GetPooledMemory() << " bytes\n";
    info << "Total Memory: " << GetTotalMemoryUsed() << " bytes\n";
    info << "Number of Size Classes: " << size_pools.size() << "\n";
    
    // Add breakdown by size class
    info << "Size Classes:\n";
    for (const auto& pair : size_pools) {
        size_t active_blocks = 0;
        for (const auto& block : pair.second) {
            if (block.used) active_blocks++;
        }
        info << "  " << pair.first << " bytes: " 
             << pair.second.size() << " total, " 
             << active_blocks << " active, " 
             << (pair.second.size() - active_blocks) << " free\n";
    }
    
    return info;
}

MemoryPool& ThreadLocalMemoryPool::Get() {
    if (!thread_pool) {
        thread_pool = std::make_unique<MemoryPool>();
    }
    return *thread_pool;
}

void ThreadLocalMemoryPool::Reset() {
    thread_pool.reset();
}

} // namespace ConvNet