#ifndef _ConvNet_MemoryPool_h_
#define _ConvNet_MemoryPool_h_

#include "ConvNet.h"
#include <unordered_map>
#include <memory>

namespace ConvNet {

// Memory pool for efficient allocation and deallocation of neural network tensors
class MemoryPool {
private:
    struct PoolBlock {
        void* ptr;
        size_t size;
        bool used;
        
        PoolBlock(void* p, size_t s) : ptr(p), size(s), used(false) {}
    };
    
    // Memory blocks for different sizes (powers of 2 for efficient allocation)
    std::unordered_map<size_t, std::vector<PoolBlock>> size_pools;
    std::unordered_map<void*, size_t> allocated_blocks; // Track currently allocated blocks
    std::unordered_map<void*, size_t> block_sizes;      // Size of each allocated block
    SpinLock lock;
    
    // Default pool sizes - powers of 2 from 1KB to 16MB
    static const std::vector<size_t> DEFAULT_SIZES;
    
    // Try to allocate from existing blocks of the same size class
    void* AllocateFromPool(size_t size);
    void DeallocateToPool(void* ptr, size_t size);
    
public:
    MemoryPool();
    ~MemoryPool();
    
    // Allocate memory of specified size
    void* Allocate(size_t size);
    
    // Deallocate memory back to pool
    void Deallocate(void* ptr);
    
    // Clear all pools (free memory)
    void Clear();
    
    // Get memory statistics
    size_t GetAllocatedMemory() const;
    size_t GetPooledMemory() const;
    size_t GetTotalMemoryUsed() const;
    
    // Pre-allocate a specific size pool if needed
    void PreAllocate(size_t size, int count);
    
    // Get memory usage details
    String GetMemoryUsageInfo() const;
};

// Thread-local memory pool for better performance
class ThreadLocalMemoryPool {
private:
    static thread_local std::unique_ptr<MemoryPool> thread_pool;
    
public:
    static MemoryPool& Get();
    static void Reset();
};

// Smart pointer that automatically manages memory from the pool
template<typename T>
class PooledPtr {
private:
    T* ptr;
    size_t count;
    
public:
    PooledPtr() : ptr(nullptr), count(0) {}
    
    explicit PooledPtr(size_t n) : count(n) {
        ptr = static_cast<T*>(ThreadLocalMemoryPool::Get().Allocate(sizeof(T) * n));
        for (size_t i = 0; i < n; i++) {
            new(&ptr[i]) T(); // Placement new to construct objects
        }
    }
    
    ~PooledPtr() {
        if (ptr) {
            // Destruct objects
            for (size_t i = 0; i < count; i++) {
                ptr[i].~T();
            }
            ThreadLocalMemoryPool::Get().Deallocate(ptr);
            ptr = nullptr;
        }
    }
    
    // Move constructor
    PooledPtr(PooledPtr&& other) noexcept : ptr(other.ptr), count(other.count) {
        other.ptr = nullptr;
        other.count = 0;
    }
    
    // Move assignment
    PooledPtr& operator=(PooledPtr&& other) noexcept {
        if (this != &other) {
            // Clean up current resources
            if (ptr) {
                for (size_t i = 0; i < count; i++) {
                    ptr[i].~T();
                }
                ThreadLocalMemoryPool::Get().Deallocate(ptr);
            }
            // Transfer ownership
            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }
    
    // Disable copying
    PooledPtr(const PooledPtr&) = delete;
    PooledPtr& operator=(const PooledPtr&) = delete;
    
    T* Get() { return ptr; }
    const T* Get() const { return ptr; }
    
    T& operator[](size_t index) { 
        ASSERT(index < count);
        return ptr[index]; 
    }
    
    const T& operator[](size_t index) const { 
        ASSERT(index < count);
        return ptr[index]; 
    }
    
    size_t GetCount() const { return count; }
};

// Wrapper for U++ Vector<double> that uses memory pool
class PooledVector {
private:
    double* data;
    int count;
    int allocated_count;
    
public:
    PooledVector() : data(nullptr), count(0), allocated_count(0) {}
    
    explicit PooledVector(int n, double init_value = 0.0) : count(n), allocated_count(n) {
        data = static_cast<double*>(ThreadLocalMemoryPool::Get().Allocate(sizeof(double) * n));
        for (int i = 0; i < n; i++) {
            data[i] = init_value;
        }
    }
    
    ~PooledVector() {
        if (data) {
            ThreadLocalMemoryPool::Get().Deallocate(data);
            data = nullptr;
        }
    }
    
    // Move constructor
    PooledVector(PooledVector&& other) noexcept 
        : data(other.data), count(other.count), allocated_count(other.allocated_count) {
        other.data = nullptr;
        other.count = 0;
        other.allocated_count = 0;
    }
    
    // Move assignment
    PooledVector& operator=(PooledVector&& other) noexcept {
        if (this != &other) {
            if (data) {
                ThreadLocalMemoryPool::Get().Deallocate(data);
            }
            data = other.data;
            count = other.count;
            allocated_count = other.allocated_count;
            other.data = nullptr;
            other.count = 0;
            other.allocated_count = 0;
        }
        return *this;
    }
    
    // Disable copying for now to avoid complexity
    PooledVector(const PooledVector&) = delete;
    PooledVector& operator=(const PooledVector&) = delete;
    
    double& operator[](int i) { 
        ASSERT(i >= 0 && i < count);
        return data[i]; 
    }
    
    const double& operator[](int i) const { 
        ASSERT(i >= 0 && i < count);
        return data[i]; 
    }
    
    double Get(int i) const {
        if (i >= 0 && i < count) return data[i];
        return 0.0;
    }
    
    void Set(int i, double value) {
        if (i >= 0 && i < count) data[i] = value;
    }
    
    int GetCount() const { return count; }
    
    void SetCount(int new_count, double init_value = 0.0) {
        if (new_count <= allocated_count) {
            // We have enough allocated space
            count = new_count;
            // Initialize new elements if needed
            for (int i = count; i < new_count; i++) {
                data[i] = init_value;
            }
        } else {
            // Need to reallocate
            double* new_data = static_cast<double*>(ThreadLocalMemoryPool::Get().Allocate(sizeof(double) * new_count));
            // Copy existing data
            for (int i = 0; i < count && i < new_count; i++) {
                new_data[i] = data[i];
            }
            // Initialize new elements
            for (int i = count; i < new_count; i++) {
                new_data[i] = init_value;
            }
            // Deallocate old data
            if (data) {
                ThreadLocalMemoryPool::Get().Deallocate(data);
            }
            data = new_data;
            count = new_count;
            allocated_count = new_count;
        }
    }
    
    double* Begin() { return data; }
    const double* Begin() const { return data; }
};

// Memory pool compatible Mat class
class PoolMat {
private:
    PooledVector weights;
    PooledVector weight_gradients;
    int width;
    int height;
    int length;

public:
    PoolMat() : width(0), height(0), length(0) {}

    PoolMat(int width, int height) : width(0), height(0), length(0) {
        Init(width, height);
    }

    PoolMat(int width, int height, double default_value) : width(0), height(0), length(0) {
        Init(width, height, default_value);
    }

    PoolMat(const Vector<double>& src_weights) : width(0), height(0), length(0) {
        width = 1;
        height = src_weights.GetCount();
        length = height;
        weights.SetCount(length);
        
        for (int i = 0; i < length; i++) {
            weights.Set(i, src_weights[i]);
        }
        
        weight_gradients.SetCount(length, 0.0);
    }

    PoolMat(int width, int height, const Vector<double>& src_weights) : width(width), height(height) {
        length = width * height;
        ASSERT(length == src_weights.GetCount());

        weights.SetCount(length);
        for (int i = 0; i < length; i++) {
            weights.Set(i, src_weights[i]);
        }
        
        weight_gradients.SetCount(length, 0.0);
    }

    PoolMat& Init(int width, int height) {
        ASSERT(width > 0 && height > 0);
        
        this->width = width;
        this->height = height;
        int n = width * height;
        length = n;
        
        weights.SetCount(n, 0.0);
        weight_gradients.SetCount(n, 0.0);

        RandomGaussian& rand = GetRandomGaussian(length);

        for (int i = 0; i < n; i++) {
            weights.Set(i, rand);
        }

        return *this;
    }

    PoolMat& Init(int width, int height, double default_value) {
        ASSERT(width > 0 && height > 0);

        this->width = width;
        this->height = height;
        int n = width * height;
        length = n;
        
        weights.SetCount(n, default_value);
        weight_gradients.SetCount(n, 0.0);

        return *this;
    }

    int GetPos(int x, int y) const {
        ASSERT(x >= 0 && y >= 0 && x < width && y < height);
        return (width * y) + x;
    }
    
    double Get(int x, int y) const {
        int ix = GetPos(x,y);
        return weights.Get(ix);
    }
    
    void Set(int x, int y, double v) {
        int ix = GetPos(x,y);
        weights.Set(ix, v);
    }
    
    void Add(int x, int y, double v) {
        int ix = GetPos(x,y);
        weights.Set(ix, weights.Get(ix) + v);
    }
    
    void AddGradient(int x, int y, double v) {
        int ix = GetPos(x,y);
        weight_gradients.Set(ix, weight_gradients.Get(ix) + v);
    }
    
    double GetGradient(int x, int y) const {
        int ix = GetPos(x,y);
        return weight_gradients.Get(ix);
    }
    
    double Get(int i) const {
        return weights.Get(i);
    }
    
    void Set(int i, double v) {
        weights.Set(i, v);
    }
    
    void Add(int i, double v) {
        weights.Set(i, weights.Get(i) + v);
    }
    
    double GetGradient(int i) const {
        return weight_gradients.Get(i);
    }
    
    void SetGradient(int i, double v) {
        weight_gradients.Set(i, v);
    }
    
    void AddGradient(int i, double v) {
        weight_gradients.Set(i, weight_gradients.Get(i) + v);
    }
    
    void ZeroGradients() {
        for (int i = 0; i < weight_gradients.GetCount(); i++) {
            weight_gradients.Set(i, 0.0);
        }
    }
    
    void AddFrom(const PoolMat& volume) {
        for (int i = 0; i < weights.GetCount() && i < volume.weights.GetCount(); i++) {
            weights.Set(i, weights.Get(i) + volume.Get(i));
        }
    }
    
    void AddGradientFrom(const PoolMat& volume) {
        for (int i = 0; i < weight_gradients.GetCount() && i < volume.weight_gradients.GetCount(); i++) {
            weight_gradients.Set(i, weight_gradients.Get(i) + volume.GetGradient(i));
        }
    }
    
    void AddFromScaled(const PoolMat& volume, double a) {
        for (int i = 0; i < weights.GetCount() && i < volume.weights.GetCount(); i++) {
            weights.Set(i, weights.Get(i) + a * volume.Get(i));
        }
    }
    
    void SetConst(double c) {
        for (int i = 0; i < weights.GetCount(); i++) {
            weights.Set(i, c);
        }
    }
    
    void SetConstGradient(double c) {
        for (int i = 0; i < weight_gradients.GetCount(); i++) {
            weight_gradients.Set(i, c);
        }
    }

    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
    int GetLength() const { return length; }
};

// Memory pool compatible Volume class (3D tensor)
class PoolVolume {
private:
    PooledVector weights;
    PooledVector weight_gradients;
    int width;
    int height;
    int depth;
    int length;

public:
    PoolVolume() : width(0), height(0), depth(0), length(0) {}

    PoolVolume(int width, int height, int depth) : width(0), height(0), depth(0), length(0) {
        Init(width, height, depth);
    }

    PoolVolume(int width, int height, int depth, double default_value) : width(0), height(0), depth(0), length(0) {
        Init(width, height, depth, default_value);
    }

    PoolVolume(const Vector<double>& src_weights) : width(1), height(1), depth(src_weights.GetCount()), length(src_weights.GetCount()) {
        weights.SetCount(length);
        
        for (int i = 0; i < length; i++) {
            weights.Set(i, src_weights[i]);
        }
        
        weight_gradients.SetCount(length, 0.0);
    }

    PoolVolume(int width, int height, int depth, const Vector<double>& src_weights) 
        : width(width), height(height), depth(depth), length(width * height * depth) {
        ASSERT(length == src_weights.GetCount());

        weights.SetCount(length);
        for (int i = 0; i < length; i++) {
            weights.Set(i, src_weights[i]);
        }
        
        weight_gradients.SetCount(length, 0.0);
    }

    PoolVolume& Init(int width, int height, int depth) {
        ASSERT(width > 0 && height > 0 && depth > 0);
        
        this->width = width;
        this->height = height;
        this->depth = depth;
        int n = width * height * depth;
        length = n;
        
        weights.SetCount(n, 0.0);
        weight_gradients.SetCount(n, 0.0);

        RandomGaussian& rand = GetRandomGaussian(length);

        for (int i = 0; i < n; i++) {
            weights.Set(i, rand);
        }

        return *this;
    }

    PoolVolume& Init(int width, int height, int depth, double default_value) {
        ASSERT(width > 0 && height > 0 && depth > 0);

        this->width = width;
        this->height = height;
        this->depth = depth;
        int n = width * height * depth;
        length = n;
        
        weights.SetCount(n, default_value);
        weight_gradients.SetCount(n, 0.0);

        return *this;
    }

    int GetPos(int x, int y, int d) const {
        ASSERT(x >= 0 && y >= 0 && d >= 0 && x < width && y < height && d < depth);
        return ((width * y) + x) * depth + d;
    }
    
    double Get(int x, int y, int d) const {
        int ix = GetPos(x, y, d);
        return weights.Get(ix);
    }
    
    void Set(int x, int y, int d, double v) {
        int ix = GetPos(x, y, d);
        weights.Set(ix, v);
    }
    
    void Add(int x, int y, int d, double v) {
        int ix = GetPos(x, y, d);
        weights.Set(ix, weights.Get(ix) + v);
    }
    
    void AddGradient(int x, int y, int d, double v) {
        int ix = GetPos(x, y, d);
        weight_gradients.Set(ix, weight_gradients.Get(ix) + v);
    }
    
    double GetGradient(int x, int y, int d) const {
        int ix = GetPos(x, y, d);
        return weight_gradients.Get(ix);
    }
    
    double Get(int i) const {
        return weights.Get(i);
    }
    
    void Set(int i, double v) {
        weights.Set(i, v);
    }
    
    void Add(int i, double v) {
        weights.Set(i, weights.Get(i) + v);
    }
    
    double GetGradient(int i) const {
        return weight_gradients.Get(i);
    }
    
    void SetGradient(int i, double v) {
        weight_gradients.Set(i, v);
    }
    
    void AddGradient(int i, double v) {
        weight_gradients.Set(i, weight_gradients.Get(i) + v);
    }
    
    void ZeroGradients() {
        for (int i = 0; i < weight_gradients.GetCount(); i++) {
            weight_gradients.Set(i, 0.0);
        }
    }
    
    void AddFrom(const PoolVolume& volume) {
        for (int i = 0; i < weights.GetCount() && i < volume.weights.GetCount(); i++) {
            weights.Set(i, weights.Get(i) + volume.Get(i));
        }
    }
    
    void AddGradientFrom(const PoolVolume& volume) {
        for (int i = 0; i < weight_gradients.GetCount() && i < volume.weight_gradients.GetCount(); i++) {
            weight_gradients.Set(i, weight_gradients.Get(i) + volume.GetGradient(i));
        }
    }
    
    void AddFromScaled(const PoolVolume& volume, double a) {
        for (int i = 0; i < weights.GetCount() && i < volume.weights.GetCount(); i++) {
            weights.Set(i, weights.Get(i) + a * volume.Get(i));
        }
    }
    
    void SetConst(double c) {
        for (int i = 0; i < weights.GetCount(); i++) {
            weights.Set(i, c);
        }
    }
    
    void SetConstGradient(double c) {
        for (int i = 0; i < weight_gradients.GetCount(); i++) {
            weight_gradients.Set(i, c);
        }
    }

    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
    int GetDepth() const { return depth; }
    int GetLength() const { return length; }
};

#ifdef flagGPU // Define this flag to enable GPU support

// GPU memory pool for efficient allocation and deallocation of GPU tensors
class GPUMemoryPool {
private:
    struct GPUChunk {
        void* ptr;
        size_t size;
        bool used;

        GPUChunk(void* p, size_t s) : ptr(p), size(s), used(false) {}
    };

    // Memory blocks for different sizes (powers of 2 for efficient allocation)
    std::unordered_map<size_t, std::vector<GPUChunk>> size_pools;
    std::unordered_map<void*, size_t> allocated_blocks; // Track currently allocated blocks
    SpinLock lock;

    // Default pool sizes - powers of 2 from 1KB to 16MB
    static const std::vector<size_t> DEFAULT_SIZES;

public:
    GPUMemoryPool();
    ~GPUMemoryPool();

    // Allocate GPU memory of specified size
    void* Allocate(size_t size);

    // Deallocate GPU memory back to pool
    void Deallocate(void* ptr);

    // Clear all pools (free GPU memory)
    void Clear();

    // Get memory statistics
    size_t GetAllocatedMemory() const;
    size_t GetPooledMemory() const;
    size_t GetTotalMemoryUsed() const;

    // Pre-allocate a specific size pool if needed
    void PreAllocate(size_t size, int count);

    // Get memory usage details
    String GetMemoryUsageInfo() const;

    // Copy data from host to device
    static bool CopyToGPU(void* gpu_ptr, const void* host_ptr, size_t size);

    // Copy data from device to host
    static bool CopyFromGPU(void* host_ptr, const void* gpu_ptr, size_t size);
    
    // Static access method for singleton or thread-local pool
    static GPUMemoryPool& Get();
};

// GPU-compatible Matrix class
class GPUMat {
private:
    void* gpu_weights;      // Pointer to GPU memory
    void* gpu_gradients;    // Pointer to GPU memory for gradients
    int width;
    int height;
    int length;

public:
    GPUMat() : gpu_weights(nullptr), gpu_gradients(nullptr), width(0), height(0), length(0) {}

    GPUMat(int width, int height) : gpu_weights(nullptr), gpu_gradients(nullptr), width(0), height(0), length(0) {
        Init(width, height);
    }

    GPUMat(int width, int height, double default_value) : gpu_weights(nullptr), gpu_gradients(nullptr), width(0), height(0), length(0) {
        Init(width, height, default_value);
    }

    GPUMat(const Vector<double>& src_weights) : gpu_weights(nullptr), gpu_gradients(nullptr), width(1), height(src_weights.GetCount()), length(src_weights.GetCount()) {
        Init(width, height);
        CopyFromHost(src_weights, 0.0);
    }

    GPUMat(const PoolMat& pool_mat) : gpu_weights(nullptr), gpu_gradients(nullptr), width(pool_mat.GetWidth()), height(pool_mat.GetHeight()), length(pool_mat.GetLength()) {
        Init(width, height);
        CopyFromHost(pool_mat);
    }

    ~GPUMat() {
        if (gpu_weights) {
            GPUMemoryPool::Get().Deallocate(gpu_weights);
            gpu_weights = nullptr;
        }
        if (gpu_gradients) {
            GPUMemoryPool::Get().Deallocate(gpu_gradients);
            gpu_gradients = nullptr;
        }
    }

    GPUMat& Init(int width, int height) {
        ASSERT(width > 0 && height > 0);

        if (gpu_weights) {
            GPUMemoryPool::Get().Deallocate(gpu_weights);
            gpu_weights = nullptr;
        }
        if (gpu_gradients) {
            GPUMemoryPool::Get().Deallocate(gpu_gradients);
            gpu_gradients = nullptr;
        }

        this->width = width;
        this->height = height;
        length = width * height;

        gpu_weights = GPUMemoryPool::Get().Allocate(sizeof(double) * length);
        gpu_gradients = GPUMemoryPool::Get().Allocate(sizeof(double) * length);

        ZeroWeights();
        ZeroGradients();

        return *this;
    }

    GPUMat& Init(int width, int height, double default_value) {
        Init(width, height);
        SetConst(default_value);
        return *this;
    }

    void ZeroWeights() {
        // Initialize GPU weights to zero
        Vector<double> zeros(length, 0.0);
        CopyFromHost(zeros, 0.0);
    }

    void ZeroGradients() {
        // Initialize GPU gradients to zero
        Vector<double> zeros(length, 0.0);
        CopyFromHostGradients(zeros);
    }

    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
    int GetLength() const { return length; }

    // Copy data from host Vector to GPU
    bool CopyFromHost(const Vector<double>& host_data, double default_value = 0.0) {
        if (host_data.GetCount() != length) return false;

        return GPUMemoryPool::CopyToGPU(gpu_weights, host_data.Begin(), sizeof(double) * length);
    }

    // Copy data from another GPUMat to this one
    bool CopyFromHost(const GPUMat& other) {
        if (other.GetLength() != length) return false;

        bool success = GPUMemoryPool::CopyToGPU(gpu_weights, other.gpu_weights, sizeof(double) * length);
        if (success) {
            success = GPUMemoryPool::CopyToGPU(gpu_gradients, other.gpu_gradients, sizeof(double) * length);
        }
        return success;
    }

    // Copy data from PoolMat to GPU
    bool CopyFromHost(const PoolMat& pool_mat) {
        if (pool_mat.GetLength() != length) return false;

        // Create temporary host vector
        Vector<double> host_data;
        host_data.SetCount(length);
        for (int i = 0; i < length; i++) {
            host_data[i] = pool_mat.Get(i);
        }

        return CopyFromHost(host_data, 0.0);
    }

    // Copy gradients from host Vector to GPU
    bool CopyFromHostGradients(const Vector<double>& host_gradients) {
        if (host_gradients.GetCount() != length) return false;

        return GPUMemoryPool::CopyToGPU(gpu_gradients, host_gradients.Begin(), sizeof(double) * length);
    }

    // Copy data from GPU to host Vector
    bool CopyToHost(Vector<double>& host_data) const {
        host_data.SetCount(length);
        return GPUMemoryPool::CopyFromGPU(host_data.Begin(), gpu_weights, sizeof(double) * length);
    }

    // Copy gradients from GPU to host Vector
    bool CopyGradientsToHost(Vector<double>& host_gradients) const {
        host_gradients.SetCount(length);
        return GPUMemoryPool::CopyFromGPU(host_gradients.Begin(), gpu_gradients, sizeof(double) * length);
    }

    void SetConst(double c);
    void AddFrom(const GPUMat& mat);
    void AddFromScaled(const GPUMat& mat, double scale);
    void AddGradientFrom(const GPUMat& mat);
    void AddGradientFromScaled(const GPUMat& mat, double scale);
};

// GPU-compatible Volume class (3D tensor)
class GPUVolume {
private:
    void* gpu_weights;      // Pointer to GPU memory
    void* gpu_gradients;    // Pointer to GPU memory for gradients
    int width;
    int height;
    int depth;
    int length;

public:
    GPUVolume() : gpu_weights(nullptr), gpu_gradients(nullptr), width(0), height(0), depth(0), length(0) {}

    GPUVolume(int width, int height, int depth) : gpu_weights(nullptr), gpu_gradients(nullptr), width(0), height(0), depth(0), length(0) {
        Init(width, height, depth);
    }

    GPUVolume(int width, int height, int depth, double default_value) : gpu_weights(nullptr), gpu_gradients(nullptr), width(0), height(0), depth(0), length(0) {
        Init(width, height, depth, default_value);
    }

    GPUVolume(const Vector<double>& src_weights) : gpu_weights(nullptr), gpu_gradients(nullptr), width(1), height(1), depth(src_weights.GetCount()), length(src_weights.GetCount()) {
        Init(width, height, depth);
        CopyFromHost(src_weights, 0.0);
    }

    GPUVolume(const PoolVolume& pool_volume) : gpu_weights(nullptr), gpu_gradients(nullptr), width(pool_volume.GetWidth()), height(pool_volume.GetHeight()), depth(pool_volume.GetDepth()), length(pool_volume.GetLength()) {
        Init(width, height, depth);
        CopyFromHost(pool_volume);
    }

    ~GPUVolume() {
        if (gpu_weights) {
            GPUMemoryPool::Get().Deallocate(gpu_weights);
            gpu_weights = nullptr;
        }
        if (gpu_gradients) {
            GPUMemoryPool::Get().Deallocate(gpu_gradients);
            gpu_gradients = nullptr;
        }
    }

    GPUVolume& Init(int width, int height, int depth) {
        ASSERT(width > 0 && height > 0 && depth > 0);

        if (gpu_weights) {
            GPUMemoryPool::Get().Deallocate(gpu_weights);
            gpu_weights = nullptr;
        }
        if (gpu_gradients) {
            GPUMemoryPool::Get().Deallocate(gpu_gradients);
            gpu_gradients = nullptr;
        }

        this->width = width;
        this->height = height;
        this->depth = depth;
        length = width * height * depth;

        gpu_weights = GPUMemoryPool::Get().Allocate(sizeof(double) * length);
        gpu_gradients = GPUMemoryPool::Get().Allocate(sizeof(double) * length);

        ZeroWeights();
        ZeroGradients();

        return *this;
    }

    GPUVolume& Init(int width, int height, int depth, double default_value) {
        Init(width, height, depth);
        SetConst(default_value);
        return *this;
    }

    void ZeroWeights() {
        // Initialize GPU weights to zero
        Vector<double> zeros(length, 0.0);
        CopyFromHost(zeros, 0.0);
    }

    void ZeroGradients() {
        // Initialize GPU gradients to zero
        Vector<double> zeros(length, 0.0);
        CopyFromHostGradients(zeros);
    }

    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
    int GetDepth() const { return depth; }
    int GetLength() const { return length; }

    // Copy data from host Vector to GPU
    bool CopyFromHost(const Vector<double>& host_data, double default_value = 0.0) {
        if (host_data.GetCount() != length) return false;

        return GPUMemoryPool::CopyToGPU(gpu_weights, host_data.Begin(), sizeof(double) * length);
    }

    // Copy data from another GPUVolume to this one
    bool CopyFromHost(const GPUVolume& other) {
        if (other.GetLength() != length) return false;

        bool success = GPUMemoryPool::CopyToGPU(gpu_weights, other.gpu_weights, sizeof(double) * length);
        if (success) {
            success = GPUMemoryPool::CopyToGPU(gpu_gradients, other.gpu_gradients, sizeof(double) * length);
        }
        return success;
    }

    // Copy data from PoolVolume to GPU
    bool CopyFromHost(const PoolVolume& pool_volume) {
        if (pool_volume.GetLength() != length) return false;

        // Create temporary host vector
        Vector<double> host_data;
        host_data.SetCount(length);
        for (int i = 0; i < length; i++) {
            host_data[i] = pool_volume.Get(i);
        }

        return CopyFromHost(host_data, 0.0);
    }

    // Copy gradients from host Vector to GPU
    bool CopyFromHostGradients(const Vector<double>& host_gradients) {
        if (host_gradients.GetCount() != length) return false;

        return GPUMemoryPool::CopyToGPU(gpu_gradients, host_gradients.Begin(), sizeof(double) * length);
    }

    // Copy data from GPU to host Vector
    bool CopyToHost(Vector<double>& host_data) const {
        host_data.SetCount(length);
        return GPUMemoryPool::CopyFromGPU(host_data.Begin(), gpu_weights, sizeof(double) * length);
    }

    // Copy gradients from GPU to host Vector
    bool CopyGradientsToHost(Vector<double>& host_gradients) const {
        host_gradients.SetCount(length);
        return GPUMemoryPool::CopyFromGPU(host_gradients.Begin(), gpu_gradients, sizeof(double) * length);
    }

    void SetConst(double c);
    void AddFrom(const GPUVolume& volume);
    void AddFromScaled(const GPUVolume& volume, double scale);
    void AddGradientFrom(const GPUVolume& volume);
    void AddGradientFromScaled(const GPUVolume& volume, double scale);
};

// Thread-local GPU memory pool
class ThreadLocalGPUMemoryPool {
private:
    static thread_local std::unique_ptr<GPUMemoryPool> thread_pool;

public:
    static GPUMemoryPool& Get();
    static void Reset();
};

#endif // flagGPU

} // namespace ConvNet

#endif