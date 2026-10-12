#ifndef CONVNET_CUDA_UTILS_H
#define CONVNET_CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <cmath>

namespace ConvNet {
namespace Cuda {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _cuda_err = (call); \
        if (_cuda_err != cudaSuccess) { \
            std::ostringstream _oss; \
            _oss << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                 << " in " << __FUNCTION__ << "(): " \
                 << cudaGetErrorString(_cuda_err) << " (code: " << _cuda_err << ")"; \
            throw std::runtime_error(_oss.str()); \
        } \
    } while (0)

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t _cuda_sync_err = cudaGetLastError(); \
        if (_cuda_sync_err != cudaSuccess) { \
            std::ostringstream _oss; \
            _oss << "CUDA Kernel Launch Error at " << __FILE__ << ":" << __LINE__ \
                 << ": " << cudaGetErrorString(_cuda_sync_err); \
            throw std::runtime_error(_oss.str()); \
        } \
    } while (0)

enum class ActivationType {
    None = 0,
    ReLU = 1,
    LeakyReLU = 2,
    Sigmoid = 3,
    Tanh = 4,
    GELU = 5
};

inline const char* ActivationTypeToString(ActivationType act) {
    switch (act) {
        case ActivationType::None: return "None";
        case ActivationType::ReLU: return "ReLU";
        case ActivationType::LeakyReLU: return "LeakyReLU";
        case ActivationType::Sigmoid: return "Sigmoid";
        case ActivationType::Tanh: return "Tanh";
        case ActivationType::GELU: return "GELU";
        default: return "Unknown";
    }
}

// RAII CUDA Stream
class CudaStream {
public:
    explicit CudaStream(unsigned int flags = cudaStreamNonBlocking) : stream_(nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    void Synchronize() const {
        if (stream_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }

    cudaStream_t Get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }

private:
    cudaStream_t stream_;
};

// RAII CUDA Event for precise GPU timing
class CudaEvent {
public:
    explicit CudaEvent(unsigned int flags = cudaEventDefault) : event_(nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }

    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
            event_ = nullptr;
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    void Record(cudaStream_t stream = 0) const {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void Synchronize() const {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    static float ElapsedTimeMs(const CudaEvent& start, const CudaEvent& stop) {
        stop.Synchronize();
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, stop.event_));
        return ms;
    }

    cudaEvent_t Get() const { return event_; }
    operator cudaEvent_t() const { return event_; }

private:
    cudaEvent_t event_;
};

// RAII Device Buffer
template <typename T>
class CudaDeviceBuffer {
public:
    CudaDeviceBuffer() : d_ptr_(nullptr), count_(0) {}

    explicit CudaDeviceBuffer(size_t count) : d_ptr_(nullptr), count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ptr_), count_ * sizeof(T)));
        }
    }

    ~CudaDeviceBuffer() {
        Free();
    }

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept : d_ptr_(other.d_ptr_), count_(other.count_) {
        other.d_ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
        if (this != &other) {
            Free();
            d_ptr_ = other.d_ptr_;
            count_ = other.count_;
            other.d_ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void Resize(size_t count) {
        if (count == count_) return;
        Free();
        count_ = count;
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ptr_), count_ * sizeof(T)));
        }
    }

    void CopyFromHost(const T* h_ptr, size_t count, cudaStream_t stream = 0) {
        if (count > count_) {
            Resize(count);
        }
        if (stream) {
            CUDA_CHECK(cudaMemcpyAsync(d_ptr_, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpy(d_ptr_, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void CopyToHost(T* h_ptr, size_t count, cudaStream_t stream = 0) const {
        if (count > count_) {
            throw std::runtime_error("CopyToHost: requested count exceeds buffer size");
        }
        if (stream) {
            CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        } else {
            CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    void Zero(cudaStream_t stream = 0) {
        if (count_ > 0 && d_ptr_) {
            if (stream) {
                CUDA_CHECK(cudaMemsetAsync(d_ptr_, 0, count_ * sizeof(T), stream));
            } else {
                CUDA_CHECK(cudaMemset(d_ptr_, 0, count_ * sizeof(T)));
            }
        }
    }

    T* Data() { return d_ptr_; }
    const T* Data() const { return d_ptr_; }
    size_t Count() const { return count_; }
    size_t Bytes() const { return count_ * sizeof(T); }

private:
    void Free() {
        if (d_ptr_) {
            cudaFree(d_ptr_);
            d_ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* d_ptr_;
    size_t count_;
};

// RAII Pinned Host Buffer
template <typename T>
class CudaPinnedBuffer {
public:
    CudaPinnedBuffer() : h_ptr_(nullptr), count_(0) {}

    explicit CudaPinnedBuffer(size_t count, unsigned int flags = cudaHostAllocDefault)
        : h_ptr_(nullptr), count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_ptr_), count_ * sizeof(T), flags));
        }
    }

    ~CudaPinnedBuffer() {
        Free();
    }

    CudaPinnedBuffer(const CudaPinnedBuffer&) = delete;
    CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;

    CudaPinnedBuffer(CudaPinnedBuffer&& other) noexcept : h_ptr_(other.h_ptr_), count_(other.count_) {
        other.h_ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaPinnedBuffer& operator=(CudaPinnedBuffer&& other) noexcept {
        if (this != &other) {
            Free();
            h_ptr_ = other.h_ptr_;
            count_ = other.count_;
            other.h_ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void Resize(size_t count) {
        if (count == count_) return;
        Free();
        count_ = count;
        if (count_ > 0) {
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_ptr_), count_ * sizeof(T), cudaHostAllocDefault));
        }
    }

    T* Data() { return h_ptr_; }
    const T* Data() const { return h_ptr_; }
    size_t Count() const { return count_; }
    size_t Bytes() const { return count_ * sizeof(T); }

    T& operator[](size_t idx) { return h_ptr_[idx]; }
    const T& operator[](size_t idx) const { return h_ptr_[idx]; }

private:
    void Free() {
        if (h_ptr_) {
            cudaFreeHost(h_ptr_);
            h_ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* h_ptr_;
    size_t count_;
};

// Device properties helper
struct DeviceProperties {
    std::string name;
    int major = 0;
    int minor = 0;
    size_t total_memory = 0;
    int multi_processor_count = 0;
    int max_threads_per_block = 0;
    int clock_rate_khz = 0;
    int memory_bus_width = 0;
    double memory_bandwidth_gbps = 0.0;
    double peak_fp32_tflops = 0.0;
};

inline DeviceProperties QueryDeviceProperties(int device_id = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    DeviceProperties dp;
    dp.name = prop.name;
    dp.major = prop.major;
    dp.minor = prop.minor;
    dp.total_memory = prop.totalGlobalMem;
    dp.multi_processor_count = prop.multiProcessorCount;
    dp.max_threads_per_block = prop.maxThreadsPerBlock;
    dp.clock_rate_khz = prop.clockRate;
    dp.memory_bus_width = prop.memoryBusWidth;
    
    // Memory bandwidth: 2 * memoryClockRate * busWidth / 8 / 1e6 GB/s (for DDR/GDDR5)
    double mem_clock_hz = static_cast<double>(prop.memoryClockRate) * 1e3;
    dp.memory_bandwidth_gbps = (2.0 * mem_clock_hz * (prop.memoryBusWidth / 8.0)) / 1e9;

    // Peak FP32 TFLOPS: SM count * cores per SM * clock * 2 FLOPs per FMA
    int cores_per_sm = 128; // Pascal sm_61 has 128 FP32 cores per SM
    if (prop.major == 7) cores_per_sm = 64; // Volta / Turing
    else if (prop.major == 8) cores_per_sm = 128; // Ampere
    else if (prop.major == 9) cores_per_sm = 128; // Hopper
    
    double clock_ghz = static_cast<double>(prop.clockRate) / 1e6;
    dp.peak_fp32_tflops = (dp.multi_processor_count * cores_per_sm * clock_ghz * 2.0) / 1000.0;

    return dp;
}

} // namespace Cuda
} // namespace ConvNet

#endif // CONVNET_CUDA_UTILS_H
