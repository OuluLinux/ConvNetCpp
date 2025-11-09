#ifndef _ConvNet_ParallellaSupport_h_
#define _ConvNet_ParallellaSupport_h_

#include "ConvNet.h"

#ifdef PARALLELLA_SUPPORT
#include <e-hal.h>
#endif

namespace ConvNet {

// Forward declarations for Parallella-related classes
class ParallellaDevice;
class ParallellaMemory;
class ParallellaKernel;

#ifdef PARALLELLA_SUPPORT
// Parallella/Epiphany specific memory management
class ParallellaMemory {
private:
    void* host_addr;
    void* device_addr;
    size_t size;
    e_mem_t mem_desc;  // Epiphany memory descriptor
    
public:
    ParallellaMemory(size_t size);
    ~ParallellaMemory();
    
    bool Allocate();
    void* GetHostPtr() { return host_addr; }
    void* GetDevicePtr() { return device_addr; }
    size_t GetSize() const { return size; }
    
    // Memory transfer operations
    bool HostToDevice(const void* src, size_t size);
    bool DeviceToHost(void* dst, size_t size);
    bool SyncToDevice();
    bool SyncFromDevice();
    
    typedef ParallellaMemory CLASSNAME;
};

// Parallella/Epiphany kernel wrapper
class ParallellaKernel {
private:
    String kernel_name;
    e_kernel_t* kernel;
    ParallellaDevice* device;
    
public:
    ParallellaKernel(const String& name, ParallellaDevice* dev);
    ~ParallellaKernel();
    
    bool Load(const String& kernel_path);
    bool Execute(int rows, int cols, void* args, size_t arg_size);
    
    typedef ParallellaKernel CLASSNAME;
};
#endif

// Parallella device manager
class ParallellaDevice {
private:
    bool initialized;
#ifdef PARALLELLA_SUPPORT
    e_platform_t platform;
    e_epiphany_t dev;
    e_group_config_t dev_config;
#endif
    int num_cores;
    int rows, cols;
    
public:
    ParallellaDevice();
    ~ParallellaDevice();
    
    bool Initialize();
    void Shutdown();
    
    bool IsInitialized() const { return initialized; }
    
#ifdef PARALLELLA_SUPPORT
    e_epiphany_t* GetDeviceHandle() { return &dev; }
    e_platform_t* GetPlatformHandle() { return &platform; }
#endif
    
    // Get device information
    int GetNumCores() const { return num_cores; }
    int GetRows() const { return rows; }
    int GetCols() const { return cols; }
    
    // Neural network operations that can be offloaded
    bool RunConvolution(Volume& input, Volume& output, Volume& filters, Volume& biases);
    bool RunFullyConnected(Volume& input, Volume& output, Volume& weights, Volume& biases);
    bool RunActivation(Volume& input, Volume& output, const String& activation_type);
    bool RunSoftmax(Volume& input, Volume& output);
    
    typedef ParallellaDevice CLASSNAME;
};

// Parallella-aware layer base class
class ParallellaLayerBase {
protected:
    bool use_parallella;
    ParallellaDevice* device;
    
public:
    ParallellaLayerBase();
    virtual ~ParallellaLayerBase();
    
    void SetParallellaDevice(ParallellaDevice* dev) { device = dev; }
    bool IsUsingParallella() const { return use_parallella && device && device->IsInitialized(); }
    
    // Virtual methods to be implemented by derived layers
    virtual bool OffloadToParallella() { return false; }
    virtual bool ProcessOnParallella(Volume& input, Volume& output) { return false; }
    
    typedef ParallellaLayerBase CLASSNAME;
};

// Parallella-aware Convolution Layer
class ParallellaConvLayer : public ParallellaLayerBase {
private:
    int width, height, filter_count;
    int stride, pad;
    Volume filters;
    Volume biases;
    Volume output_activation;
    
public:
    ParallellaConvLayer(int w, int h, int f_count, int s = 1, int p = 0);
    
    Volume& Forward(Volume& input, bool is_training = false);
    void Backward();
    
    bool OffloadToParallella() override { return true; }
    bool ProcessOnParallella(Volume& input, Volume& output) override;
    
    typedef ParallellaConvLayer CLASSNAME;
};

// Parallella-aware Fully Connected Layer
class ParallellaFullyConnLayer : public ParallellaLayerBase {
private:
    int input_count, neuron_count;
    Volume weights;
    Volume biases;
    Volume output_activation;
    
public:
    ParallellaFullyConnLayer(int input_cnt, int neuron_cnt);
    
    Volume& Forward(Volume& input, bool is_training = false);
    void Backward();
    
    bool OffloadToParallella() override { return true; }
    bool ProcessOnParallella(Volume& input, Volume& output) override;
    
    typedef ParallellaFullyConnLayer CLASSNAME;
};

// Parallella-enabled Network class
class ParallellaNetwork {
private:
    Vector<ParallellaLayerBase*> layers;
    ParallellaDevice device;
    bool use_hardware_acceleration;
    
public:
    ParallellaNetwork();
    ~ParallellaNetwork();
    
    void AddLayer(ParallellaLayerBase* layer);
    Volume& Forward(Volume& input, bool is_training = false);
    void Backward();
    
    bool InitializeHardware();
    void SetHardwareAcceleration(bool enable) { use_hardware_acceleration = enable; }
    bool IsHardwareAccelerated() const { return use_hardware_acceleration && device.IsInitialized(); }
    
    typedef ParallellaNetwork CLASSNAME;
};

// Utility functions for Parallella
namespace ParallellaUtils {
    bool IsParallellaAvailable();
    String GetParallellaDeviceInfo();
    bool InitializeParallella();
    void ShutdownParallella();
}

} // namespace ConvNet

#endif