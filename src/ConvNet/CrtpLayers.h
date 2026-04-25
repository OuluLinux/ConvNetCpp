#ifndef _ConvNet_CrtpLayers_h_
#define _ConvNet_CrtpLayers_h_

#include "Utilities.h"
#include "MemoryPool.h"
#include "MemoryPool.h"


namespace ConvNet {

// CRTP base class for layers - provides common interface while enabling compile-time optimization
template<typename Derived>
class LayerBaseCRTP {
public:
    // CRTP pattern: static polymorphism instead of virtual functions
    Volume& Forward(Volume& input, bool is_training = false) {
        return static_cast<Derived*>(this)->ForwardImpl(input, is_training);
    }
    
    void Backward() {
        static_cast<Derived*>(this)->BackwardImpl();
    }
    
    void Init(int input_width, int input_height, int input_depth) {
        static_cast<Derived*>(this)->InitImpl(input_width, input_height, input_depth);
    }
    
    Vector<ParametersAndGradients>& GetParametersAndGradients() {
        return static_cast<Derived*>(this)->GetParametersAndGradientsImpl();
    }
    
    String GetKey() const {
        return static_cast<const Derived*>(this)->GetKeyImpl();
    }
    
    void Store(ValueMap& map) const {
        static_cast<const Derived*>(this)->StoreImpl(map);
    }
    
    void Load(const ValueMap& map) {
        static_cast<Derived*>(this)->LoadImpl(map);
    }
    
    String ToString() const {
        return static_cast<const Derived*>(this)->ToStringImpl();
    }

    Volume& GetOutput() {
        return static_cast<Derived*>(this)->GetOutputImpl();
    }

    // Default implementations for common methods
    virtual ~LayerBaseCRTP() = default;
};

// CRTP ConvLayer implementation
class ConvLayerCRTP : public LayerBaseCRTP<ConvLayerCRTP> {
private:
    friend class LayerBaseCRTP<ConvLayerCRTP>;  // Allow base class to access private members
    
    // Core data
    int width;
    int height;
    Volume biases;
    Vector<Volume> filters;
    int filter_count;
    double l1_decay_mul;
    double l2_decay_mul;
    int stride;
    int pad;
    Volume output_activation;  // Cache for forward pass
    Volume input_activation;   // Cache for backward pass
    Volume weighted_input;     // Cache for backward pass
    
    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "conv"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }
    void SerializeImpl(Stream& s) {
        s % width % height % filter_count % l1_decay_mul % l2_decay_mul % stride % pad;
        s % biases;
        s % filters;
        s % output_activation;
        s % input_activation;
        s % weighted_input;
    }
    
public:
    ConvLayerCRTP(int width, int height, int filter_count);
    ConvLayerCRTP(ValueMap values) { LoadImpl(values); }

    // Public interface
    int GetStride() const { return stride; }
    int GetPad() const { return pad; }

    // Access to internal data for compatibility
    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
    int GetFilterCount() const { return filter_count; }
    const Vector<Volume>& GetFilters() const { return filters; }
    const Volume& GetBiases() const { return biases; }

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % width % height % filter_count % l1_decay_mul % l2_decay_mul % stride % pad;
        s % biases;
        s % filters;
        s % output_activation;
        s % input_activation;
        s % weighted_input;
    }
};

// CRTP Fully Connected Layer implementation
class FullyConnLayerCRTP : public LayerBaseCRTP<FullyConnLayerCRTP> {
private:
    friend class LayerBaseCRTP<FullyConnLayerCRTP>;
    
    // Core data
    int input_count;
    Volume biases;
    Vector<Volume> filters;  // Each represents a neuron's weights
    int neuron_count;
    double l1_decay_mul;
    double l2_decay_mul;
    Volume output_activation;
    Volume input_activation;
    
    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "fc"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }
    void SerializeImpl(Stream& s) {
        s % input_count % neuron_count % l1_decay_mul % l2_decay_mul;
        s % biases;
        s % filters;
        s % output_activation;
        s % input_activation;
    }
    
public:
    FullyConnLayerCRTP(int neuron_count);
    FullyConnLayerCRTP(ValueMap values) { LoadImpl(values); }

    // Copy constructor
    FullyConnLayerCRTP(const FullyConnLayerCRTP& other)
        : input_count(other.input_count), biases(other.biases),
          neuron_count(other.neuron_count), l1_decay_mul(other.l1_decay_mul),
          l2_decay_mul(other.l2_decay_mul), output_activation(other.output_activation),
          input_activation(other.input_activation) {
        // Explicitly copy Vector members using deep copy
        for(int i = 0; i < other.filters.GetCount(); i++) {
            filters.Add() = other.filters[i];
        }
    }

    // Copy assignment operator
    FullyConnLayerCRTP& operator=(const FullyConnLayerCRTP& other) {
        if (this != &other) {
            input_count = other.input_count;
            biases = other.biases;
            neuron_count = other.neuron_count;
            l1_decay_mul = other.l1_decay_mul;
            l2_decay_mul = other.l2_decay_mul;
            output_activation = other.output_activation;
            input_activation = other.input_activation;

            // Explicitly copy Vector members using deep copy
            filters.Clear();
            for(int i = 0; i < other.filters.GetCount(); i++) {
                filters.Add() = other.filters[i];
            }
        }
        return *this;
    }

    // Public interface
    int GetInputCount() const { return input_count; }
    int GetNeuronCount() const { return neuron_count; }
    const Vector<Volume>& GetFilters() const { return filters; }
    const Volume& GetBiases() const { return biases; }

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % input_count % neuron_count % l1_decay_mul % l2_decay_mul;
        s % biases;
        s % filters;
        s % output_activation;
        s % input_activation;
    }
};

// CRTP Activation Layer Base (for ReLU, Sigmoid, etc.)
template<typename Derived>
class ActivationLayerBaseCRTP : public LayerBaseCRTP<Derived> {
protected:
    friend class LayerBaseCRTP<Derived>;
    
    Volume output_activation;
    Volume input_activation;
    Vector<int> switchx;  // For max pooling type activations
    Vector<int> switchy;
    Vector<int> switchd;
    
    // Internal implementation methods (to be overridden by derived classes)
    virtual Volume& ForwardImpl(Volume& input, bool is_training) = 0;
    virtual void BackwardImpl() = 0;
    virtual void InitImpl(int input_width, int input_height, int input_depth);
    virtual Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    virtual void StoreImpl(ValueMap& map) const;
    virtual void LoadImpl(const ValueMap& map);
    virtual String ToStringImpl() const;
    virtual Volume& GetOutputImpl() { return output_activation; }
    virtual void SerializeImpl(Stream& s) {
        s % output_activation;
        s % input_activation;
        s % switchx;
        s % switchy;
        s % switchd;
    }

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % output_activation;
        s % input_activation;
        s % switchx;
        s % switchy;
        s % switchd;
    }

public:
    virtual ~ActivationLayerBaseCRTP() = default;
};

// CRTP ReLU Layer implementation
class ReluLayerCRTP : public ActivationLayerBaseCRTP<ReluLayerCRTP> {
private:
    friend class LayerBaseCRTP<ReluLayerCRTP>;
    
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    String GetKeyImpl() const { return "relu"; }
    
public:
    ReluLayerCRTP();
    ReluLayerCRTP(ValueMap values) { LoadImpl(values); }
};

// CRTP Sigmoid Layer implementation
class SigmoidLayerCRTP : public ActivationLayerBaseCRTP<SigmoidLayerCRTP> {
private:
    friend class LayerBaseCRTP<SigmoidLayerCRTP>;
    
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    String GetKeyImpl() const { return "sigmoid"; }
    
public:
    SigmoidLayerCRTP();
    SigmoidLayerCRTP(ValueMap values) { LoadImpl(values); }
};

// CRTP Tanh Layer implementation
class TanhLayerCRTP : public ActivationLayerBaseCRTP<TanhLayerCRTP> {
private:
    friend class LayerBaseCRTP<TanhLayerCRTP>;
    
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    String GetKeyImpl() const { return "tanh"; }
    
public:
    TanhLayerCRTP();
    TanhLayerCRTP(ValueMap values) { LoadImpl(values); }
};

// CRTP Pool Layer implementation
class PoolLayerCRTP : public LayerBaseCRTP<PoolLayerCRTP> {
private:
    friend class LayerBaseCRTP<PoolLayerCRTP>;
    
    // Core data
    int width;
    int height;
    int stride;
    int pad;
    Volume output_activation;
    Volume input_activation;
    Vector<int> switchx;
    Vector<int> switchy;
    Vector<int> switchd;
    
    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "pool"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }
    void SerializeImpl(Stream& s) {
        s % width % height % stride % pad;
        s % output_activation;
        s % input_activation;
        s % switchx;
        s % switchy;
        s % switchd;
    }
    
public:
    PoolLayerCRTP(int width, int height);
    PoolLayerCRTP(ValueMap values) { LoadImpl(values); }

    // Public interface
    int GetWidth() const { return width; }
    int GetHeight() const { return height; }
    int GetStride() const { return stride; }
    int GetPad() const { return pad; }

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % width % height % stride % pad;
        s % output_activation;
        s % input_activation;
        s % switchx;
        s % switchy;
        s % switchd;
    }
};

// CRTP Softmax Layer implementation (for classification)
class SoftmaxLayerCRTP : public LayerBaseCRTP<SoftmaxLayerCRTP> {
private:
    friend class LayerBaseCRTP<SoftmaxLayerCRTP>;
    
    // Core data
    int class_count;
    Volume output_activation;
    Volume input_activation;
    Vector<double> es;  // Exponentials cache
    
    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "softmax"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }
    void SerializeImpl(Stream& s) {
        s % class_count;
        s % output_activation;
        s % input_activation;
        s % es;
    }
    
public:
    SoftmaxLayerCRTP(int class_count);
    SoftmaxLayerCRTP(ValueMap values) { LoadImpl(values); }

    // Public interface
    int GetClassCount() const { return class_count; }

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % class_count;
        s % output_activation;
        s % input_activation;
        s % es;
    }
};

// CRTP Dropout Layer implementation
class DropOutLayerCRTP : public LayerBaseCRTP<DropOutLayerCRTP> {
private:
    friend class LayerBaseCRTP<DropOutLayerCRTP>;
    
    // Core data
    double drop_prob;
    Vector<bool> dropped;  // Track which neurons were dropped
    Volume output_activation;
    Volume input_activation;
    
    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "dropout"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }
    void SerializeImpl(Stream& s) {
        s % drop_prob;
        s % dropped;
        s % output_activation;
        s % input_activation;
    }
    
public:
    DropOutLayerCRTP(double drop_prob);
    DropOutLayerCRTP(ValueMap values) { LoadImpl(values); }

    // Copy constructor
    DropOutLayerCRTP(const DropOutLayerCRTP& other)
        : drop_prob(other.drop_prob), output_activation(other.output_activation),
          input_activation(other.input_activation) {
        // Explicitly copy Vector members
        for(int i = 0; i < other.dropped.GetCount(); i++) {
            dropped.Add() = other.dropped[i];
        }
    }

    // Copy assignment operator
    DropOutLayerCRTP& operator=(const DropOutLayerCRTP& other) {
        if (this != &other) {
            drop_prob = other.drop_prob;
            output_activation = other.output_activation;
            input_activation = other.input_activation;

            // Explicitly copy Vector members
            dropped.Clear();
            for(int i = 0; i < other.dropped.GetCount(); i++) {
                dropped.Add() = other.dropped[i];
            }
        }
        return *this;
    }

    // Public interface
    double GetDropProb() const { return drop_prob; }

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % drop_prob;
        s % dropped;
        s % output_activation;
        s % input_activation;
    }
};

// CRTP Input Layer implementation
class InputLayerCRTP : public LayerBaseCRTP<InputLayerCRTP> {
private:
    friend class LayerBaseCRTP<InputLayerCRTP>;

    // Core data
    Volume output_activation;

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "input"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }
    void SerializeImpl(Stream& s) {
        s % output_activation;
    }

public:
    InputLayerCRTP(int input_width, int input_height, int input_depth);
    InputLayerCRTP(ValueMap values) { LoadImpl(values); }

    // Public interface
    Volume& ForwardImpl(bool is_training);

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % output_activation;
    }
};

// CRTP Layer Normalization implementation
class LayerNormCRTP : public LayerBaseCRTP<LayerNormCRTP> {
private:
    friend class LayerBaseCRTP<LayerNormCRTP>;

    // Core data
    int normalized_shape;  // Size of the last dimension we're normalizing over
    double eps;            // Small epsilon for numerical stability
    Volume gamma;          // Learnable scale parameter
    Volume beta;           // Learnable bias parameter
    Volume output_activation;
    Volume input_activation;

    // Cached values for backward pass
    Volume mean_cache;
    Volume var_cache;

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "layer_norm"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }
    void SerializeImpl(Stream& s) {
        s % normalized_shape % eps;
        s % gamma;
        s % beta;
        s % output_activation;
        s % input_activation;
        s % mean_cache;
        s % var_cache;
    }

public:
    LayerNormCRTP(int normalized_shape, double eps = 1e-5);
    LayerNormCRTP() : normalized_shape(0), eps(1e-5) {}  // Default constructor
    LayerNormCRTP(ValueMap values) { LoadImpl(values); }

    // Public interface
    int GetNormalizedShape() const { return normalized_shape; }
    double GetEps() const { return eps; }
    const Volume& GetGamma() const { return gamma; }
    const Volume& GetBeta() const { return beta; }

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % normalized_shape % eps;
        s % gamma;
        s % beta;
        s % output_activation;
        s % input_activation;
        s % mean_cache;
        s % var_cache;
    }
};

// CRTP-based Network class for maximum performance
template<typename... LayerTypes>
class NetworkCRTP {
private:
    std::tuple<LayerTypes...> layers;
    static constexpr int num_layers = sizeof...(LayerTypes);
    
public:
    template<size_t I = 0>
    typename std::enable_if<I == num_layers>::type
    InitAll(int input_width, int input_height, int input_depth) {}
    
    template<size_t I = 0>
    typename std::enable_if<I < num_layers>::type
    InitAll(int input_width, int input_height, int input_depth) {
        std::get<I>(layers).Init(input_width, input_height, input_depth);
        InitAll<I + 1>(input_width, input_height, input_depth);
    }
    
    // Forward pass through all layers
    Volume& Forward(Volume& input, bool is_training = false) {
        Volume* current = &input;
        
        // Apply each layer sequentially
        ApplyForward<0>(current, is_training);
        return *current;
    }
    
    // Backward pass through all layers
    void Backward() {
        // Go through layers in reverse order
        ApplyBackward<num_layers - 1>();
    }
    
    // Get parameters and gradients for all layers
    Vector<ParametersAndGradients> GetParametersAndGradients() {
        Vector<ParametersAndGradients> result;
        GetParametersAndGradients<0>(result);
        return result;
    }
    
    // Serialization
    void Store(ValueMap& map) const {
        Value layers_array;
        StoreLayers<0>(layers_array);
        map.GetAdd("layers") = layers_array;
    }
    
    void Load(const ValueMap& map) {
        Value layers_array = map.GetValue(map.Find("layers"));
        LoadLayers<0>(layers_array);
    }
    
    // Access individual layers
    template<size_t I>
    auto GetLayer() -> decltype(std::get<I>(layers))& {
        return std::get<I>(layers);
    }

private:
    // Helper for forward pass
    template<size_t I = 0>
    typename std::enable_if<I == num_layers>::type
    ApplyForward(Volume*&, bool) const {}
    
    template<size_t I = 0>
    typename std::enable_if<I < num_layers>::type
    ApplyForward(Volume*& current, bool is_training) const {
        current = &std::get<I>(layers).Forward(*current, is_training);
        ApplyForward<I + 1>(current, is_training);
    }
    
    // Helper for backward pass
    template<size_t I>
    typename std::enable_if<I >= num_layers>::type
    ApplyBackward() const {}
    
    template<size_t I>
    typename std::enable_if<I < num_layers && I >= 0>::type
    ApplyBackward() const {
        std::get<I>(layers).Backward();
        if constexpr (I > 0) {
            ApplyBackward<I - 1>();
        }
    }
    
    // Helper for getting parameters
    template<size_t I = 0>
    typename std::enable_if<I == num_layers>::type
    GetParametersAndGradients(Vector<ParametersAndGradients>&) const {}
    
    template<size_t I = 0>
    typename std::enable_if<I < num_layers>::type
    GetParametersAndGradients(Vector<ParametersAndGradients>& result) const {
        auto& layer_params = std::get<I>(layers).GetParametersAndGradients();
        for (int j = 0; j < layer_params.GetCount(); j++) {
            result.Add() = layer_params[j];
        }
        GetParametersAndGradients<I + 1>(result);
    }
    
    // Helper for storing layers
    template<size_t I = 0>
    typename std::enable_if<I == num_layers>::type
    StoreLayers(Value&) const {}
    
    template<size_t I = 0>
    typename std::enable_if<I < num_layers>::type
    StoreLayers(Value& layers_array) const {
        ValueMap layer_map;
        std::get<I>(layers).Store(layer_map);
        layers_array.Add(layer_map);
        StoreLayers<I + 1>(layers_array);
    }
    
    // Helper for loading layers
    template<size_t I = 0>
    typename std::enable_if<I == num_layers>::type
    LoadLayers(const Value&) {}
    
    template<size_t I = 0>
    typename std::enable_if<I < num_layers>::type
    LoadLayers(const Value& layers_array) {
        if (I < layers_array.GetCount()) {
            std::get<I>(layers).Load(layers_array[I]);
        }
        LoadLayers<I + 1>(layers_array);
    }
};

// Simple CRTP Network builder for common configurations
template<typename... LayerTypes>
auto MakeNetwork(LayerTypes... layers) {
    return NetworkCRTP<LayerTypes...>{std::make_tuple(layers...)};
}

} // namespace ConvNet

#endif