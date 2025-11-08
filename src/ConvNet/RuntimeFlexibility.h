#ifndef _ConvNet_RuntimeFlexibility_h_
#define _ConvNet_RuntimeFlexibility_h_

#include "ConvNet.h"
#include "MemoryPool.h"
#include <functional>

namespace ConvNet {

// Base interface for runtime-flexible layers using type erasure
class RuntimeLayer {
public:
    virtual ~RuntimeLayer() = default;
    
    // Essential layer methods
    virtual Volume& Forward(Volume& input, bool is_training = false) = 0;
    virtual void Backward() = 0;
    virtual void Init(int input_width, int input_height, int input_depth) = 0;
    virtual Vector<ParametersAndGradients>& GetParametersAndGradients() = 0;
    virtual String GetKey() const = 0;
    virtual void Store(ValueMap& map) const = 0;
    virtual void Load(const ValueMap& map) = 0;
    virtual String ToString() const = 0;
    
    // Get output volume
    virtual Volume& GetOutput() = 0;
    
    // Clone for multithreading safety if needed
    virtual std::unique_ptr<RuntimeLayer> Clone() const = 0;
};

// Type-erased wrapper for CRTP-based layers
template<typename ConcreteLayer>
class RuntimeLayerWrapper : public RuntimeLayer {
private:
    ConcreteLayer layer;

public:
    template<typename... Args>
    RuntimeLayerWrapper(Args&&... args) : layer(std::forward<Args>(args)...) {}
    
    // Implement the RuntimeLayer interface by forwarding to the concrete layer
    Volume& Forward(Volume& input, bool is_training = false) override {
        return layer.Forward(input, is_training);
    }
    
    void Backward() override {
        layer.Backward();
    }
    
    void Init(int input_width, int input_height, int input_depth) override {
        layer.Init(input_width, input_height, input_depth);
    }
    
    Vector<ParametersAndGradients>& GetParametersAndGradients() override {
        return layer.GetParametersAndGradients();
    }
    
    String GetKey() const override {
        return layer.GetKey();
    }
    
    void Store(ValueMap& map) const override {
        layer.Store(map);
    }
    
    void Load(const ValueMap& map) override {
        layer.Load(map);
    }
    
    String ToString() const override {
        return layer.ToString();
    }
    
    Volume& GetOutput() override {
        return layer.GetOutput();
    }
    
    std::unique_ptr<RuntimeLayer> Clone() const override {
        return std::make_unique<RuntimeLayerWrapper<ConcreteLayer>>(layer);
    }
};

// Runtime-flexible Network class using type erasure
class RuntimeNet {
private:
    // Use U++ Vector with raw pointers since U++ containers can't hold std::unique_ptr
    Vector<RuntimeLayer*> layers;
    
public:
    RuntimeNet() = default;
    
    // Add a layer through type erasure - caller must ensure lifetime
    template<typename LayerType, typename... Args>
    LayerType& AddLayer(Args&&... args) {
        auto* layer_ptr = new RuntimeLayerWrapper<LayerType>(std::forward<Args>(args)...);
        LayerType* ptr = &(static_cast<RuntimeLayerWrapper<LayerType>*>(layer_ptr)->layer);
        layers.Add(layer_ptr);
        return *ptr;
    }
    
    // Add an already constructed layer - caller must ensure lifetime
    void AddLayer(RuntimeLayer* layer) {
        layers.Add(layer);
    }
    
    // Runtime-accessible methods
    Volume& Forward(Volume& input, bool is_training = false) {
        Volume* current = &input;
        for (int i = 0; i < layers.GetCount(); i++) {
            current = &layers[i]->Forward(*current, is_training);
        }
        return *current;
    }
    
    Volume& Forward(const Vector<VolumePtr>& inputs, bool is_training = false) {
        // For now, use the first input - this can be extended for more complex cases
        if (inputs.GetCount() > 0) {
            return Forward(*inputs[0], is_training);
        }
        throw std::runtime_error("No inputs provided to network forward pass");
    }
    
    double Backward(const Vector<double>& y) {
        // Backpropagate through layers in reverse order
        for (int i = layers.GetCount() - 1; i >= 0; i--) {
            layers[i]->Backward();
        }
        // Return some cost value (implementation depends on the last layer)
        return 0.0; // Placeholder - should come from loss calculation
    }
    
    int GetPrediction() {
        if (layers.GetCount() > 0) {
            return layers.Top()->GetOutput().GetMaxColumn();
        }
        return -1;
    }
    
    // Get parameters and gradients for all layers
    Vector<ParametersAndGradients> GetParametersAndGradients() {
        Vector<ParametersAndGradients> result;
        for (int i = 0; i < layers.GetCount(); i++) {
            auto& layer_params = layers[i]->GetParametersAndGradients();
            for (int j = 0; j < layer_params.GetCount(); j++) {
                result.Add() = layer_params[j];
            }
        }
        return result;
    }
    
    // Serialization methods
    void Store(ValueMap& map) const {
        Value layers_array;
        for (int i = 0; i < layers.GetCount(); i++) {
            ValueMap layer_map;
            layers[i]->Store(layer_map);
            layers_array.Add(layer_map);
        }
        map.GetAdd("layers") = layers_array;
    }
    
    void Load(const ValueMap& map) {
        // Note: This just clears existing layers - we don't implement full loading
        // since that would require a factory system
        for (int i = 0; i < layers.GetCount(); i++) {
            delete layers[i];  // Clean up memory - only safe if we own it
        }
        layers.Clear();
        
        Value layers_array = map.GetValue(map.Find("layers"));
        // We're not implementing the loading here - would need factory pattern
    }
    
    // Get layer count
    int GetLayerCount() const {
        return layers.GetCount();
    }
    
    // Get a specific layer by index (for inspection only)
    RuntimeLayer& GetLayer(int index) {
        ASSERT(index >= 0 && index < layers.GetCount());
        return *layers[index];
    }
    
    // Clear all layers
    void Clear() {
        for (int i = 0; i < layers.GetCount(); i++) {
            delete layers[i];  // Clean up memory - only safe if we own it
        }
        layers.Clear();
    }
    
    ~RuntimeNet() {
        // Clean up memory - this assumes we own the layers
        // In a real implementation, ownership would need to be carefully managed
        for (int i = 0; i < layers.GetCount(); i++) {
            delete layers[i];
        }
    }
};

// Factory function to create runtime-flexible network with JSON configuration
class RuntimeNetworkFactory {
public:
    // Create a network from JSON configuration
    static std::unique_ptr<RuntimeNet> CreateFromJSON(const ValueMap& config) {
        auto net = std::make_unique<RuntimeNet>();
        
        Value layers_config = config.GetValue(config.Find("layers"));
        // In U++, check if value is an array by checking its type or using other methods
        // For now, we'll just have an empty implementation as creating from JSON 
        // would require full factory pattern with layer creation
        
        return net;
    }
};

// Type-erased function wrappers for trainers to enable scripting
class RuntimeTrainer {
public:
    virtual ~RuntimeTrainer() = default;
    
    virtual double Train(Volume& input, const Vector<double>& labels) = 0;
    virtual void Init(RuntimeNet& net) = 0;
    virtual void SetOption(const String& name, double value) = 0;
    virtual double GetOption(const String& name) = 0;
    virtual String GetKey() const = 0;
    virtual void Store(ValueMap& map) const = 0;
    virtual void Load(const ValueMap& map) = 0;
};

// Wrapper for concrete trainers (like SgdTrainer, AdamTrainer, etc.)
template<typename ConcreteTrainer>
class RuntimeTrainerWrapper : public RuntimeTrainer {
private:
    ConcreteTrainer trainer;

public:
    template<typename... Args>
    RuntimeTrainerWrapper(Args&&... args) : trainer(std::forward<Args>(args)...) {}

    double Train(Volume& input, const Vector<double>& labels) override {
        return trainer.Train(input, labels);
    }

    void Init(RuntimeNet& net) override {
        // Note: This is simplified since we can't directly convert between
        // RuntimeNet and the original ConvNet::Net
        // This would require additional bridge functionality in a full implementation
    }

    void SetOption(const String& name, double value) override {
        trainer.SetOption(name, value);
    }

    double GetOption(const String& name) override {
        return trainer.GetOption(name);
    }

    String GetKey() const override {
        return trainer.GetKey();
    }

    void Store(ValueMap& map) const override {
        trainer.Store(map);
    }

    void Load(const ValueMap& map) override {
        trainer.Load(map);
    }
};

// Runtime-flexible session for training and inference
class RuntimeSession {
private:
    std::unique_ptr<RuntimeNet> network;
    std::unique_ptr<RuntimeTrainer> trainer;
    
public:
    RuntimeSession() = default;
    
    void SetNetwork(std::unique_ptr<RuntimeNet> net) {
        network = std::move(net);
    }
    
    void SetTrainer(std::unique_ptr<RuntimeTrainer> train) {
        trainer = std::move(train);
    }
    
    Volume& Forward(Volume& input, bool is_training = false) {
        if (!network) {
            throw std::runtime_error("Network not set in session");
        }
        return network->Forward(input, is_training);
    }
    
    double Train(Volume& input, const Vector<double>& labels) {
        if (!trainer) {
            throw std::runtime_error("Trainer not set in session");
        }
        return trainer->Train(input, labels);
    }
    
    // Export the trained model
    ValueMap ExportModel() const {
        ValueMap result;
        if (network) {
            network->Store(result);
        }
        return result;
    }
    
    // Import a trained model
    void ImportModel(const ValueMap& model) {
        if (network) {
            network->Load(model);
        }
    }
    
    // Create a session from JSON configuration
    static std::unique_ptr<RuntimeSession> CreateFromConfig(const ValueMap& config) {
        auto session = std::make_unique<RuntimeSession>();
        
        // Create network if specified
        if (config.Find("network") != -1) {
            // Note: In a full implementation, this would create from config
            // For now we'll create an empty session
        }
        
        // Create trainer if specified
        if (config.Find("trainer") != -1) {
            // Note: In a full implementation, this would create trainer from config
        }
        
        return session;
    }
};

} // namespace ConvNet

#endif