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
    
    // Serialization support
    void Serialize(Stream& s) {
        int layer_count = layers.GetCount();
        s % layer_count;
        
        // Serialize each layer
        for (int i = 0; i < layers.GetCount(); i++) {
            ValueMap layer_map;
            layers[i]->Store(layer_map);
            
            // Convert ValueMap to string for serialization via Stream
            String layer_data = AsJSON(layer_map);
            s % layer_data;
        }
    }
    
    void LoadFromStream(Stream& s) {
        // Clear existing layers
        Clear();
        
        int layer_count = 0;
        s % layer_count;
        
        // Load each layer - note: this requires a factory system to recreate the specific layer types
        // For now, we'll just recreate empty layers. A full implementation would require 
        // a factory with knowledge of all possible layer types
        for (int i = 0; i < layer_count; i++) {
            String layer_data;
            s % layer_data;
            
            Value parsed = ParseJSON(layer_data);
            if (!parsed.IsNull()) {
                // In a full implementation: create the appropriate layer type from the parsed data
                // For now, we'll skip loading layer data
            }
        }
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
    // Create a network from array-of-objects JSON format (compatible with old Session::MakeLayers)
    // This is the main method that will be used since old JSON format is an array of layer objects
    static std::unique_ptr<RuntimeNet> CreateFromJSONArray(const Value& layers_array) {
        auto net = std::make_unique<RuntimeNet>();
        
        // Since the CRTP layer implementations may not be complete yet, 
        // we'll return an empty RuntimeNet for now.
        // In a full implementation, this would parse the JSON and create CRTP layers.
        
        // For now, we'll just return an empty network to allow compilation
        return net;
    }
    
    // Create a network from JSON configuration (object with layers property)
    static std::unique_ptr<RuntimeNet> CreateFromJSON(const ValueMap& config) {
        // Extract layers array from config
        Value layers_array = config.GetValue(config.Find("layers"));
        
        // Use the array method which handles the same logic
        if (!layers_array.IsNull()) {
            return CreateFromJSONArray(layers_array);
        }
        
        return std::make_unique<RuntimeNet>();
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
    
    // Serialize to JSON string format (compatible with old system)
    String ToJSONString() const {
        ValueMap result;
        if (network) {
            network->Store(result);
        }
        
        // Convert the ValueMap back to JSON string
        String json_str = AsJSON(result);
        return json_str;
    }
    
    // Deserialize from JSON string format (compatible with old system)
    void FromJSONString(const String& json_str) {
        Value parsed = ParseJSON(json_str);
        if (parsed.IsNull()) {
            throw std::runtime_error("Invalid JSON string for deserialization");
        }
        
        // Convert to ValueMap and load
        ValueMap config;
        // This is a simplified approach - in a full implementation, 
        // we'd handle both maps and arrays appropriately
        ImportModel(config);
    }
    
    // Create a session from JSON configuration
    static std::unique_ptr<RuntimeSession> CreateFromConfig(const ValueMap& config) {
        auto session = std::make_unique<RuntimeSession>();

        // For now, this is a minimal implementation since RuntimeNetworkFactory::CreateFromJSON
        // is also minimal. In a complete implementation, this would create the network
        // and potentially the trainer from the configuration.
        auto network = RuntimeNetworkFactory::CreateFromJSON(config);
        session->SetNetwork(std::move(network));

        return session;
    }
    
    // Create a session from JSON string (to match the old Session::MakeLayers interface)
    static std::unique_ptr<RuntimeSession> CreateFromJSONString(const String& json) {
        Value parsed = ParseJSON(json);
        if (parsed.IsNull()) {
            throw std::runtime_error("Invalid JSON: could not parse");
        }

        // Check if it's an array of layers (old Session::MakeLayers format)
        // or a single object with layers property (new format)
        std::unique_ptr<RuntimeSession> session = std::make_unique<RuntimeSession>();
        
        if (parsed.GetCount() > 0) {  // It's an array of layer objects (old format)
            auto network = RuntimeNetworkFactory::CreateFromJSONArray(parsed);
            session->SetNetwork(std::move(network));
        } else {
            // It's an object with layers property (new format) - this is more complex to detect
            // We'll fall back to object format if array detection doesn't work properly
            // For now, treat as array format since that's the common case for Session::MakeLayers
            auto network = RuntimeNetworkFactory::CreateFromJSONArray(parsed);
            session->SetNetwork(std::move(network));
        }

        return session;
    }
    
    // Serialization support for the runtime session
    void Serialize(Stream& s) {
        // Serialize network if available
        bool has_network = network != nullptr;
        s % has_network;
        if (has_network && network) {
            // Runtime network serialization
            // This would serialize the structure and parameters of the network
            ValueMap net_map;
            network->Store(net_map);
            
            // Convert ValueMap to string for serialization via Stream
            String net_data = AsJSON(net_map);
            s % net_data;
        }
        
        // Serialize trainer if available
        bool has_trainer = trainer != nullptr;
        s % has_trainer;
        if (has_trainer && trainer) {
            // Runtime trainer serialization
            ValueMap trainer_map;
            trainer->Store(trainer_map);
            
            // Convert ValueMap to string for serialization via Stream
            String trainer_data = AsJSON(trainer_map);
            s % trainer_data;
        }
    }
    
    void LoadFromStream(Stream& s) {
        // Load network if available
        bool has_network = false;
        s % has_network;
        if (has_network) {
            String net_data;
            s % net_data;
            
            Value parsed = ParseJSON(net_data);
            if (!parsed.IsNull()) {
                ValueMap net_map;
                // In a complete implementation, we would convert the parsed Value back to ValueMap
                // For now, we'll recreate an empty network
                network = std::make_unique<RuntimeNet>();
            }
        }
        
        // Load trainer if available
        bool has_trainer = false;
        s % has_trainer;
        if (has_trainer) {
            String trainer_data;
            s % trainer_data;
            
            Value parsed = ParseJSON(trainer_data);
            if (!parsed.IsNull()) {
                // In a complete implementation, we would create the trainer from the parsed data
                // For now, we'll skip loading the trainer
            }
        }
    }
};

} // namespace ConvNet

#endif