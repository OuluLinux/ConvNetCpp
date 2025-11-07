#ifndef _ConvNet_CrtpLayers_h_
#define _ConvNet_CrtpLayers_h_

#include "Mat.h"
#include "Utilities.h"
#include <memory>

namespace ConvNet {

// Forward declarations
class FullyConnLayer;
class InputLayer;
class ConvLayer;

// CRTP Base Layer template
template<typename Derived>
class CrtpLayerBase {
public:
    // Common interface that all layers will implement
    Volume& Forward(Volume& input, bool is_training = false) {
        return static_cast<Derived*>(this)->ForwardImpl(input, is_training);
    }

    double Backward() {
        return static_cast<Derived*>(this)->BackwardImpl();
    }

    double Backward(const Vector<double>& y) {
        return static_cast<Derived*>(this)->BackwardImpl(y);
    }

    void Init(int input_width, int input_height, int input_depth) {
        static_cast<Derived*>(this)->InitImpl(input_width, input_height, input_depth);
    }

    // Getters
    const Volume& GetOutputActivation() const {
        return static_cast<Derived const*>(this)->GetOutputActivationImpl();
    }

    Volume& GetOutputActivation() {
        return static_cast<Derived*>(this)->GetOutputActivationImpl();
    }

    int GetOutputDepth() const {
        return static_cast<Derived const*>(this)->GetOutputDepthImpl();
    }

    int GetOutputWidth() const {
        return static_cast<Derived const*>(this)->GetOutputWidthImpl();
    }

    int GetOutputHeight() const {
        return static_cast<Derived const*>(this)->GetOutputHeightImpl();
    }
};

// CRTP Input Layer
class InputLayer : public CrtpLayerBase<InputLayer> {
public:
    int output_depth = 0;
    int output_width = 0;
    int output_height = 0;

    Volume output_activation;

public:
    InputLayer() = default;
    InputLayer(InputLayer&&) = default;
    InputLayer& operator=(InputLayer&&) = default;
    ~InputLayer() = default;
    // Copy operations are not supported due to Vector<Volume> members
    InputLayer(const InputLayer&) = delete;
    InputLayer& operator=(const InputLayer&) = delete;

    Volume& ForwardImpl(Volume& input, bool is_training = false) {
        output_activation = input; // Copy input directly
        return output_activation;
    }

    double BackwardImpl() {
        // Input layer has no backward pass
        return 0.0;
    }

    double BackwardImpl(const Vector<double>& y) {
        // Input layer has no backward pass
        return 0.0;
    }

    void InitImpl(int input_width, int input_height, int input_depth) {
        output_width = input_width;
        output_height = input_height;
        output_depth = input_depth;
        output_activation.Init(input_width, input_height, input_depth, 0.0);
    }

    const Volume& GetOutputActivationImpl() const { return output_activation; }
    Volume& GetOutputActivationImpl() { return output_activation; }
    int GetOutputDepthImpl() const { return output_depth; }
    int GetOutputWidthImpl() const { return output_width; }
    int GetOutputHeightImpl() const { return output_height; }
    
    String ToString() const {
        return Format("Input: w:%d, h:%d, d:%d", output_width, output_height, output_depth);
    }
};

// CRTP Fully Connected Layer
class FullyConnLayer : public CrtpLayerBase<FullyConnLayer> {
public:
    // Persistent
    int output_depth = 0;
    int output_width = 0;
    int output_height = 0;
    int input_count = 0;
    int neuron_count = 0;
    double bias_pref = 0.0;
    double l1_decay_mul = 0.0;
    double l2_decay_mul = 1.0;

    // Layers
    Volume biases;
    Vector<Volume> filters;

    // Runtime state
    Volume* input_activation = nullptr;
    Volume output_activation;

public:
    FullyConnLayer() = default;
    FullyConnLayer(FullyConnLayer&&) = default;
    FullyConnLayer& operator=(FullyConnLayer&&) = default;
    ~FullyConnLayer() = default;
    // Copy operations are not supported due to Vector<Volume> members
    FullyConnLayer(const FullyConnLayer&) = delete;
    FullyConnLayer& operator=(const FullyConnLayer&) = delete;

    // CRTP Interface Implementation
    Volume& ForwardImpl(Volume& input, bool is_training = false) {
        input_activation = &input;
        output_activation.Init(1, 1, output_depth, 0.0);

        for (int i = 0; i < output_depth; i++) {
            double a = 0.0;
            for (int d = 0; d < input_count; d++) {
                a += input.Get(d) * filters[i].Get(d); // for efficiency use Vols directly for now
            }

            a += biases.Get(i);
            output_activation.Set(i, a);
        }

        return output_activation;
    }

    double BackwardImpl() {
        if (!input_activation) throw Exception("input_activation is null");
        Volume& input = *input_activation;
        if (output_activation.GetLength() == 0) throw Exception("output_activation length is zero");

        input.ZeroGradients(); // zero out the gradient in input Vol

        // compute gradient wrt weights and data
        for (int i = 0; i < output_depth; i++) {
            Volume& tfi = filters[i];
            double chain_gradient_ = output_activation.GetGradient(i);

            for (int d = 0; d < input_count; d++) {
                input.SetGradient(d, input.GetGradient(d) + tfi.Get(d) * chain_gradient_); // grad wrt input data
                tfi.SetGradient(d, tfi.GetGradient(d) + input.Get(d) * chain_gradient_); // grad wrt params
            }
            biases.SetGradient(i, biases.GetGradient(i) + chain_gradient_);
        }

        double loss = 0;
        for(int i = 0; i < input_count; i++) {
            double dy = input.GetGradient(i);
            loss += 0.5 * dy * dy;
        }
        return loss;
    }

    double BackwardImpl(const Vector<double>& y) {
        for(int i = 0; i < y.GetCount(); i++)
            output_activation.SetGradient(i, y[i]);

        return BackwardImpl();
    }

    void InitImpl(int input_width, int input_height, int input_depth) {
        // Computed values
        output_depth = neuron_count;
        output_width = 1;
        output_height = 1;
        input_count = input_width * input_height * input_depth;

        // Initializations
        double bias = bias_pref;
        for (int i = 0; i < output_depth; i++) {
            filters.Add().Init(1, 1, input_count);
        }

        biases.Init(1, 1, output_depth, bias);
    }

    // Interface implementations for base class
    const Volume& GetOutputActivationImpl() const { return output_activation; }
    Volume& GetOutputActivationImpl() { return output_activation; }
    int GetOutputDepthImpl() const { return output_depth; }
    int GetOutputWidthImpl() const { return output_width; }
    int GetOutputHeightImpl() const { return output_height; }
    
    String ToString() const {
        return Format("Fully Connected: w:%d, h:%d, d:%d, bias-pref:%2!,n, neurons:%d, l1-decay:%2!,n, l2-decay:%2!,n",
            output_width, output_height, output_depth, bias_pref, neuron_count, l1_decay_mul, l2_decay_mul);
    }
};

// CRTP Convolutional Layer
class ConvLayer : public CrtpLayerBase<ConvLayer> {
public:
    // Persistent
    int output_depth = 0;
    int output_width = 0;
    int output_height = 0;
    int input_depth = 0;
    int input_width = 0;
    int input_height = 0;
    int filter_count = 0;
    int stride = 1;
    int pad = 0;
    double bias_pref = 0.0;
    double l1_decay_mul = 0.0;
    double l2_decay_mul = 1.0;

    // Layers
    Volume biases;
    Vector<Volume> filters;

    // Runtime state
    Volume* input_activation = nullptr;
    Volume output_activation;

public:
    ConvLayer() = default;
    ConvLayer(ConvLayer&&) = default;
    ConvLayer& operator=(ConvLayer&&) = default;
    ~ConvLayer() = default;
    // Copy operations are not supported due to Vector<Volume> members
    ConvLayer(const ConvLayer&) = delete;
    ConvLayer& operator=(const ConvLayer&) = delete;

    Volume& ForwardImpl(Volume& input, bool is_training = false) {
        input_activation = &input;
        output_width = (input_width + 2 * pad - filters[0].GetWidth()) / stride + 1;
        output_height = (input_height + 2 * pad - filters[0].GetHeight()) / stride + 1;
        output_activation.Init(output_width, output_height, output_depth, 0.0);

        for (int d = 0; d < filter_count; d++) {
            for (int y = 0; y < output_height; y++) {
                int y_start = y * stride - pad;
                int y_end = y_start + filters[d].GetHeight();

                for (int x = 0; x < output_width; x++) {
                    int x_start = x * stride - pad;
                    int x_end = x_start + filters[d].GetWidth();

                    double a = 0.0;
                    for (int fy = y_start; fy < y_end; fy++) {
                        for (int fx = x_start; fx < x_end; fx++) {
                            for (int fd = 0; fd < input_depth; fd++) {
                                if (fy >= 0 && fy < input_height && fx >= 0 && fx < input_width) {
                                    a += input.Get(fx, fy, fd) * filters[d].Get(fx - x_start, fy - y_start, fd);
                                }
                            }
                        }
                    }

                    a += biases.Get(d);
                    output_activation.Set(x, y, d, a);
                }
            }
        }

        return output_activation;
    }

    double BackwardImpl() {
        if (!input_activation) throw Exception("input_activation is null");
        Volume& input = *input_activation;

        input.ZeroGradients(); // zero out the gradient in input Vol

        for (int d = 0; d < filter_count; d++) {
            for (int y = 0; y < output_height; y++) {
                int y_start = y * stride - pad;
                int y_end = y_start + filters[d].GetHeight();

                for (int x = 0; x < output_width; x++) {
                    int x_start = x * stride - pad;
                    int x_end = x_start + filters[d].GetWidth();

                    double chain_gradient = output_activation.GetGradient(x, y, d);
                    biases.SetGradient(d, biases.GetGradient(d) + chain_gradient);

                    for (int fy = y_start; fy < y_end; fy++) {
                        for (int fx = x_start; fx < x_end; fx++) {
                            for (int fd = 0; fd < input_depth; fd++) {
                                if (fy >= 0 && fy < input_height && fx >= 0 && fx < input_width) {
                                    input.SetGradient(fx, fy, fd, input.GetGradient(fx, fy, fd) + 
                                        filters[d].Get(fx - x_start, fy - y_start, fd) * chain_gradient);

                                    filters[d].SetGradient(fx - x_start, fy - y_start, fd, 
                                        filters[d].GetGradient(fx - x_start, fy - y_start, fd) + 
                                        input.Get(fx, fy, fd) * chain_gradient);
                                }
                            }
                        }
                    }
                }
            }
        }

        double loss = 0;
        for(int i = 0; i < input.GetLength(); i++) {
            double dy = input.GetGradient(i);
            loss += 0.5 * dy * dy;
        }
        return loss;
    }

    double BackwardImpl(const Vector<double>& y) {
        // For conv layer, we typically don't use this form of backward in this way
        throw NotImplementedException();
    }

    void InitImpl(int input_width, int input_height, int input_depth) {
        this->input_width = input_width;
        this->input_height = input_height;
        this->input_depth = input_depth;

        output_width = (input_width + 2 * pad - filters[0].GetWidth()) / stride + 1;
        output_height = (input_height + 2 * pad - filters[0].GetHeight()) / stride + 1;
        output_depth = filter_count;

        // Initialize filters and biases
        double bias = bias_pref;
        for (int i = 0; i < filter_count; i++) {
            filters.Add().Init(filters[0].GetWidth(), filters[0].GetHeight(), input_depth, 0.0);
        }

        biases.Init(1, 1, filter_count, bias);
    }

    const Volume& GetOutputActivationImpl() const { return output_activation; }
    Volume& GetOutputActivationImpl() { return output_activation; }
    int GetOutputDepthImpl() const { return output_depth; }
    int GetOutputWidthImpl() const { return output_width; }
    int GetOutputHeightImpl() const { return output_height; }
    
    String ToString() const {
        return Format("Conv: w:%d, h:%d, d:%d", output_width, output_height, output_depth);
    }
};

// Type-erased wrapper for runtime flexibility
class LayerWrapper {
private:
    struct LayerConcept {
        virtual ~LayerConcept() = default;
        virtual ConvNet::Volume& Forward(ConvNet::Volume& input, bool is_training) = 0;
        virtual double Backward() = 0;
        virtual double Backward(const ConvNet::Vector<double>& y) = 0;
        virtual void Init(int input_width, int input_height, int input_depth) = 0;
        virtual const ConvNet::Volume& GetOutputActivation() const = 0;
        virtual ConvNet::Volume& GetOutputActivation() = 0;
        virtual int GetOutputDepth() const = 0;
        virtual int GetOutputWidth() const = 0;
        virtual int GetOutputHeight() const = 0;
        virtual ConvNet::String ToString() const = 0;
    };

    template<typename T>
    struct LayerModel : LayerConcept {
        T layer;

        LayerModel(T&& l) : layer(std::move(l)) {}
        LayerModel(const T& l) : layer(l) {}

        ConvNet::Volume& Forward(ConvNet::Volume& input, bool is_training) override {
            return layer.Forward(input, is_training);
        }

        double Backward() override {
            return layer.Backward();
        }

        double Backward(const ConvNet::Vector<double>& y) override {
            return layer.Backward(y);
        }

        void Init(int input_width, int input_height, int input_depth) override {
            layer.Init(input_width, input_height, input_depth);
        }

        const ConvNet::Volume& GetOutputActivation() const override {
            return layer.GetOutputActivation();
        }

        ConvNet::Volume& GetOutputActivation() override {
            return layer.GetOutputActivation();
        }

        int GetOutputDepth() const override { return layer.GetOutputDepth(); }
        int GetOutputWidth() const override { return layer.GetOutputWidth(); }
        int GetOutputHeight() const override { return layer.GetOutputHeight(); }

        ConvNet::String ToString() const override {
            return layer.ToString();
        }
    };

    std::unique_ptr<LayerConcept> concept;

public:
    template<typename T>
    explicit LayerWrapper(T&& layer) : concept(std::make_unique<LayerModel<std::decay_t<T>>>(std::forward<T>(layer))) {}

    LayerWrapper(const LayerWrapper& other) = delete; // Disable copy - can be implemented with clone later
    
    LayerWrapper(LayerWrapper&&) = default;
    LayerWrapper& operator=(LayerWrapper&&) = default;

    ConvNet::Volume& Forward(ConvNet::Volume& input, bool is_training = false) {
        return concept->Forward(input, is_training);
    }

    double Backward() {
        return concept->Backward();
    }

    double Backward(const ConvNet::Vector<double>& y) {
        return concept->Backward(y);
    }

    void Init(int input_width, int input_height, int input_depth) {
        concept->Init(input_width, input_height, input_depth);
    }

    const ConvNet::Volume& GetOutputActivation() const {
        return concept->GetOutputActivation();
    }

    ConvNet::Volume& GetOutputActivation() {
        return concept->GetOutputActivation();
    }

    int GetOutputDepth() const { return concept->GetOutputDepth(); }
    int GetOutputWidth() const { return concept->GetOutputWidth(); }
    int GetOutputHeight() const { return concept->GetOutputHeight(); }
    
    ConvNet::String ToString() const {
        return concept->ToString();
    }
};

} // namespace ConvNet

#endif