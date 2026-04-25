#include "CrtpLayers.h"
#include <cmath>

namespace ConvNet {

FullyConnLayerCRTP::FullyConnLayerCRTP(int neuron_count) : neuron_count(neuron_count) {
    l1_decay_mul = 0.0;
    l2_decay_mul = 1.0;
}

Volume& FullyConnLayerCRTP::ForwardImpl(Volume& input, bool is_training) {
    input_activation = input;
    output_activation.Init(1, 1, neuron_count, 0.0);
    int input_count_loc = input.GetLength();
    for (int i = 0; i < neuron_count; i++) {
        double a = 0.0;
        const Volume& f = filters[i];
        for (int d = 0; d < input_count_loc; d++) {
            a += input.Get(d) * f.Get(d);
        }
        a += biases.Get(i);
        output_activation.Set(i, a);
    }
    return output_activation;
}

void FullyConnLayerCRTP::BackwardImpl() {
    input_activation.ZeroGradients();
    int input_count_loc = input_activation.GetLength();
    for (int i = 0; i < neuron_count; i++) {
        Volume& f = filters[i];
        double g = output_activation.GetGradient(i);
        for (int d = 0; d < input_count_loc; d++) {
            input_activation.SetGradient(d, input_activation.GetGradient(d) + f.Get(d) * g);
            f.SetGradient(d, f.GetGradient(d) + input_activation.Get(d) * g);
        }
        biases.SetGradient(i, biases.GetGradient(i) + g);
    }
}

void FullyConnLayerCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    input_count = input_width * input_height * input_depth;
    filters.SetCount(neuron_count);
    for (int i = 0; i < neuron_count; i++) {
        filters[i].Init(1, 1, input_count);
    }
    biases.Init(1, 1, neuron_count, 0.1);
}

Vector<ParametersAndGradients>& FullyConnLayerCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> p;
    p.Clear();
    for (int i = 0; i < neuron_count; i++) {
        ParametersAndGradients& pg = p.Add();
        pg.volume = &filters[i];
    }
    ParametersAndGradients& pg = p.Add();
    pg.volume = &biases;
    return p;
}

void FullyConnLayerCRTP::StoreImpl(ValueMap& map) const { map.GetAdd("neuron_count") = neuron_count; }
void FullyConnLayerCRTP::LoadImpl(const ValueMap& map) {}
String FullyConnLayerCRTP::ToStringImpl() const { return "FullyConnLayerCRTP"; }

DropOutLayerCRTP::DropOutLayerCRTP(double drop_prob) : drop_prob(drop_prob) {}

Volume& DropOutLayerCRTP::ForwardImpl(Volume& input, bool is_training) {
    input_activation = input;
    output_activation = input;
    if (is_training) {
        dropped.SetCount(input.GetLength());
        for (int i = 0; i < input.GetLength(); i++) {
            if (Random() < drop_prob) {
                output_activation.Set(i, 0);
                dropped[i] = 1;
            } else {
                dropped[i] = 0;
            }
        }
    } else {
        for (int i = 0; i < input.GetLength(); i++) { output_activation.Set(i, output_activation.Get(i) * (1.0 - drop_prob)); }
    }
    return output_activation;
}

void DropOutLayerCRTP::BackwardImpl() {
    input_activation.ZeroGradients();
    for (int i = 0; i < input_activation.GetLength(); i++) {
        if (dropped[i] == 0) { input_activation.SetGradient(i, output_activation.GetGradient(i)); }
    }
}

void DropOutLayerCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    output_activation.Init(input_width, input_height, input_depth, 0.0);
    input_activation.Init(input_width, input_height, input_depth, 0.0);
    dropped.SetCount(input_width * input_height * input_depth);
}

Vector<ParametersAndGradients>& DropOutLayerCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> p; p.Clear(); return p;
}

void DropOutLayerCRTP::StoreImpl(ValueMap& map) const { map.GetAdd("drop_prob") = drop_prob; }
void DropOutLayerCRTP::LoadImpl(const ValueMap& map) { if (map.Find("drop_prob") >= 0) drop_prob = map.GetValue(map.Find("drop_prob")); }
String DropOutLayerCRTP::ToStringImpl() const { return "DropOutLayerCRTP"; }

Volume& ReluLayerCRTP::ForwardImpl(Volume& input, bool is_training) {
    input_activation = input; output_activation = input;
    for (int i = 0; i < input.GetLength(); i++) { if (output_activation.Get(i) < 0) output_activation.Set(i, 0); }
    return output_activation;
}
void ReluLayerCRTP::BackwardImpl() {
    input_activation.ZeroGradients();
    for (int i = 0; i < input_activation.GetLength(); i++) { if (output_activation.Get(i) > 0) input_activation.SetGradient(i, output_activation.GetGradient(i)); }
}

Volume& SigmoidLayerCRTP::ForwardImpl(Volume& input, bool is_training) {
    input_activation = input; output_activation.Init(input.GetWidth(), input.GetHeight(), input.GetDepth(), 0.0);
    for (int i = 0; i < input.GetLength(); i++) { output_activation.Set(i, 1.0 / (1.0 + exp(-input.Get(i)))); }
    return output_activation;
}
void SigmoidLayerCRTP::BackwardImpl() {
    input_activation.ZeroGradients();
    for (int i = 0; i < input_activation.GetLength(); i++) {
        double v = output_activation.Get(i); input_activation.SetGradient(i, v * (1.0 - v) * output_activation.GetGradient(i));
    }
}

Volume& TanhLayerCRTP::ForwardImpl(Volume& input, bool is_training) {
    input_activation = input; output_activation.Init(input.GetWidth(), input.GetHeight(), input.GetDepth(), 0.0);
    for (int i = 0; i < input.GetLength(); i++) { output_activation.Set(i, tanh(input.Get(i))); }
    return output_activation;
}
void TanhLayerCRTP::BackwardImpl() {
    input_activation.ZeroGradients();
    for (int i = 0; i < input_activation.GetLength(); i++) {
        double v = output_activation.Get(i); input_activation.SetGradient(i, (1.0 - v * v) * output_activation.GetGradient(i));
    }
}

template<typename T> void ActivationLayerBaseCRTP<T>::InitImpl(int input_width, int input_height, int input_depth) {
    output_activation.Init(input_width, input_height, input_depth, 0.0); input_activation.Init(input_width, input_height, input_depth, 0.0);
}
template<typename T> Vector<ParametersAndGradients>& ActivationLayerBaseCRTP<T>::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> p; p.Clear(); return p;
}
template<typename T> void ActivationLayerBaseCRTP<T>::StoreImpl(ValueMap& map) const {}
template<typename T> void ActivationLayerBaseCRTP<T>::LoadImpl(const ValueMap& map) {}
template<typename T> String ActivationLayerBaseCRTP<T>::ToStringImpl() const { return "Activation"; }

template class ActivationLayerBaseCRTP<ReluLayerCRTP>;
template class ActivationLayerBaseCRTP<SigmoidLayerCRTP>;
template class ActivationLayerBaseCRTP<TanhLayerCRTP>;

} // namespace ConvNet
