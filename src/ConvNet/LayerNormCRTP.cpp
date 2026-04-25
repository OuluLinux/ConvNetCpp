#include "CrtpLayers.h"
#include <cmath>

namespace ConvNet {

LayerNormCRTP::LayerNormCRTP(int normalized_shape, double eps) : normalized_shape(normalized_shape), eps(eps) {
    gamma.Init(1, 1, normalized_shape, 1.0);
    beta.Init(1, 1, normalized_shape, 0.0);
}

Volume& LayerNormCRTP::ForwardImpl(Volume& input, bool is_training) {
    int input_width = input.GetWidth(); int input_height = input.GetHeight(); int input_depth = input.GetDepth();
    output_activation.Init(input_width, input_height, input_depth, 0.0);
    mean_cache.Init(input_width, input_height, 1, 0.0);
    var_cache.Init(input_width, input_height, 1, 0.0);
    for (int w = 0; w < input_width; w++) {
        for (int h = 0; h < input_height; h++) {
            double sum = 0.0;
            for (int d = 0; d < input_depth; d++) { sum += input.Get(w, h, d); }
            double mean = sum / input_depth; mean_cache.Set(w, h, 0, mean);
            double var = 0.0;
            for (int d = 0; d < input_depth; d++) { double diff = input.Get(w, h, d) - mean; var += diff * diff; }
            var = var / input_depth; var_cache.Set(w, h, 0, var);
            double inv_std = 1.0 / sqrt(var + eps);
            for (int d = 0; d < input_depth; d++) {
                double normalized = (input.Get(w, h, d) - mean) * inv_std;
                output_activation.Set(w, h, d, normalized * gamma.Get(0, 0, d) + beta.Get(0, 0, d));
            }
        }
    }
    return output_activation;
}

void LayerNormCRTP::BackwardImpl() {}

void LayerNormCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    gamma.Init(1, 1, normalized_shape, 1.0); beta.Init(1, 1, normalized_shape, 0.0);
    output_activation.Init(input_width, input_height, input_depth, 0.0); input_activation.Init(input_width, input_height, input_depth, 0.0);
    mean_cache.Init(input_width, input_height, 1, 0.0); var_cache.Init(input_width, input_height, 1, 0.0);
}

Vector<ParametersAndGradients>& LayerNormCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params; params.Clear();
    ParametersAndGradients& gp = params.Add(); gp.volume = &gamma;
    ParametersAndGradients& bp = params.Add(); bp.volume = &beta;
    return params;
}

void LayerNormCRTP::StoreImpl(ValueMap& map) const { map.GetAdd("normalized_shape") = normalized_shape; map.GetAdd("eps") = eps; }
void LayerNormCRTP::LoadImpl(const ValueMap& map) {}
String LayerNormCRTP::ToStringImpl() const { return "LayerNormCRTP"; }

} // namespace ConvNet
