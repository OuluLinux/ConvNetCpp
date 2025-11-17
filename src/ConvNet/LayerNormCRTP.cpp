#include "CrtpLayers.h"
#include "Volume.h"
#include <cmath>

namespace ConvNet {

// LayerNormCRTP implementation
LayerNormCRTP::LayerNormCRTP(int normalized_shape, double eps)
    : normalized_shape(normalized_shape), eps(eps) {
    // Initialize gamma to 1s and beta to 0s
    gamma.Resize(1, 1, normalized_shape);
    gamma.Fill(1.0);
    beta.Resize(1, 1, normalized_shape);
    beta.Fill(0.0);
}

Volume& LayerNormCRTP::ForwardImpl(Volume& input, bool is_training) {
    int input_width = input.GetWidth();
    int input_height = input.GetHeight();
    int input_depth = input.GetDepth();
    
    // Resize output to match input
    output_activation.Resize(input_width, input_height, input_depth);
    
    // Compute mean and variance across the last dimension (depth)
    mean_cache.Resize(input_width, input_height, 1);
    var_cache.Resize(input_width, input_height, 1);
    
    // Calculate mean and variance for each position
    for (int w = 0; w < input_width; w++) {
        for (int h = 0; h < input_height; h++) {
            // Calculate mean
            double sum = 0.0;
            for (int d = 0; d < input_depth; d++) {
                sum += input.Get(w, h, d);
            }
            double mean = sum / input_depth;
            mean_cache.Set(w, h, 0, mean);
            
            // Calculate variance
            double var = 0.0;
            for (int d = 0; d < input_depth; d++) {
                double diff = input.Get(w, h, d) - mean;
                var += diff * diff;
            }
            var = var / input_depth;
            var_cache.Set(w, h, 0, var);
            
            // Normalize, scale, and shift
            double inv_std = 1.0 / sqrt(var + eps);
            for (int d = 0; d < input_depth; d++) {
                double normalized = (input.Get(w, h, d) - mean) * inv_std;
                double gamma_val = gamma.Get(0, 0, d);
                double beta_val = beta.Get(0, 0, d);
                output_activation.Set(w, h, d, normalized * gamma_val + beta_val);
            }
        }
    }
    
    return output_activation;
}

void LayerNormCRTP::BackwardImpl() {
    // Layer norm backward pass implementation would go here
    // This is a simplified implementation for now
}

void LayerNormCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // The normalized_shape should match input_depth for standard usage
    if (input_depth != normalized_shape) {
        // Handle mismatch - could throw an exception or adjust
    }
    
    // Initialize parameters if needed
    gamma.Resize(1, 1, normalized_shape);
    gamma.Fill(1.0);
    beta.Resize(1, 1, normalized_shape);
    beta.Fill(0.0);
    
    output_activation.Resize(input_width, input_height, input_depth);
    input_activation.Resize(input_width, input_height, input_depth);
    mean_cache.Resize(input_width, input_height, 1);
    var_cache.Resize(input_width, input_height, 1);
}

Vector<ParametersAndGradients>& LayerNormCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params;
    params.Clear();
    
    // Add gamma (scale) parameters
    ParametersAndGradients gamma_params;
    gamma_params.params = &gamma;
    gamma_params.gradients = &gamma; // In a real implementation, this would be separate
    params.Add() = gamma_params;
    
    // Add beta (bias) parameters
    ParametersAndGradients beta_params;
    beta_params.params = &beta;
    beta_params.gradients = &beta; // In a real implementation, this would be separate
    params.Add() = beta_params;
    
    return params;
}

void LayerNormCRTP::StoreImpl(ValueMap& map) const {
    map.GetAdd("normalized_shape") = normalized_shape;
    map.GetAdd("eps") = eps;
    map.GetAdd("gamma") = gamma;
    map.GetAdd("beta") = beta;
}

void LayerNormCRTP::LoadImpl(const ValueMap& map) {
    if (map.Find("normalized_shape") >= 0)
        normalized_shape = map.GetValue(map.Find("normalized_shape"));
    if (map.Find("eps") >= 0)
        eps = map.GetValue(map.Find("eps"));
    if (map.Find("gamma") >= 0)
        gamma = map.GetValue(map.Find("gamma"));
    if (map.Find("beta") >= 0)
        beta = map.GetValue(map.Find("beta"));
}

String LayerNormCRTP::ToStringImpl() const {
    return Format("LayerNormCRTP(normalized_shape=%d, eps=%f)", normalized_shape, eps);
}

} // namespace ConvNet