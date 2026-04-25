#include "TransformerLayers.h"

namespace ConvNet {

// SwinPatchMergingCRTP Implementation
SwinPatchMergingCRTP::SwinPatchMergingCRTP(int dim, int out_dim)
    : dim(dim), out_dim(out_dim) {
    // Initialize the layer
}

Volume& SwinPatchMergingCRTP::ForwardImpl(Volume& input, bool is_training) {
    input_activation = input;
    
    // Assuming input has shape [height, width, channels] where height and width should be even
    int input_height = input.GetWidth();   // Using width as height (for square patches)
    int input_width = input.GetHeight();   // Using height as width (for square patches) 
    int input_channels = input.GetDepth();
    
    // Calculate output dimensions after merging
    int output_height = input_height / 2;
    int output_width = input_width / 2;
    int output_channels = out_dim;
    
    // The output shape is [output_height, output_width, output_channels]
    output_activation.Init(output_height, output_width, output_channels);
    
    // Perform patch merging by concatenating adjacent patches and applying linear transformation
    for (int h = 0; h < output_height; h++) {
        for (int w = 0; w < output_width; w++) {
            // Each output patch combines 4 input patches: (2h, 2w), (2h, 2w+1), (2h+1, 2w), (2h+1, 2w+1)
            int input_h1 = 2 * h;
            int input_h2 = 2 * h + 1;
            int input_w1 = 2 * w;
            int input_w2 = 2 * w + 1;
            
            // Concatenate features from 4 patches - create a temporary volume for this
            Volume temp_concat(1, 1, input_channels * 4);
            
            // Copy channel values from each of the 4 patches
            int concat_idx = 0;
            for (int c = 0; c < input_channels; c++) {
                if (input_h1 < input_height && input_w1 < input_width) {
                    temp_concat.Set(0, 0, concat_idx++, input.Get(input_h1, input_w1, c));
                } else {
                    temp_concat.Set(0, 0, concat_idx++, 0.0);  // Padding with zero
                }
                
                if (input_h1 < input_height && input_w2 < input_width) {
                    temp_concat.Set(0, 0, concat_idx++, input.Get(input_h1, input_w2, c));
                } else {
                    temp_concat.Set(0, 0, concat_idx++, 0.0);  // Padding with zero
                }
                
                if (input_h2 < input_height && input_w1 < input_width) {
                    temp_concat.Set(0, 0, concat_idx++, input.Get(input_h2, input_w1, c));
                } else {
                    temp_concat.Set(0, 0, concat_idx++, 0.0);  // Padding with zero
                }
                
                if (input_h2 < input_height && input_w2 < input_width) {
                    temp_concat.Set(0, 0, concat_idx++, input.Get(input_h2, input_w2, c));
                } else {
                    temp_concat.Set(0, 0, concat_idx++, 0.0);  // Padding with zero
                }
            }
            
            // Apply linear transformation: multiply concatenated features with weight matrix and add bias
            for (int out_c = 0; out_c < output_channels; out_c++) {
                double sum = 0.0;
                for (int concat_c = 0; concat_c < input_channels * 4; concat_c++) {
                    sum += temp_concat.Get(0, 0, concat_c) * reduction_weight.Get(out_c, concat_c, 0);
                }
                sum += reduction_bias.Get(out_c, 0, 0);
                output_activation.Set(h, w, out_c, sum);
            }
        }
    }
    
    return output_activation;
}

void SwinPatchMergingCRTP::BackwardImpl() {
    // Implementation of backward pass for patch merging
    // This is a simplified implementation - in practice, this would be more complex
}

void SwinPatchMergingCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // Note: in the context of patch merging, we treat input_width as the number of patches height-wise
    // and input_height as the number of patches width-wise (this might be opposite to conventional usage)
    input_resolution[0] = input_height;
    input_resolution[1] = input_width;
    dim = input_depth;  // Number of channels per patch
    
    // The reduction layer takes 4 * input_depth features (from 4 patches) and outputs out_dim features
    int input_features = 4 * dim;  // 4 patches concatenated
    reduction_weight.Init(out_dim, input_features, 1);
    
    // Initialize weights using Xavier/Glorot initialization
    double scale = sqrt(2.0 / (input_features + out_dim));
    for (int i = 0; i < reduction_weight.GetSize(); i++) {
        reduction_weight.Set(i, Randomf() * scale);
    }
    
    // Initialize bias to zeros
    reduction_bias.Init(out_dim, 1, 1);
    for (int i = 0; i < reduction_bias.GetSize(); i++) {
        reduction_bias.Set(i, 0.0);
    }
}

Vector<ParametersAndGradients>& SwinPatchMergingCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params;
    params.Clear();

    if (reduction_weight.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &reduction_weight;
        // No explicit grad member in ParametersAndGradients; gradient is typically stored elsewhere
        params.Add() = pg;
    }

    if (reduction_bias.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &reduction_bias;
        params.Add() = pg;
    }

    return params;
}

void SwinPatchMergingCRTP::StoreImpl(ValueMap& map) const {
    map.Set("dim", dim);
    map.Set("out_dim", out_dim);
    
    // Store weight and bias data
    Vector<double> weight_data;
    for (int i = 0; i < reduction_weight.GetSize(); i++) {
        weight_data.Add(reduction_weight.Get(i));
    }
    map.GetAdd("reduction_weight") = (Value)weight_data;
    
    Vector<double> bias_data;
    for (int i = 0; i < reduction_bias.GetSize(); i++) {
        bias_data.Add(reduction_bias.Get(i));
    }
    map.GetAdd("reduction_bias") = (Value)bias_data;
}

void SwinPatchMergingCRTP::LoadImpl(const ValueMap& map) {
    int dim_idx = map.Find("dim");
    dim = (dim_idx >= 0) ? (int)map.GetValue(dim_idx) : 96;

    int out_dim_idx = map.Find("out_dim");
    out_dim = (out_dim_idx >= 0) ? (int)map.GetValue(out_dim_idx) : dim * 2;  // Default to doubling the dim

    // Load weight and bias data
    int weight_idx = map.Find("reduction_weight");
    if (weight_idx >= 0) {
        Vector<double> weight_data = (Vector<double>)map.GetValue(weight_idx);
        int input_features = 4 * dim;  // 4 patches concatenated
        reduction_weight.Init(out_dim, input_features, 1);

        for (int i = 0; i < min(reduction_weight.GetSize(), weight_data.GetCount()); i++) {
            reduction_weight.Set(i, weight_data[i]);
        }
    }

    int bias_idx = map.Find("reduction_bias");
    if (bias_idx >= 0) {
        Vector<double> bias_data = (Vector<double>)map.GetValue(bias_idx);
        reduction_bias.Init(out_dim, 1, 1);

        for (int i = 0; i < min(reduction_bias.GetSize(), bias_data.GetCount()); i++) {
            reduction_bias.Set(i, bias_data[i]);
        }
    }
}

String SwinPatchMergingCRTP::ToStringImpl() const {
    String s;
    s << "SwinPatchMerging: input_dim=" << dim 
      << ", output_dim=" << out_dim;
    return s;
}

// WindowAttentionCRTP Implementation
WindowAttentionCRTP::WindowAttentionCRTP(int window_size, int num_heads, int input_dim)
    : window_size(window_size), num_heads(num_heads), input_dim(input_dim) {
    head_dim = input_dim / num_heads;
    
    // Initialize relative position bias table
    // For a window of size M x M, we have (2M-1)^2 possible relative positions
    int table_size = (2 * window_size - 1) * (2 * window_size - 1);
    relative_position_bias_table.Init(table_size, 1, num_heads);
}

Volume& WindowAttentionCRTP::ForwardImpl(Volume& input, bool is_training) {
    input_activation = input;
    
    // The input has shape [num_windows, window_size*window_size, input_dim]
    int num_windows = input.GetWidth();
    int tokens_per_window = input.GetHeight();
    int feature_dim = input.GetDepth();
    
    output_activation.Init(num_windows, tokens_per_window, input_dim);
    
    // For each window, apply window-based attention
    for (int win = 0; win < num_windows; win++) {
        // Process each token in the window (each patch within the window)
        for (int token = 0; token < tokens_per_window; token++) {
            // Calculate Q, K, V for all tokens in the window
            Volume Q(tokens_per_window, head_dim, num_heads);
            Volume K(tokens_per_window, head_dim, num_heads);
            Volume V(tokens_per_window, head_dim, num_heads);
            
            // Compute Q, K, V by linear projection
            for (int t = 0; t < tokens_per_window; t++) {
                for (int head = 0; head < num_heads; head++) {
                    for (int hd = 0; hd < head_dim; hd++) {
                        double sum_q = 0.0, sum_k = 0.0, sum_v = 0.0;
                        
                        // Get the input for current token t and compute Q, K, V
                        for (int d = 0; d < feature_dim; d++) {
                            double input_val = input.Get(win, t, d);
                            sum_q += input_val * wq.Get(head * head_dim + hd, d, 0);
                            sum_k += input_val * wk.Get(head * head_dim + hd, d, 0);
                            sum_v += input_val * wv.Get(head * head_dim + hd, d, 0);
                        }
                        
                        // Add bias
                        sum_q += bq.Get(head * head_dim + hd, 0, 0);
                        sum_k += bk.Get(head * head_dim + hd, 0, 0);
                        sum_v += bv.Get(head * head_dim + hd, 0, 0);
                        
                        Q.Set(t, hd, head, sum_q);
                        K.Set(t, hd, head, sum_k);
                        V.Set(t, hd, head, sum_v);
                    }
                }
            }
            
            // Apply scaled dot-product attention for current token
            for (int head = 0; head < num_heads; head++) {
                // Calculate attention scores for current token
                Volume scores(tokens_per_window, 1, 1);  // Attention scores for this token
                for (int t = 0; t < tokens_per_window; t++) {
                    double score = 0.0;
                    for (int hd = 0; hd < head_dim; hd++) {
                        score += Q.Get(token, hd, head) * K.Get(t, hd, head);
                    }
                    scores.Set(t, 0, 0, score / sqrt(head_dim));  // Scale by sqrt(d_k)
                }
                
                // Add relative position bias
                for (int t = 0; t < tokens_per_window; t++) {
                    // In a real implementation, we would use the relative_position_index
                    // Here we'll simplify by using a flat index for the bias
                    int bias_idx = 0;  // This should be calculated based on relative positions
                    double rel_bias = relative_position_bias_table.Get(bias_idx, 0, head);
                    double current_score = scores.Get(t, 0, 0);
                    scores.Set(t, 0, 0, current_score + rel_bias);
                }
                
                // Apply softmax to get attention weights
                // Find max for numerical stability
                double max_score = scores.Get(0, 0, 0);
                for (int t = 1; t < tokens_per_window; t++) {
                    max_score = max(max_score, scores.Get(t, 0, 0));
                }
                
                // Compute softmax
                double sum_exp = 0.0;
                for (int t = 0; t < tokens_per_window; t++) {
                    double exp_val = exp(scores.Get(t, 0, 0) - max_score);
                    scores.Set(t, 0, 0, exp_val);
                    sum_exp += exp_val;
                }
                
                // Normalize
                for (int t = 0; t < tokens_per_window; t++) {
                    double norm_val = scores.Get(t, 0, 0) / sum_exp;
                    scores.Set(t, 0, 0, norm_val);
                }
                
                // Compute output for this head and token
                double output_value = 0.0;
                for (int t = 0; t < tokens_per_window; t++) {
                    for (int hd = 0; hd < head_dim; hd++) {
                        output_value += scores.Get(t, 0, 0) * V.Get(t, hd, head);
                    }
                }
                
                // Store output in the right position
                int output_dim_idx = head * head_dim + 0;  // Simplified - only first dimension
                if (output_dim_idx < input_dim) {
                    output_activation.Set(win, token, output_dim_idx, output_value);
                }
            }
        }
    }
    
    return output_activation;
}

void WindowAttentionCRTP::BackwardImpl() {
    // Implementation of backward pass for window attention
    // This would involve backpropagating gradients through the attention mechanism
}

void WindowAttentionCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // Initialize parameters for query, key, value, and output projections
    wq.Init(input_dim, input_dim, 1);
    wk.Init(input_dim, input_dim, 1);
    wv.Init(input_dim, input_dim, 1);
    wo.Init(input_dim, input_dim, 1);
    
    bq.Init(input_dim, 1, 1);
    bk.Init(input_dim, 1, 1);
    bv.Init(input_dim, 1, 1);
    bo.Init(input_dim, 1, 1);
    
    // Initialize with Xavier/Glorot initialization
    double scale = sqrt(2.0 / (input_dim + input_dim));
    for (int i = 0; i < wq.GetSize(); i++) {
        wq.Set(i, Randomf() * scale);
        wk.Set(i, Randomf() * scale);
        wv.Set(i, Randomf() * scale);
        wo.Set(i, Randomf() * scale);
    }
    
    // Initialize bias to zeros
    for (int i = 0; i < bq.GetSize(); i++) {
        bq.Set(i, 0.0);
        bk.Set(i, 0.0);
        bv.Set(i, 0.0);
        bo.Set(i, 0.0);
    }
    
    // Initialize relative position bias table
    int table_size = (2 * window_size - 1) * (2 * window_size - 1);
    relative_position_bias_table.Init(table_size, 1, num_heads);
    for (int i = 0; i < relative_position_bias_table.GetSize(); i++) {
        relative_position_bias_table.Set(i, Randomf() * 0.02);  // Small random initialization
    }
}

Vector<ParametersAndGradients>& WindowAttentionCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params;
    params.Clear();

    if (wq.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &wq;
        params.Add() = pg;
    }

    if (wk.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &wk;
        params.Add() = pg;
    }

    if (wv.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &wv;
        params.Add() = pg;
    }

    if (wo.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &wo;
        params.Add() = pg;
    }

    if (bq.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &bq;
        params.Add() = pg;
    }

    if (bk.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &bk;
        params.Add() = pg;
    }

    if (bv.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &bv;
        params.Add() = pg;
    }

    if (bo.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &bo;
        params.Add() = pg;
    }

    if (relative_position_bias_table.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &relative_position_bias_table;
        params.Add() = pg;
    }

    return params;
}

void WindowAttentionCRTP::StoreImpl(ValueMap& map) const {
    map.Set("window_size", window_size);
    map.Set("num_heads", num_heads);
    map.Set("input_dim", input_dim);
    
    // Store all weight and bias parameters
    #define STORE_PARAM(name) \
        do { \
            Vector<double> name##_data; \
            for (int i = 0; i < name.GetSize(); i++) { \
                name##_data.Add(name.Get(i)); \
            } \
            map.GetAdd(#name) = (Value)name##_data; \
        } while(0)
    
    STORE_PARAM(wq);
    STORE_PARAM(wk);
    STORE_PARAM(wv);
    STORE_PARAM(wo);
    STORE_PARAM(bq);
    STORE_PARAM(bk);
    STORE_PARAM(bv);
    STORE_PARAM(bo);
    STORE_PARAM(relative_position_bias_table);
    
    #undef STORE_PARAM
}

void WindowAttentionCRTP::LoadImpl(const ValueMap& map) {
    int window_size_idx = map.Find("window_size");
    window_size = (window_size_idx >= 0) ? (int)map.GetValue(window_size_idx) : 7;

    int num_heads_idx = map.Find("num_heads");
    num_heads = (num_heads_idx >= 0) ? (int)map.GetValue(num_heads_idx) : 12;

    int input_dim_idx = map.Find("input_dim");
    input_dim = (input_dim_idx >= 0) ? (int)map.GetValue(input_dim_idx) : 768;

    head_dim = input_dim / num_heads;

    #define LOAD_PARAM(name, exp_width, exp_height, exp_depth) \
        do { \
            int name##_idx = map.Find(#name); \
            if (name##_idx >= 0) { \
                Vector<double> name##_data = (Vector<double>)map.GetValue(name##_idx); \
                name.Init(exp_width, exp_height, exp_depth); \
                for (int i = 0; i < min(name.GetSize(), name##_data.GetCount()); i++) { \
                    name.Set(i, name##_data[i]); \
                } \
            } \
        } while(0)

    LOAD_PARAM(wq, input_dim, input_dim, 1);
    LOAD_PARAM(wk, input_dim, input_dim, 1);
    LOAD_PARAM(wv, input_dim, input_dim, 1);
    LOAD_PARAM(wo, input_dim, input_dim, 1);
    LOAD_PARAM(bq, input_dim, 1, 1);
    LOAD_PARAM(bk, input_dim, 1, 1);
    LOAD_PARAM(bv, input_dim, 1, 1);
    LOAD_PARAM(bo, input_dim, 1, 1);

    int table_size = (2 * window_size - 1) * (2 * window_size - 1);
    LOAD_PARAM(relative_position_bias_table, table_size, 1, num_heads);

    #undef LOAD_PARAM
}

String WindowAttentionCRTP::ToStringImpl() const {
    String s;
    s << "WindowAttention: window_size=" << window_size 
      << ", num_heads=" << num_heads 
      << ", input_dim=" << input_dim
      << ", head_dim=" << head_dim;
    return s;
}

// SwinTransformerBlockCRTP Implementation
SwinTransformerBlockCRTP::SwinTransformerBlockCRTP(int dim, int input_resolution_param[2], int num_heads, int window_size,
                                                   int shift_size, int mlp_ratio, bool mlp_bias, double mlp_dropout)
    : dim(dim), num_heads(num_heads), window_size(window_size), shift_size(shift_size),
      mlp_ratio(mlp_ratio), mlp_bias(mlp_bias), mlp_dropout(mlp_dropout),
      window_attn(window_size, num_heads, dim),
      feed_forward(dim * mlp_ratio),  // MLP hidden dim
      dropout1(mlp_dropout), dropout2(mlp_dropout) {
    this->input_resolution[0] = input_resolution_param[0];
    this->input_resolution[1] = input_resolution_param[1];
}

Volume& SwinTransformerBlockCRTP::ForwardImpl(Volume& input, bool is_training) {
    input_activation = input;
    
    // This is a simplified forward implementation
    // In a real Swin Transformer block, we would:
    // 1. Apply layer norm
    // 2. Apply window attention or shifted window attention depending on the shift_size
    // 3. Apply residual connection
    // 4. Apply second layer norm
    // 5. Apply MLP
    // 6. Apply residual connection
    
    // For simplification, we'll just pass the input through the window attention with skip connection
    Volume& attention_output = window_attn.Forward(input, is_training);
    
    // Residual connection with input
    output_activation.Init(input.GetWidth(), input.GetHeight(), input.GetDepth());
    
    for (int i = 0; i < input.GetSize(); i++) {
        // Add original input to attention output (residual connection)
        output_activation.Set(i, input.Get(i) + attention_output.Get(i));
    }
    
    return output_activation;
}

void SwinTransformerBlockCRTP::BackwardImpl() {
    // Implementation of backward pass for Swin Transformer block
    // This would involve backpropagating gradients through the attention and MLP layers
}

void SwinTransformerBlockCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // Initialize the components
    input_resolution[0] = input_width;  // Height of feature map
    input_resolution[1] = input_height; // Width of feature map
    
    // Initialize sub-components
    window_attn.Init(input_width, input_height, input_depth);
    
    // For the MLP (feed-forward network), we need to compute the hidden dimension
    int mlp_hidden_dim = dim * mlp_ratio;
    
    // Note: The actual MLP initialization would need to be done differently
    // This is a simplified representation
}

Vector<ParametersAndGradients>& SwinTransformerBlockCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params;
    params.Clear();

    // Add parameters from window attention
    Vector<ParametersAndGradients>& attn_params = window_attn.GetParametersAndGradients();
    for (int i = 0; i < attn_params.GetCount(); i++) {
        params.Add() = attn_params[i];
    }

    // Add parameters from feed-forward network
    Vector<ParametersAndGradients>& ff_params = feed_forward.GetParametersAndGradients();
    for (int i = 0; i < ff_params.GetCount(); i++) {
        params.Add() = ff_params[i];
    }

    return params;
}

void SwinTransformerBlockCRTP::StoreImpl(ValueMap& map) const {
    map.GetAdd("dim") = dim;
    // Store input_resolution as an array in ValueMap
    Value input_resolution_val;
    input_resolution_val.Add(input_resolution[0]);
    input_resolution_val.Add(input_resolution[1]);
    map.GetAdd("input_resolution") = input_resolution_val;
    map.GetAdd("num_heads") = num_heads;
    map.GetAdd("window_size") = window_size;
    map.GetAdd("shift_size") = shift_size;
    map.GetAdd("mlp_ratio") = mlp_ratio;
    map.GetAdd("mlp_bias") = mlp_bias;
    map.GetAdd("mlp_dropout") = mlp_dropout;

    // Store sub-components
    ValueMap attn_map;
    window_attn.Store(attn_map);
    map.GetAdd("window_attn") = attn_map;

    // For feed_forward, we'll need to store it separately as well
    // This is a simplified approach
}

void SwinTransformerBlockCRTP::LoadImpl(const ValueMap& map) {
    int dim_idx = map.Find("dim");
    dim = (dim_idx >= 0) ? (int)map.GetValue(dim_idx) : 96;

    int res_idx = map.Find("input_resolution");
    if (res_idx >= 0) {
        Value res = map.GetValue(res_idx);
        if (res.GetCount() >= 2) {
            input_resolution[0] = (int)res[0];
            input_resolution[1] = (int)res[1];
        } else {
            // Default resolution
            input_resolution[0] = 56;
            input_resolution[1] = 56;
        }
    } else {
        // Default resolution
        input_resolution[0] = 56;
        input_resolution[1] = 56;
    }

    int num_heads_idx = map.Find("num_heads");
    num_heads = (num_heads_idx >= 0) ? (int)map.GetValue(num_heads_idx) : 3;

    int window_size_idx = map.Find("window_size");
    window_size = (window_size_idx >= 0) ? (int)map.GetValue(window_size_idx) : 7;

    int shift_size_idx = map.Find("shift_size");
    shift_size = (shift_size_idx >= 0) ? (int)map.GetValue(shift_size_idx) : 0;

    int mlp_ratio_idx = map.Find("mlp_ratio");
    mlp_ratio = (mlp_ratio_idx >= 0) ? (int)map.GetValue(mlp_ratio_idx) : 4;

    int mlp_bias_idx = map.Find("mlp_bias");
    mlp_bias = (mlp_bias_idx >= 0) ? (bool)map.GetValue(mlp_bias_idx) : true;

    int mlp_dropout_idx = map.Find("mlp_dropout");
    mlp_dropout = (mlp_dropout_idx >= 0) ? (double)map.GetValue(mlp_dropout_idx) : 0.0;

    // Load sub-components
    int attn_idx = map.Find("window_attn");
    if (attn_idx >= 0) {
        ValueMap attn_map = map.GetValue(attn_idx);
        window_attn = WindowAttentionCRTP(window_size, num_heads, dim);
        window_attn.Load(attn_map);
    }
}

String SwinTransformerBlockCRTP::ToStringImpl() const {
    String s;
    s << "SwinTransformerBlock: dim=" << dim 
      << ", input_resolution=[" << input_resolution[0] << "," << input_resolution[1] << "]"
      << ", num_heads=" << num_heads 
      << ", window_size=" << window_size
      << ", shift_size=" << shift_size;
    return s;
}

} // namespace ConvNet


namespace ConvNet {

// MaskedMultiHeadAttentionCRTP Implementation
MaskedMultiHeadAttentionCRTP::MaskedMultiHeadAttentionCRTP(int embed_dim, int num_heads)
    : embed_dim(embed_dim), num_heads(num_heads) {
    head_dim = embed_dim / num_heads;

    // Initialize the layer
}

Volume& MaskedMultiHeadAttentionCRTP::ForwardImpl(Volume& input, bool is_training) {
    input_activation = input;

    int seq_len = input.GetWidth();
    int batch_size = input.GetHeight();
    int feature_dim = input.GetDepth();

    // Resize output activation
    output_activation.Init(seq_len, batch_size, feature_dim);

    // Calculate Q, K, V matrices
    Volume Q(seq_len, batch_size, embed_dim);
    Volume K(seq_len, batch_size, embed_dim);
    Volume V(seq_len, batch_size, embed_dim);

    // Linear projections to get Q, K, V
    for (int i = 0; i < seq_len; i++) {
        for (int b = 0; b < batch_size; b++) {
            for (int head = 0; head < num_heads; head++) {
                for (int d = 0; d < head_dim; d++) {
                    double input_sum_q = 0.0, input_sum_k = 0.0, input_sum_v = 0.0;

                    // Calculate input feature weighted sum for each head
                    for (int fd = 0; fd < feature_dim; fd++) {
                        double input_val = input.Get(i, b, fd);
                        input_sum_q += input_val * wq.Get(head * head_dim + d, fd, 0);
                        input_sum_k += input_val * wk.Get(head * head_dim + d, fd, 0);
                        input_sum_v += input_val * wv.Get(head * head_dim + d, fd, 0);
                    }

                    // Add bias and store in appropriate head/dim position
                    Q.Set(i, b, head * head_dim + d, input_sum_q + bq.Get(head * head_dim + d, 0, 0));
                    K.Set(i, b, head * head_dim + d, input_sum_k + bk.Get(head * head_dim + d, 0, 0));
                    V.Set(i, b, head * head_dim + d, input_sum_v + bv.Get(head * head_dim + d, 0, 0));
                }
            }
        }
    }

    // Apply attention with masking
    for (int b = 0; b < batch_size; b++) {  // For each batch
        for (int head = 0; head < num_heads; head++) {  // For each head
            // Calculate attention scores for this head
            Volume scores(seq_len, seq_len, 1);  // [seq_len, seq_len] attention matrix for this head

            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    double score = 0.0;

                    // Calculate dot product between Q[i] and K[j] for this head
                    for (int d = 0; d < head_dim; d++) {
                        score += Q.Get(i, b, head * head_dim + d) * K.Get(j, b, head * head_dim + d);
                    }

                    score = score / sqrt(head_dim);  // Scale

                    // Apply mask if available
                    if (mask.GetSize() > 0 && i < mask.GetWidth() && j < mask.GetHeight()) {
                        double mask_val = mask.Get(i, j, 0);
                        if (mask_val == 0) {  // Masked position
                            score = -1e9;  // Large negative value so it becomes ~0 after softmax
                        }
                    }

                    scores.Set(i, j, 0, score);
                }
            }

            // Apply softmax to get attention weights
            for (int i = 0; i < seq_len; i++) {
                // Find max value for numerical stability
                double max_score = scores.Get(i, 0, 0);
                for (int j = 1; j < seq_len; j++) {
                    max_score = max(max_score, scores.Get(i, j, 0));
                }

                // Calculate softmax
                double sum_exp = 0.0;
                for (int j = 0; j < seq_len; j++) {
                    double exp_val = exp(scores.Get(i, j, 0) - max_score);
                    scores.Set(i, j, 0, exp_val);
                    sum_exp += exp_val;
                }

                // Normalize and update scores
                for (int j = 0; j < seq_len; j++) {
                    double norm_val = scores.Get(i, j, 0) / sum_exp;
                    scores.Set(i, j, 0, norm_val);
                }
            }

            // Calculate output for this head
            for (int i = 0; i < seq_len; i++) {
                for (int d = 0; d < head_dim; d++) {
                    double output_val = 0.0;

                    // Weighted sum of values
                    for (int j = 0; j < seq_len; j++) {
                        output_val += scores.Get(i, j, 0) * V.Get(j, b, head * head_dim + d);
                    }

                    // Store in output activation
                    output_activation.Set(i, b, head * head_dim + d, output_val);
                }
            }
        }
    }

    // Apply final linear projection and residual connection
    for (int i = 0; i < seq_len; i++) {
        for (int b = 0; b < batch_size; b++) {
            for (int fd = 0; fd < feature_dim; fd++) {
                double sum = 0.0;

                // Linear projection back to original dimension
                for (int ed = 0; ed < embed_dim; ed++) {
                    sum += output_activation.Get(i, b, ed) * wo.Get(fd, ed, 0);
                }

                sum += bo.Get(fd, 0, 0);  // Add bias
                output_activation.Set(i, b, fd, sum + input.Get(i, b, fd));  // Add residual connection
            }
        }
    }

    return output_activation;
}

void MaskedMultiHeadAttentionCRTP::BackwardImpl() {
    // Implementation of backward pass for masked multi-head attention
    // This would involve backpropagating gradients through the attention mechanism
}

void MaskedMultiHeadAttentionCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // Store the input_depth for later use in serialization
    this->input_depth = input_depth;

    // Initialize parameters for query, key, value, and output projections
    wq.Init(embed_dim, input_depth, 1);
    wk.Init(embed_dim, input_depth, 1);
    wv.Init(embed_dim, input_depth, 1);
    wo.Init(input_depth, embed_dim, 1);  // Output projection back to original dimension

    bq.Init(embed_dim, 1, 1);
    bk.Init(embed_dim, 1, 1);
    bv.Init(embed_dim, 1, 1);
    bo.Init(input_depth, 1, 1);

    // Initialize with Xavier/Glorot initialization
    double scale = sqrt(2.0 / (input_depth + embed_dim));
    for (int i = 0; i < wq.GetSize(); i++) {
        wq.Set(i, Randomf() * scale);
        wk.Set(i, Randomf() * scale);
        wv.Set(i, Randomf() * scale);
    }

    for (int i = 0; i < wo.GetSize(); i++) {
        wo.Set(i, Randomf() * scale);
    }

    // Initialize bias to zeros
    for (int i = 0; i < bq.GetSize(); i++) {
        bq.Set(i, 0.0);
        bk.Set(i, 0.0);
        bv.Set(i, 0.0);
    }

    for (int i = 0; i < bo.GetSize(); i++) {
        bo.Set(i, 0.0);
    }
}

Vector<ParametersAndGradients>& MaskedMultiHeadAttentionCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params;
    params.Clear();

    if (wq.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &wq;
        params.Add() = pg;
    }

    if (wk.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &wk;
        params.Add() = pg;
    }

    if (wv.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &wv;
        params.Add() = pg;
    }

    if (wo.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &wo;
        params.Add() = pg;
    }

    if (bq.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &bq;
        params.Add() = pg;
    }

    if (bk.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &bk;
        params.Add() = pg;
    }

    if (bv.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &bv;
        params.Add() = pg;
    }

    if (bo.GetSize() > 0) {
        ParametersAndGradients pg;
        pg.volume = &bo;
        params.Add() = pg;
    }

    return params;
}

void MaskedMultiHeadAttentionCRTP::StoreImpl(ValueMap& map) const {
    map.Set("embed_dim", embed_dim);
    map.Set("num_heads", num_heads);
    map.Set("input_depth", input_depth);

    // Store all weight and bias parameters
    #define STORE_PARAM(name) \
        do { \
            Vector<double> name##_data; \
            for (int i = 0; i < name.GetSize(); i++) { \
                name##_data.Add(name.Get(i)); \
            } \
            map.GetAdd(#name) = (Value)name##_data; \
        } while(0)

    STORE_PARAM(wq);
    STORE_PARAM(wk);
    STORE_PARAM(wv);
    STORE_PARAM(wo);
    STORE_PARAM(bq);
    STORE_PARAM(bk);
    STORE_PARAM(bv);
    STORE_PARAM(bo);

    #undef STORE_PARAM
}

void MaskedMultiHeadAttentionCRTP::LoadImpl(const ValueMap& map) {
    int embed_dim_idx = map.Find("embed_dim");
    embed_dim = (embed_dim_idx >= 0) ? (int)map.GetValue(embed_dim_idx) : 768;

    int num_heads_idx = map.Find("num_heads");
    num_heads = (num_heads_idx >= 0) ? (int)map.GetValue(num_heads_idx) : 12;

    // Load input_depth which was stored in the InitImpl
    int input_depth_idx = map.Find("input_depth");
    int input_depth = (input_depth_idx >= 0) ? (int)map.GetValue(input_depth_idx) : embed_dim;

    head_dim = embed_dim / num_heads;

    #define LOAD_PARAM(name, exp_width, exp_height, exp_depth) \
        do { \
            int name##_idx = map.Find(#name); \
            if (name##_idx >= 0) { \
                Vector<double> name##_data = (Vector<double>)map.GetValue(name##_idx); \
                name.Init(exp_width, exp_height, exp_depth); \
                for (int i = 0; i < min(name.GetSize(), name##_data.GetCount()); i++) { \
                    name.Set(i, name##_data[i]); \
                } \
            } \
        } while(0)

    LOAD_PARAM(wq, embed_dim, input_depth, 1);
    LOAD_PARAM(wk, embed_dim, input_depth, 1);
    LOAD_PARAM(wv, embed_dim, input_depth, 1);
    LOAD_PARAM(wo, input_depth, embed_dim, 1);
    LOAD_PARAM(bq, embed_dim, 1, 1);
    LOAD_PARAM(bk, embed_dim, 1, 1);
    LOAD_PARAM(bv, embed_dim, 1, 1);
    LOAD_PARAM(bo, input_depth, 1, 1);

    #undef LOAD_PARAM
}

String MaskedMultiHeadAttentionCRTP::ToStringImpl() const {
    String s;
    s << "MaskedMultiHeadAttention: embed_dim=" << embed_dim
      << ", num_heads=" << num_heads
      << ", head_dim=" << head_dim;
    return s;
}

} // namespace ConvNet