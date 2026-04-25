#include "TransformerLayers.h"

namespace ConvNet {

// ViTPatchEmbeddingCRTP Implementation
ViTPatchEmbeddingCRTP::ViTPatchEmbeddingCRTP(int patch_size, int embed_dim, int num_patches)
    : patch_size(patch_size), embed_dim(embed_dim), num_patches(num_patches) {
    // Initialize the layer
}

Volume& ViTPatchEmbeddingCRTP::ForwardImpl(Volume& input, bool is_training) {
    // Store the input for potential backward pass
    input_activation = input;
    
    int input_width = input.GetWidth();
    int input_height = input.GetHeight();
    int input_depth = input.GetDepth();
    
    // Calculate patch size and number of patches
    int patch_height = patch_size;
    int patch_width = patch_size;
    int patches_per_row = input_width / patch_width;
    int patches_per_col = input_height / patch_height;
    int total_patches = patches_per_row * patches_per_col;
    
    // The output will have dimensions [total_patches + 1, embed_dim, 1] to account for the class token
    output_activation.Init(total_patches + 1, embed_dim, 1);
    
    // Process the patches
    for (int patch_idx = 0; patch_idx < total_patches; patch_idx++) {
        int patch_row = patch_idx / patches_per_row;
        int patch_col = patch_idx % patches_per_row;
        
        // Extract patch and project to embedding dimension
        for (int embed_d = 0; embed_d < embed_dim; embed_d++) {
            double sum = 0.0;
            for (int p_h = 0; p_h < patch_height; p_h++) {
                for (int p_w = 0; p_w < patch_width; p_w++) {
                    for (int d = 0; d < input_depth; d++) {
                        int input_h = patch_row * patch_height + p_h;
                        int input_w = patch_col * patch_width + p_w;
                        // Add contribution from each pixel in the patch
                        sum += input.Get(input_w, input_h, d) * 
                               proj_weight.Get(embed_d, p_h * patch_width * input_depth + p_w * input_depth + d, 0);
                    }
                }
            }
            // Add bias
            sum += proj_bias.Get(embed_d, 0, 0);
            output_activation.Set(patch_idx, embed_d, 0, sum);
        }
    }
    
    // Add positional embedding to each patch
    for (int patch_idx = 0; patch_idx < total_patches; patch_idx++) {
        for (int embed_d = 0; embed_d < embed_dim; embed_d++) {
            double val = output_activation.Get(patch_idx, embed_d, 0);
            val += pos_embed.Get(patch_idx, embed_d, 0);
            output_activation.Set(patch_idx, embed_d, 0, val);
        }
    }
    
    return output_activation;
}

void ViTPatchEmbeddingCRTP::BackwardImpl() {
    // Implementation of backward pass for patch embedding
    // This is a simplified implementation - in practice, this would be more complex
}

void ViTPatchEmbeddingCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // Calculate number of patches
    int patches_per_row = input_width / patch_size;
    int patches_per_col = input_height / patch_size;
    num_patches = patches_per_row * patches_per_col;
    
    // Initialize projection weight matrix
    // The input to each patch is patch_size * patch_size * input_depth dimensional
    int patch_input_dim = patch_size * patch_size * input_depth;
    proj_weight.Init(embed_dim, patch_input_dim, 1);
    
    // Initialize bias
    proj_bias.Init(embed_dim, 1, 1);
    
    // Initialize positional embeddings [sequence_length, embed_dim]
    // Sequence length includes one extra for the class token
    pos_embed.Init(num_patches + 1, embed_dim, 1);
    
    // Initialize weights using Xavier/Glorot initialization
    double scale = sqrt(2.0 / (patch_input_dim + embed_dim));
    for (int i = 0; i < proj_weight.GetSize(); i++) {
        proj_weight.Set(i, Randomf() * scale);
    }

    // Initialize bias to zeros
    for (int i = 0; i < proj_bias.GetSize(); i++) {
        proj_bias.Set(i, 0.0);
    }

    // Initialize positional embeddings with learned values
    for (int i = 0; i < pos_embed.GetSize(); i++) {
        pos_embed.Set(i, Randomf() * 0.02);  // Small random initialization
    }
}

Vector<ParametersAndGradients>& ViTPatchEmbeddingCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params_and_grads;
    params_and_grads.Clear();

    if (proj_weight.GetSize() > 0) {
        ParametersAndGradients& pg = params_and_grads.Add();
        pg.volume = &proj_weight;
    }

    if (proj_bias.GetSize() > 0) {
        ParametersAndGradients& pg = params_and_grads.Add();
        pg.volume = &proj_bias;
    }

    if (pos_embed.GetSize() > 0) {
        ParametersAndGradients& pg = params_and_grads.Add();
        pg.volume = &pos_embed;
    }

    return params_and_grads;
}

void ViTPatchEmbeddingCRTP::StoreImpl(ValueMap& map) const {
    map.GetAdd("patch_size") = patch_size;
    map.GetAdd("embed_dim") = embed_dim;
    map.GetAdd("num_patches") = num_patches;
    
    // Store weights and embeddings
    Vector<double> weight_data;
    for (int i = 0; i < proj_weight.GetSize(); i++) {
        weight_data.Add(proj_weight.Get(i));
    }
    map.GetAdd("proj_weight") = (Value)weight_data;

    Vector<double> bias_data;
    for (int i = 0; i < proj_bias.GetSize(); i++) {
        bias_data.Add(proj_bias.Get(i));
    }
    map.GetAdd("proj_bias") = (Value)bias_data;

    Vector<double> pos_embed_data;
    for (int i = 0; i < pos_embed.GetSize(); i++) {
        pos_embed_data.Add(pos_embed.Get(i));
    }
    map.GetAdd("pos_embed") = (Value)pos_embed_data;
}

void ViTPatchEmbeddingCRTP::LoadImpl(const ValueMap& map) {
    int patch_size_idx = map.Find("patch_size");
    patch_size = (patch_size_idx >= 0) ? (int)map.GetValue(patch_size_idx) : 16;

    int embed_dim_idx = map.Find("embed_dim");
    embed_dim = (embed_dim_idx >= 0) ? (int)map.GetValue(embed_dim_idx) : 768;

    int num_patches_idx = map.Find("num_patches");
    num_patches = (num_patches_idx >= 0) ? (int)map.GetValue(num_patches_idx) : 196;  // Default for 224x224 image with 16x16 patches

    // Load weights and embeddings
    int weight_idx = map.Find("proj_weight");
    if (weight_idx >= 0) {
        Vector<double> weight_data = (Vector<double>)map.GetValue(weight_idx);
        proj_weight.Init(embed_dim, patch_size * patch_size * 3, 1);  // Assuming 3 for RGB

        for (int i = 0; i < min(proj_weight.GetSize(), weight_data.GetCount()); i++) {
            proj_weight.Set(i, weight_data[i]);
        }
    }

    int bias_idx = map.Find("proj_bias");
    if (bias_idx >= 0) {
        Vector<double> bias_data = (Vector<double>)map.GetValue(bias_idx);
        proj_bias.Init(embed_dim, 1, 1);

        for (int i = 0; i < min(proj_bias.GetSize(), bias_data.GetCount()); i++) {
            proj_bias.Set(i, bias_data[i]);
        }
    }

    int pos_embed_idx = map.Find("pos_embed");
    if (pos_embed_idx >= 0) {
        Vector<double> pos_embed_data = (Vector<double>)map.GetValue(pos_embed_idx);
        pos_embed.Init(num_patches + 1, embed_dim, 1);  // +1 for class token

        for (int i = 0; i < min(pos_embed.GetSize(), pos_embed_data.GetCount()); i++) {
            pos_embed.Set(i, pos_embed_data[i]);
        }
    }
}

String ViTPatchEmbeddingCRTP::ToStringImpl() const {
    String s;
    s << "ViTPatchEmbedding: patch_size=" << patch_size 
      << ", embed_dim=" << embed_dim 
      << ", num_patches=" << num_patches;
    return s;
}

// ViTEncoderCRTP Implementation
ViTEncoderCRTP::ViTEncoderCRTP(int embed_dim, int num_heads, int ff_dim, int num_layers, double dropout_rate)
    : embed_dim(embed_dim), num_heads(num_heads), ff_dim(ff_dim), num_layers(num_layers), 
      dropout_rate(dropout_rate) {
    // Initialize encoder layers
    encoder_layers.SetCount(num_layers);
    for (int i = 0; i < num_layers; i++) {
        encoder_layers[i] = EncoderLayerCRTP(embed_dim, num_heads, ff_dim, dropout_rate);
    }
}

Volume& ViTEncoderCRTP::ForwardImpl(Volume& input, bool is_training) {
    input_activation = input;
    
    // Initialize class token if not already done
    if (class_token.GetSize() == 0) {
        // Class token has same dimension as embedding
        class_token.Init(1, embed_dim, 1);
        // Initialize with small random values
        for (int i = 0; i < class_token.GetSize(); i++) {
            class_token.Set(i, Randomf() * 0.02);
        }
    }
    
    // Add class token to the beginning of the sequence
    int seq_len = input.GetWidth();
    output_activation.Init(seq_len + 1, embed_dim, 1);
    
    // Copy the class token to position 0
    for (int emb_idx = 0; emb_idx < embed_dim; emb_idx++) {
        output_activation.Set(0, emb_idx, 0, class_token.Get(0, emb_idx, 0));
    }
    
    // Copy the input patches to positions 1 to seq_len
    for (int patch_idx = 0; patch_idx < seq_len; patch_idx++) {
        for (int emb_idx = 0; emb_idx < embed_dim; emb_idx++) {
            output_activation.Set(patch_idx + 1, emb_idx, 0, 
                                input.Get(patch_idx, emb_idx, 0));
        }
    }
    
    // Pass through encoder layers
    Volume* current_output = &output_activation;
    for (int i = 0; i < num_layers; i++) {
        current_output = &encoder_layers[i].Forward(*current_output, is_training);
    }
    
    // Store the final output
    output_activation = *current_output;
    
    return output_activation;
}

void ViTEncoderCRTP::BackwardImpl() {
    // Implementation of backward pass for ViT encoder
    // This would involve backpropagating through all encoder layers in reverse order
}

void ViTEncoderCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // Initialize encoder layers
    encoder_layers.SetCount(num_layers);
    for (int i = 0; i < num_layers; i++) {
        // Initialize each encoder layer with the appropriate dimensions
        encoder_layers[i] = EncoderLayerCRTP(embed_dim, num_heads, ff_dim, dropout_rate);
        encoder_layers[i].Init(embed_dim, 1, 1);  // Each position is treated as having depth 1
    }
    
    // Initialize class token
    class_token.Init(1, embed_dim, 1);
    for (int i = 0; i < class_token.GetSize(); i++) {
        class_token.Set(i, Randomf() * 0.02);
    }
}

Vector<ParametersAndGradients>& ViTEncoderCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params_and_grads;
    params_and_grads.Clear();

    // Add class token parameters
    if (class_token.GetSize() > 0) {
        ParametersAndGradients& pg = params_and_grads.Add();
        pg.volume = &class_token;
    }

    // Add parameters from all encoder layers
    for (int i = 0; i < num_layers; i++) {
        Vector<ParametersAndGradients>& layer_params = encoder_layers[i].GetParametersAndGradients();
        for (int j = 0; j < layer_params.GetCount(); j++) {
            params_and_grads.Add() = layer_params[j];
        }
    }

    return params_and_grads;
}

void ViTEncoderCRTP::StoreImpl(ValueMap& map) const {
    map.GetAdd("embed_dim") = embed_dim;
    map.GetAdd("num_heads") = num_heads;
    map.GetAdd("ff_dim") = ff_dim;
    map.GetAdd("num_layers") = num_layers;
    map.GetAdd("dropout_rate") = dropout_rate;
    
    // Store class token
    Vector<double> class_token_data;
    for (int i = 0; i < class_token.GetSize(); i++) {
        class_token_data.Add(class_token.Get(i));
    }
    map.GetAdd("class_token") = (Value)class_token_data;

    // Store encoder layers
    for (int i = 0; i < num_layers; i++) {
        ValueMap layer_map;
        encoder_layers[i].Store(layer_map);
        String key = "encoder_layer_" + IntStr(i);
        map.GetAdd(key) = layer_map;
    }
}

void ViTEncoderCRTP::LoadImpl(const ValueMap& map) {
    int embed_dim_idx = map.Find("embed_dim");
    embed_dim = (embed_dim_idx >= 0) ? (int)map.GetValue(embed_dim_idx) : 768;

    int num_heads_idx = map.Find("num_heads");
    num_heads = (num_heads_idx >= 0) ? (int)map.GetValue(num_heads_idx) : 12;

    int ff_dim_idx = map.Find("ff_dim");
    ff_dim = (ff_dim_idx >= 0) ? (int)map.GetValue(ff_dim_idx) : 3072;

    int num_layers_idx = map.Find("num_layers");
    num_layers = (num_layers_idx >= 0) ? (int)map.GetValue(num_layers_idx) : 12;

    int dropout_rate_idx = map.Find("dropout_rate");
    dropout_rate = (dropout_rate_idx >= 0) ? (double)map.GetValue(dropout_rate_idx) : 0.1;

    // Load class token
    int class_token_idx = map.Find("class_token");
    if (class_token_idx >= 0) {
        Vector<double> class_token_data = (Vector<double>)map.GetValue(class_token_idx);
        class_token.Init(1, embed_dim, 1);

        for (int i = 0; i < min(class_token.GetSize(), class_token_data.GetCount()); i++) {
            class_token.Set(i, class_token_data[i]);
        }
    }

    // Load encoder layers
    encoder_layers.SetCount(num_layers);
    for (int i = 0; i < num_layers; i++) {
        String key = "encoder_layer_" + IntStr(i);
        int layer_idx = map.Find(key);
        if (layer_idx >= 0) {
            ValueMap layer_map = (ValueMap)map.GetValue(layer_idx);
            encoder_layers[i] = EncoderLayerCRTP(embed_dim, num_heads, ff_dim, dropout_rate);
            encoder_layers[i].Load(layer_map);
        }
    }
}

String ViTEncoderCRTP::ToStringImpl() const {
    String s;
    s << "ViTEncoder: embed_dim=" << embed_dim 
      << ", num_heads=" << num_heads 
      << ", ff_dim=" << ff_dim 
      << ", num_layers=" << num_layers;
    return s;
}

// ViTClassifierCRTP Implementation
ViTClassifierCRTP::ViTClassifierCRTP(int num_classes, int embed_dim)
    : num_classes(num_classes), embed_dim(embed_dim) {
    // Initialize the layer
}

Volume& ViTClassifierCRTP::ForwardImpl(Volume& input, bool is_training) {
    input_activation = input;
    
    // The input is expected to have shape [seq_len, embed_dim, 1]
    // The classification is based on the class token (first position)
    int seq_len = input.GetWidth();
    
    // The output should have shape [num_classes, 1, 1]
    output_activation.Init(num_classes, 1, 1);
    
    // Use only the class token representation (first position in sequence) for classification
    // input.Get(0, embed_d, 0) gets the class token embedding
    for (int class_idx = 0; class_idx < num_classes; class_idx++) {
        double sum = 0.0;
        for (int embed_d = 0; embed_d < embed_dim; embed_d++) {
            sum += input.Get(0, embed_d, 0) * classifier_weight.Get(class_idx, embed_d, 0);
        }
        sum += classifier_bias.Get(class_idx, 0, 0);
        output_activation.Set(class_idx, 0, 0, sum);
    }
    
    return output_activation;
}

void ViTClassifierCRTP::BackwardImpl() {
    // Implementation of backward pass for classifier layer
    // This would compute gradients with respect to weights, biases, and input
}

void ViTClassifierCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // For ViT, the classifier takes the class token (first position of sequence) as input
    // The input_width should be the embedding dimension
    classifier_weight.Init(num_classes, embed_dim, 1);
    
    // Initialize weights using Xavier/Glorot initialization
    double scale = sqrt(2.0 / (embed_dim + num_classes));
    for (int i = 0; i < classifier_weight.GetSize(); i++) {
        classifier_weight.Set(i, Randomf() * scale);
    }

    // Initialize bias to zeros
    classifier_bias.Init(num_classes, 1, 1);
    for (int i = 0; i < classifier_bias.GetSize(); i++) {
        classifier_bias.Set(i, 0.0);
    }
}

Vector<ParametersAndGradients>& ViTClassifierCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params_and_grads;
    params_and_grads.Clear();

    if (classifier_weight.GetSize() > 0) {
        ParametersAndGradients& pg = params_and_grads.Add();
        pg.volume = &classifier_weight;
    }

    if (classifier_bias.GetSize() > 0) {
        ParametersAndGradients& pg = params_and_grads.Add();
        pg.volume = &classifier_bias;
    }

    return params_and_grads;
}

void ViTClassifierCRTP::StoreImpl(ValueMap& map) const {
    map.GetAdd("num_classes") = num_classes;
    map.GetAdd("embed_dim") = embed_dim;

    // Store weights
    Vector<double> weight_data;
    for (int i = 0; i < classifier_weight.GetSize(); i++) {
        weight_data.Add(classifier_weight.Get(i));
    }
    map.GetAdd("classifier_weight") = (Value)weight_data;

    // Store bias
    Vector<double> bias_data;
    for (int i = 0; i < classifier_bias.GetSize(); i++) {
        bias_data.Add(classifier_bias.Get(i));
    }
    map.GetAdd("classifier_bias") = (Value)bias_data;
}

void ViTClassifierCRTP::LoadImpl(const ValueMap& map) {
    int num_classes_idx = map.Find("num_classes");
    num_classes = (num_classes_idx >= 0) ? (int)map.GetValue(num_classes_idx) : 1000;

    int embed_dim_idx = map.Find("embed_dim");
    embed_dim = (embed_dim_idx >= 0) ? (int)map.GetValue(embed_dim_idx) : 768;

    // Load weights
    int weight_idx = map.Find("classifier_weight");
    if (weight_idx >= 0) {
        Vector<double> weight_data = (Vector<double>)map.GetValue(weight_idx);
        classifier_weight.Init(num_classes, embed_dim, 1);

        for (int i = 0; i < min(classifier_weight.GetSize(), weight_data.GetCount()); i++) {
            classifier_weight.Set(i, weight_data[i]);
        }
    }

    // Load bias
    int bias_idx = map.Find("classifier_bias");
    if (bias_idx >= 0) {
        Vector<double> bias_data = (Vector<double>)map.GetValue(bias_idx);
        classifier_bias.Init(num_classes, 1, 1);

        for (int i = 0; i < min(classifier_bias.GetSize(), bias_data.GetCount()); i++) {
            classifier_bias.Set(i, bias_data[i]);
        }
    }
}

String ViTClassifierCRTP::ToStringImpl() const {
    String s;
    s << "ViTClassifier: num_classes=" << num_classes 
      << ", embed_dim=" << embed_dim;
    return s;
}

} // namespace ConvNet