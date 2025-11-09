#include "TransformerLayers.h"
#include <algorithm>
#include <cmath>

namespace ConvNet {

// MultiHeadAttentionCRTP implementation
MultiHeadAttentionCRTP::MultiHeadAttentionCRTP(int embed_dim, int num_heads) 
    : embed_dim(embed_dim), num_heads(num_heads), head_dim(embed_dim / num_heads) {
    // Validate that embedding dimension is divisible by number of heads
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("Embedding dimension must be divisible by number of heads");
    }
    
    // Initialize weight matrices: [embed_dim x embed_dim]
    wq.Init(embed_dim, embed_dim, 1);
    wk.Init(embed_dim, embed_dim, 1);
    wv.Init(embed_dim, embed_dim, 1);
    wo.Init(embed_dim, embed_dim, 1);
    
    // Initialize bias vectors: [embed_dim]
    bq.Init(embed_dim, 1, 1);
    bk.Init(embed_dim, 1, 1);
    bv.Init(embed_dim, 1, 1);
    bo.Init(embed_dim, 1, 1);
    
    // Initialize with small random values - using existing Volume methods
    wq.Fill(0.01 * Randomf());  // Simple initialization
    wk.Fill(0.01 * Randomf());
    wv.Fill(0.01 * Randomf());
    wo.Fill(0.01 * Randomf());
    
    // Initialize bias to small values
    bq.Fill(0.0);
    bk.Fill(0.0);
    bv.Fill(0.0);
    bo.Fill(0.0);
}

Volume& MultiHeadAttentionCRTP::ForwardImpl(Volume& input, bool is_training) {
    int seq_len = input.GetDepth();  // Assuming input shape is [embed_dim, 1, seq_len]
    
    // Resize caches for this forward pass
    queries.SetCount(num_heads);
    keys.SetCount(num_heads);
    values.SetCount(num_heads);
    attention_scores.SetCount(num_heads);
    attention_outputs.SetCount(num_heads);
    
    // Project input to Q, K, V
    // Shape transformation: [embed_dim, 1, seq_len] -> [embed_dim, 1, seq_len]
    // This is simplified - in practice, we'd use matrix multiplication
    Volume q_proj = input;  // Placeholder - actual matrix multiplication needed
    Volume k_proj = input;  // Placeholder
    Volume v_proj = input;  // Placeholder
    
    // Reshape to [num_heads, head_dim, seq_len]
    // This is a conceptual representation - actual implementation would require reshaping operations
    for (int h = 0; h < num_heads; h++) {
        // Extract head h from Q, K, V
        // This would involve slicing the projected matrices
        // For now, we'll use simplified approach
        queries[h].Init(head_dim, 1, seq_len);
        keys[h].Init(head_dim, 1, seq_len);
        values[h].Init(head_dim, 1, seq_len);
        
        // Apply scaled dot-product attention for each head
        attention_outputs[h] = ScaledDotProductAttention(queries[h], keys[h], values[h]);
    }
    
    // Concatenate heads and apply output projection
    // Concat shape: [embed_dim, 1, seq_len]
    // Apply output projection: [embed_dim, embed_dim] * [embed_dim, 1, seq_len]
    // Add bias and return
    
    output_activation = input;  // Placeholder
    return output_activation;
}

void MultiHeadAttentionCRTP::BackwardImpl() {
    // Implementation for gradient computation
    // This is a complex operation involving gradients flowing back through
    // attention mechanism and the projection matrices
}

void MultiHeadAttentionCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // Initialize based on input dimensions
    // For attention, the input_depth would typically be sequence length 
    // and input_width would be embedding dimension
}

Vector<ParametersAndGradients>& MultiHeadAttentionCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params;
    params.Clear();
    
    // Add parameters for Q, K, V, and output projections
    // Each would have weights and potentially gradients
    ParametersAndGradients wq_grad;
    wq_grad.volume = &wq;
    params.Add() = wq_grad;
    
    ParametersAndGradients wk_grad;
    wk_grad.volume = &wk;
    params.Add() = wk_grad;
    
    ParametersAndGradients wv_grad;
    wv_grad.volume = &wv;
    params.Add() = wv_grad;
    
    ParametersAndGradients wo_grad;
    wo_grad.volume = &wo;
    params.Add() = wo_grad;
    
    // Add bias parameters
    ParametersAndGradients bq_grad;
    bq_grad.volume = &bq;
    params.Add() = bq_grad;
    
    ParametersAndGradients bk_grad;
    bk_grad.volume = &bk;
    params.Add() = bk_grad;
    
    ParametersAndGradients bv_grad;
    bv_grad.volume = &bv;
    params.Add() = bv_grad;
    
    ParametersAndGradients bo_grad;
    bo_grad.volume = &bo;
    params.Add() = bo_grad;
    
    return params;
}

void MultiHeadAttentionCRTP::StoreImpl(ValueMap& map) const {
    map.GetAdd("embed_dim") = embed_dim;
    map.GetAdd("num_heads") = num_heads;
    map.GetAdd("head_dim") = head_dim;
    
    // Store weight matrices and biases
    // This would require serialization functions for Volume
}

void MultiHeadAttentionCRTP::LoadImpl(const ValueMap& map) {
    embed_dim = map.GetValue(map.Find("embed_dim"));
    num_heads = map.GetValue(map.Find("num_heads"));
    head_dim = map.GetValue(map.Find("head_dim"));
}

String MultiHeadAttentionCRTP::ToStringImpl() const {
    return Format("MultiHeadAttention(embed_dim=%d, num_heads=%d)", embed_dim, num_heads);
}

Volume& MultiHeadAttentionCRTP::ScaledDotProductAttention(Volume& query, Volume& key, Volume& value, 
                                                        const Volume* mask) {
    // Compute attention scores: Q * K^T / sqrt(d_k)
    int d_k = query.GetWidth();  // dimension of keys/queries
    double scale = sqrt(d_k);
    
    // Simplified matrix multiplication representation
    // Actual implementation would require actual matrix multiplication logic
    Volume scores = query;  // Placeholder for attention scores
    
    // Apply scaling
    scores.Mul(1.0 / scale);
    
    // Apply mask if provided (for causal attention in decoder)
    if (mask != nullptr) {
        // Add large negative value to masked positions
        // This would effectively zero out those positions after softmax
    }
    
    // Apply softmax to get attention weights
    // This would require softmax implementation on the scores volume
    Volume attention_weights = scores;  // Placeholder
    
    // Compute output: attention_weights * V
    // This would be another matrix multiplication
    Volume output = value;  // Placeholder
    
    return output;
}

// EncoderLayerCRTP implementation
EncoderLayerCRTP::EncoderLayerCRTP(int embed_dim, int num_heads, int ff_dim, double dropout_rate)
    : self_attention(embed_dim, num_heads), 
      feed_forward(ff_dim),  // Note: ff_dim should be different from the one used here
      dropout1(dropout_rate), dropout2(dropout_rate) {
    // Initialize layer normalization parameters
    norm1_weights.Init(embed_dim, 1, 1);
    norm1_biases.Init(embed_dim, 1, 1);
    norm2_weights.Init(embed_dim, 1, 1);
    norm2_biases.Init(embed_dim, 1, 1);
    
    // Initialize to appropriate values for layer normalization
    norm1_weights.Fill(1.0);
    norm1_biases.Fill(0.0);
    norm2_weights.Fill(1.0);
    norm2_biases.Fill(0.0);
}

Volume& EncoderLayerCRTP::ForwardImpl(Volume& input, bool is_training) {
    // Step 1: Self-attention
    Volume& attention_output = self_attention.Forward(input, is_training);
    
    // Step 2: Add & Norm (Residual connection + Layer normalization)
    // Add input to attention output (residual connection)
    // Then normalize
    Volume& norm1_output = attention_output;  // Placeholder for layer normalization
    
    // Step 3: Feed-forward network
    Volume& ff_output = feed_forward.Forward(norm1_output, is_training);
    
    // Step 4: Add & Norm (Residual connection + Layer normalization)
    // Add norm1_output to ff_output (residual connection)  
    // Then normalize
    output_activation = ff_output;  // Placeholder for final layer normalization
    return output_activation;
}

void EncoderLayerCRTP::BackwardImpl() {
    // Backward pass implementation - would flow gradients through the 
    // layer normalization, feed-forward, residual connections, and attention
}

void EncoderLayerCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // Initialize the sub-layers based on input dimensions
    self_attention.InitImpl(input_width, input_height, input_depth);
    feed_forward.InitImpl(input_width, input_height, input_depth);
}

Vector<ParametersAndGradients>& EncoderLayerCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params;
    params.Clear();
    
    // Get parameters from self-attention layer
    auto& attention_params = self_attention.GetParametersAndGradients();
    for (int i = 0; i < attention_params.GetCount(); i++) {
        params.Add() = attention_params[i];
    }
    
    // Get parameters from feed-forward layer
    auto& ff_params = feed_forward.GetParametersAndGradients();
    for (int i = 0; i < ff_params.GetCount(); i++) {
        params.Add() = ff_params[i];
    }
    
    // Add layer normalization parameters
    ParametersAndGradients norm1_w;
    norm1_w.volume = &norm1_weights;
    params.Add() = norm1_w;
    
    ParametersAndGradients norm1_b;
    norm1_b.volume = &norm1_biases;
    params.Add() = norm1_b;
    
    ParametersAndGradients norm2_w;
    norm2_w.volume = &norm2_weights;
    params.Add() = norm2_w;
    
    ParametersAndGradients norm2_b;
    norm2_b.volume = &norm2_biases;
    params.Add() = norm2_b;
    
    return params;
}

void EncoderLayerCRTP::StoreImpl(ValueMap& map) const {
    // Store encoder layer parameters
    self_attention.Store(map.GetAdd("self_attention"));
    feed_forward.Store(map.GetAdd("feed_forward"));
    // Store normalization parameters
}

void EncoderLayerCRTP::LoadImpl(const ValueMap& map) {
    // Load encoder layer parameters
    self_attention.Load(map.GetValue(map.Find("self_attention")));
    feed_forward.Load(map.GetValue(map.Find("feed_forward")));
    // Load normalization parameters
}

String EncoderLayerCRTP::ToStringImpl() const {
    return Format("EncoderLayer(embed_dim=%d, num_heads=%d)", 
                  self_attention.GetEmbedDim(), self_attention.GetNumHeads());
}

// DecoderLayerCRTP implementation
DecoderLayerCRTP::DecoderLayerCRTP(int embed_dim, int num_heads, int ff_dim, double dropout_rate)
    : self_attention(embed_dim, num_heads), 
      cross_attention(embed_dim, num_heads),  // Cross attention with encoder
      feed_forward(ff_dim),
      dropout1(dropout_rate), dropout2(dropout_rate), dropout3(dropout_rate) {
    // Initialize layer normalization parameters
    norm1_weights.Init(embed_dim, 1, 1);
    norm1_biases.Init(embed_dim, 1, 1);
    norm2_weights.Init(embed_dim, 1, 1);
    norm2_biases.Init(embed_dim, 1, 1);
    norm3_weights.Init(embed_dim, 1, 1);
    norm3_biases.Init(embed_dim, 1, 1);
    
    // Initialize to appropriate values
    norm1_weights.Fill(1.0);
    norm1_biases.Fill(0.0);
    norm2_weights.Fill(1.0);
    norm2_biases.Fill(0.0);
    norm3_weights.Fill(1.0);
    norm3_biases.Fill(0.0);
}

Volume& DecoderLayerCRTP::ForwardImpl(Volume& input, bool is_training) {
    // Step 1: Masked self-attention
    Volume& self_attn_output = self_attention.Forward(input, is_training);
    
    // Step 2: Add & Norm (Residual + Layer norm)
    Volume& norm1_output = self_attn_output;  // Placeholder
    
    // Step 3: Cross-attention with encoder output (memory)
    // In a complete implementation, we'd need encoder memory as input
    Volume& cross_attn_output = cross_attention.Forward(norm1_output, is_training);
    
    // Step 4: Add & Norm (Residual + Layer norm)
    Volume& norm2_output = cross_attn_output;  // Placeholder
    
    // Step 5: Feed-forward network
    Volume& ff_output = feed_forward.Forward(norm2_output, is_training);
    
    // Step 6: Add & Norm (Residual + Layer norm)
    output_activation = ff_output;  // Placeholder
    return output_activation;
}

void DecoderLayerCRTP::BackwardImpl() {
    // Backward pass implementation for decoder layer
}

void DecoderLayerCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // Initialize the sub-layers based on input dimensions
    self_attention.InitImpl(input_width, input_height, input_depth);
    cross_attention.InitImpl(input_width, input_height, input_depth);
    feed_forward.InitImpl(input_width, input_height, input_depth);
}

Vector<ParametersAndGradients>& DecoderLayerCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params;
    params.Clear();
    
    // Get parameters from self-attention, cross-attention, and feed-forward layers
    auto& self_attn_params = self_attention.GetParametersAndGradients();
    for (int i = 0; i < self_attn_params.GetCount(); i++) {
        params.Add() = self_attn_params[i];
    }
    
    auto& cross_attn_params = cross_attention.GetParametersAndGradients();
    for (int i = 0; i < cross_attn_params.GetCount(); i++) {
        params.Add() = cross_attn_params[i];
    }
    
    auto& ff_params = feed_forward.GetParametersAndGradients();
    for (int i = 0; i < ff_params.GetCount(); i++) {
        params.Add() = ff_params[i];
    }
    
    // Add layer normalization parameters
    ParametersAndGradients norm1_w;
    norm1_w.volume = &norm1_weights;
    params.Add() = norm1_w;
    
    ParametersAndGradients norm1_b;
    norm1_b.volume = &norm1_biases;
    params.Add() = norm1_b;
    
    ParametersAndGradients norm2_w;
    norm2_w.volume = &norm2_weights;
    params.Add() = norm2_w;
    
    ParametersAndGradients norm2_b;
    norm2_b.volume = &norm2_biases;
    params.Add() = norm2_b;
    
    ParametersAndGradients norm3_w;
    norm3_w.volume = &norm3_weights;
    params.Add() = norm3_w;
    
    ParametersAndGradients norm3_b;
    norm3_b.volume = &norm3_biases;
    params.Add() = norm3_b;
    
    return params;
}

void DecoderLayerCRTP::StoreImpl(ValueMap& map) const {
    // Store decoder layer parameters
}

void DecoderLayerCRTP::LoadImpl(const ValueMap& map) {
    // Load decoder layer parameters
}

String DecoderLayerCRTP::ToStringImpl() const {
    return Format("DecoderLayer(embed_dim=%d, num_heads=%d)", 
                  self_attention.GetEmbedDim(), self_attention.GetNumHeads());
}

// PositionalEncodingCRTP implementation
PositionalEncodingCRTP::PositionalEncodingCRTP(int max_len, int embed_dim)
    : max_len(max_len), embed_dim(embed_dim) {
    // Initialize the positional encoding volume
    pe.Init(embed_dim, 1, max_len);
    GeneratePositionalEncodings();
}

Volume& PositionalEncodingCRTP::ForwardImpl(Volume& input, bool is_training) {
    // Add positional encoding to input
    // Input shape: [embed_dim, 1, seq_len]
    // PE shape: [embed_dim, 1, max_len] (or [embed_dim, 1, seq_len] if truncated)
    int seq_len = input.GetDepth();
    
    // Add positional encoding values to the input
    // This would involve element-wise addition
    output_activation = input;  // Placeholder - actual implementation would add PE
    
    // Add the positional encoding values up to the sequence length
    for (int pos = 0; pos < seq_len && pos < max_len; pos++) {
        for (int i = 0; i < embed_dim; i++) {
            double pe_value = pe.Get(i, 0, pos);
            double input_value = input.Get(i, 0, pos);
            output_activation.Set(i, 0, pos, input_value + pe_value);
        }
    }
    
    return output_activation;
}

void PositionalEncodingCRTP::BackwardImpl() {
    // Gradient flows directly back to input (no learnable parameters in PE)
    // No parameters to update in backward pass
}

void PositionalEncodingCRTP::InitImpl(int input_width, int input_height, int input_depth) {
    // Input dimensions should match embed_dim expectations
}

Vector<ParametersAndGradients>& PositionalEncodingCRTP::GetParametersAndGradientsImpl() {
    static Vector<ParametersAndGradients> params;
    params.Clear();  // Positional encoding has no learnable parameters
    return params;
}

void PositionalEncodingCRTP::StoreImpl(ValueMap& map) const {
    // Store positional encoding values
    // Since this is deterministic, we might just store dimensions and regenerate
    map.GetAdd("max_len") = max_len;
    map.GetAdd("embed_dim") = embed_dim;
}

void PositionalEncodingCRTP::LoadImpl(const ValueMap& map) {
    max_len = map.GetValue(map.Find("max_len"));
    embed_dim = map.GetValue(map.Find("embed_dim"));
    
    // Regenerate positional encodings
    pe.Init(embed_dim, 1, max_len);
    GeneratePositionalEncodings();
}

String PositionalEncodingCRTP::ToStringImpl() const {
    return Format("PositionalEncoding(max_len=%d, embed_dim=%d)", max_len, embed_dim);
}

void PositionalEncodingCRTP::GeneratePositionalEncodings() {
    // Generate positional encodings using sine and cosine functions
    // Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    //         PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < embed_dim; i++) {
            if (i % 2 == 0) {
                // Even indices get sine
                double angle = pos / pow(10000.0, double(i) / embed_dim);
                pe.Set(i, 0, pos, sin(angle));
            } else {
                // Odd indices get cosine
                double angle = pos / pow(10000.0, double(i - 1) / embed_dim);
                pe.Set(i, 0, pos, cos(angle));
            }
        }
    }
}

// TransformerCRTP implementation
TransformerCRTP::TransformerCRTP(int src_vocab_size, int tgt_vocab_size, int embed_dim, 
                               int num_heads, int num_encoder_layers, int num_decoder_layers,
                               int ff_dim, int max_seq_len, double dropout_rate)
    : src_vocab_size(src_vocab_size), tgt_vocab_size(tgt_vocab_size), 
      embed_dim(embed_dim), num_heads(num_heads),
      positional_encoding(max_seq_len, embed_dim) {
    // Initialize embedding matrices
    src_embedding.Init(embed_dim, src_vocab_size, 1);  // [embed_dim, vocab_size]
    tgt_embedding.Init(embed_dim, tgt_vocab_size, 1);  // [embed_dim, vocab_size]
    output_projection.Init(tgt_vocab_size, embed_dim, 1);  // [vocab_size, embed_dim]
    
    // Initialize encoder layers
    for (int i = 0; i < num_encoder_layers; i++) {
        encoder_layers.Add(EncoderLayerCRTP(embed_dim, num_heads, ff_dim, dropout_rate));
    }
    
    // Initialize decoder layers
    for (int i = 0; i < num_decoder_layers; i++) {
        decoder_layers.Add(DecoderLayerCRTP(embed_dim, num_heads, ff_dim, dropout_rate));
    }
    
    // Initialize layer normalization for output
    final_norm_weights.Init(embed_dim, 1, 1);
    final_norm_biases.Init(embed_dim, 1, 1);
    final_norm_weights.Fill(1.0);
    final_norm_biases.Fill(0.0);
    
    // Initialize embeddings with small random values
    src_embedding.FillRandom();
    tgt_embedding.FillRandom();
    output_projection.FillRandom();
}

Volume& TransformerCRTP::Forward(Volume& src, Volume& tgt, bool is_training) {
    // Encode source sequence
    Volume& encoder_output = Encode(src, is_training);
    
    // Decode target sequence using encoder output as memory
    output_activation = Decode(tgt, encoder_output, is_training);
    
    return output_activation;
}

Volume& TransformerCRTP::Encode(Volume& src, bool is_training) {
    // Apply source embedding
    // This would involve looking up embeddings for each token in src
    Volume embedded_src = src;  // Placeholder
    
    // Apply positional encoding
    Volume& pos_encoded_src = positional_encoding.Forward(embedded_src, is_training);
    
    // Pass through encoder layers
    Volume* current = &pos_encoded_src;
    for (int i = 0; i < encoder_layers.GetCount(); i++) {
        current = &encoder_layers[i].Forward(*current, is_training);
    }
    
    return *current;
}

Volume& TransformerCRTP::Decode(Volume& tgt, Volume& memory, bool is_training) {
    // Apply target embedding
    Volume embedded_tgt = tgt;  // Placeholder
    
    // Apply positional encoding
    Volume& pos_encoded_tgt = positional_encoding.Forward(embedded_tgt, is_training);
    
    // Generate causal mask for target sequence
    // This ensures each position can only attend to previous positions
    // Volume subsequent_mask = GenerateSubsequentMask(tgt.GetDepth());
    
    // Pass through decoder layers
    Volume* current = &pos_encoded_tgt;
    for (int i = 0; i < decoder_layers.GetCount(); i++) {
        // In a complete implementation, we'd pass the encoder memory to the decoder layer
        current = &decoder_layers[i].Forward(*current, is_training);
    }
    
    // Apply final layer normalization
    // This would be a simplified layer norm implementation
    output_activation = *current;  // Placeholder
    
    // Apply output projection to vocab size
    // This would convert from embedding space to vocabulary space
    return output_activation;
}

Vector<ParametersAndGradients> TransformerCRTP::GetParametersAndGradients() {
    Vector<ParametersAndGradients> params;
    
    // Add embedding parameters
    ParametersAndGradients src_emb;
    src_emb.volume = &src_embedding;
    params.Add() = src_emb;
    
    ParametersAndGradients tgt_emb;
    tgt_emb.volume = &tgt_embedding;
    params.Add() = tgt_emb;
    
    ParametersAndGradients out_proj;
    out_proj.volume = &output_projection;
    params.Add() = out_proj;
    
    // Add encoder layer parameters
    for (int i = 0; i < encoder_layers.GetCount(); i++) {
        auto& layer_params = encoder_layers[i].GetParametersAndGradients();
        for (int j = 0; j < layer_params.GetCount(); j++) {
            params.Add() = layer_params[j];
        }
    }
    
    // Add decoder layer parameters
    for (int i = 0; i < decoder_layers.GetCount(); i++) {
        auto& layer_params = decoder_layers[i].GetParametersAndGradients();
        for (int j = 0; j < layer_params.GetCount(); j++) {
            params.Add() = layer_params[j];
        }
    }
    
    // Add final normalization parameters
    ParametersAndGradients final_norm_w;
    final_norm_w.volume = &final_norm_weights;
    params.Add() = final_norm_w;
    
    ParametersAndGradients final_norm_b;
    final_norm_b.volume = &final_norm_biases;
    params.Add() = final_norm_b;
    
    return params;
}

void TransformerCRTP::Store(ValueMap& map) const {
    // Store transformer parameters
    map.GetAdd("src_vocab_size") = src_vocab_size;
    map.GetAdd("tgt_vocab_size") = tgt_vocab_size;
    map.GetAdd("embed_dim") = embed_dim;
    map.GetAdd("num_heads") = num_heads;
    
    // Store embeddings
    // Store encoder/decoder layers
    // Store positional encoding
}

void TransformerCRTP::Load(const ValueMap& map) {
    // Load transformer parameters
    src_vocab_size = map.GetValue(map.Find("src_vocab_size"));
    tgt_vocab_size = map.GetValue(map.Find("tgt_vocab_size"));
    embed_dim = map.GetValue(map.Find("embed_dim"));
    num_heads = map.GetValue(map.Find("num_heads"));
    
    // Load embeddings, layers, etc.
}

Volume& TransformerCRTP::GenerateSubsequentMask(int size) {
    // Generate a mask that prevents attending to future positions
    // Result would be a matrix of shape [size, size] where positions (i,j)
    // with j > i are masked (usually with a large negative value)
    
    // This would create a causal mask for decoder self-attention
    static Volume mask;  // In practice, this would be stored or computed as needed
    mask.Init(size, size, 1);
    
    // Fill with 0s (allowed) and -inf (masked)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (j > i) {
                mask.Set(i, j, 0, -1e9);  // Large negative value to mask
            } else {
                mask.Set(i, j, 0, 0.0);   // Allow attention
            }
        }
    }
    
    return mask;
}

// Helper function to create a transformer
std::unique_ptr<TransformerCRTP> CreateTransformer(int src_vocab_size, int tgt_vocab_size, 
                                                  int embed_dim, int num_heads, 
                                                  int num_encoder_layers, int num_decoder_layers,
                                                  int ff_dim, int max_seq_len, 
                                                  double dropout_rate) {
    return std::make_unique<TransformerCRTP>(src_vocab_size, tgt_vocab_size, embed_dim, 
                                            num_heads, num_encoder_layers, num_decoder_layers,
                                            ff_dim, max_seq_len, dropout_rate);
}

} // namespace ConvNet