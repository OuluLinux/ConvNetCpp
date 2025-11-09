#ifndef _ConvNet_TransformerLayers_h_
#define _ConvNet_TransformerLayers_h_

#include "ConvNet.h"
#include "CrtpLayers.h"
#include "RuntimeFlexibility.h"  // For layer normalization implementation

namespace ConvNet {

// Multi-Head Attention Layer
class MultiHeadAttentionCRTP : public LayerBaseCRTP<MultiHeadAttentionCRTP> {
private:
    friend class LayerBaseCRTP<MultiHeadAttentionCRTP>;

    // Core data
    int embed_dim;      // Total embedding dimension
    int num_heads;      // Number of attention heads
    int head_dim;       // Dimension per head (embed_dim / num_heads)
    
    // Weight matrices for Q, K, V projections
    Volume wq;          // Query weight matrix
    Volume wk;          // Key weight matrix
    Volume wv;          // Value weight matrix
    Volume wo;          // Output weight matrix
    
    // Bias vectors
    Volume bq;          // Query bias
    Volume bk;          // Key bias
    Volume bv;          // Value bias
    Volume bo;          // Output bias
    
    // Cached values for forward/backward pass
    Volume output_activation;
    Volume input_activation;
    Vector<Volume> queries;     // Queries for each head
    Vector<Volume> keys;        // Keys for each head
    Vector<Volume> values;      // Values for each head
    Vector<Volume> attention_scores;  // Attention weights for each head
    Vector<Volume> attention_outputs; // Output from each head
    
    // Temporary volumes for attention computation
    Volume scores;              // Attention scores (Q*K^T)
    Volume attention_weights;   // Softmax output
    Volume output;              // Final attention output

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "multihead_attention"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }

public:
    MultiHeadAttentionCRTP(int embed_dim, int num_heads);
    MultiHeadAttentionCRTP(ValueMap values) { LoadImpl(values); }

    // Public interface
    int GetEmbedDim() const { return embed_dim; }
    int GetNumHeads() const { return num_heads; }
    int GetHeadDim() const { return head_dim; }
    
    // Scaled Dot-Product Attention helper
    Volume& ScaledDotProductAttention(Volume& query, Volume& key, Volume& value, 
                                     const Volume* mask = nullptr);
};

// Transformer Encoder Layer
class EncoderLayerCRTP : public LayerBaseCRTP<EncoderLayerCRTP> {
private:
    friend class LayerBaseCRTP<EncoderLayerCRTP>;

    // Core components
    MultiHeadAttentionCRTP self_attention;
    FullyConnLayerCRTP feed_forward;
    
    // Layer normalization components
    Volume norm1_weights;  // For self-attention
    Volume norm1_biases;
    Volume norm2_weights;  // For feed-forward
    Volume norm2_biases;
    
    // Dropout rates
    double dropout_rate;
    DropOutLayerCRTP dropout1;
    DropOutLayerCRTP dropout2;
    
    // Cached values
    Volume output_activation;
    Volume input_activation;

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "encoder_layer"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }

public:
    EncoderLayerCRTP(int embed_dim, int num_heads, int ff_dim, double dropout_rate = 0.1)
        : self_attention(embed_dim, num_heads), 
          feed_forward(ff_dim),  // Assuming this takes neuron_count as parameter
          dropout1(dropout_rate), dropout2(dropout_rate) {}
    EncoderLayerCRTP(ValueMap values) { LoadImpl(values); }

    // Public interface
    int GetEmbedDim() const { return self_attention.GetEmbedDim(); }
    int GetNumHeads() const { return self_attention.GetNumHeads(); }
    
    // Helper for layer normalization
    void ApplyLayerNorm(Volume& input, const Volume& gamma, const Volume& beta, int d_model, int seq_len);
};

// Transformer Decoder Layer
class DecoderLayerCRTP : public LayerBaseCRTP<DecoderLayerCRTP> {
private:
    friend class LayerBaseCRTP<DecoderLayerCRTP>;

    // Core components
    MultiHeadAttentionCRTP self_attention;
    MultiHeadAttentionCRTP cross_attention;  // Attention over encoder outputs
    FullyConnLayerCRTP feed_forward;
    
    // Layer normalization components
    Volume norm1_weights;  // For self-attention
    Volume norm1_biases;
    Volume norm2_weights;  // For cross-attention
    Volume norm2_biases;
    Volume norm3_weights;  // For feed-forward
    Volume norm3_biases;
    
    // Dropout rates
    double dropout_rate;
    DropOutLayerCRTP dropout1;
    DropOutLayerCRTP dropout2;
    DropOutLayerCRTP dropout3;
    
    // Cached values
    Volume output_activation;
    Volume input_activation;

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "decoder_layer"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }

public:
    DecoderLayerCRTP(int embed_dim, int num_heads, int ff_dim, double dropout_rate = 0.1)
        : self_attention(embed_dim, num_heads), 
          cross_attention(embed_dim, num_heads),  // Cross attention with encoder
          feed_forward(ff_dim),  // Assuming this takes neuron_count as parameter
          dropout1(dropout_rate), dropout2(dropout_rate), dropout3(dropout_rate) {}
    DecoderLayerCRTP(ValueMap values) { LoadImpl(values); }

    // Public interface
    int GetEmbedDim() const { return self_attention.GetEmbedDim(); }
    int GetNumHeads() const { return self_attention.GetNumHeads(); }
    
    // Helper for layer normalization
    void ApplyLayerNorm(Volume& input, const Volume& gamma, const Volume& beta, int d_model, int seq_len);
};

// Positional Encoding Layer
class PositionalEncodingCRTP : public LayerBaseCRTP<PositionalEncodingCRTP> {
private:
    friend class LayerBaseCRTP<PositionalEncodingCRTP>;

    // Core data
    int max_len;        // Maximum sequence length
    int embed_dim;      // Embedding dimension
    Volume pe;          // Precomputed positional encodings
    
    // Cached values
    Volume output_activation;
    Volume input_activation;

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "positional_encoding"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }

public:
    PositionalEncodingCRTP(int max_len, int embed_dim);
    PositionalEncodingCRTP(ValueMap values) { LoadImpl(values); }

    // Public interface
    int GetMaxLen() const { return max_len; }
    int GetEmbedDim() const { return embed_dim; }
    
    // Generate positional encodings using sine/cosine functions
    void GeneratePositionalEncodings();
};

// Complete Transformer Model
class TransformerCRTP {
private:
    // Core components
    Vector<EncoderLayerCRTP> encoder_layers;
    Vector<DecoderLayerCRTP> decoder_layers;
    PositionalEncodingCRTP positional_encoding;
    
    // Embedding layers
    int src_vocab_size;     // Source vocabulary size
    int tgt_vocab_size;     // Target vocabulary size
    int embed_dim;          // Embedding dimension
    Volume src_embedding;   // Source embedding matrix
    Volume tgt_embedding;   // Target embedding matrix
    Volume output_projection; // Output projection matrix
    
    // Output layer normalization
    Volume final_norm_weights;
    Volume final_norm_biases;

public:
    TransformerCRTP(int src_vocab_size, int tgt_vocab_size, int embed_dim, 
                   int num_heads, int num_encoder_layers, int num_decoder_layers,
                   int ff_dim, int max_seq_len, double dropout_rate = 0.1);
    
    // Forward pass
    Volume& Forward(Volume& src, Volume& tgt, bool is_training = false);
    
    // Encoder pass only
    Volume& Encode(Volume& src, bool is_training = false);
    
    // Decoder pass only
    Volume& Decode(Volume& tgt, Volume& memory, bool is_training = false);
    
    // Get parameters for training
    Vector<ParametersAndGradients> GetParametersAndGradients();
    
    // Serialization
    void Store(ValueMap& map) const;
    void Load(const ValueMap& map);
    void Serialize(Stream& s);
    
    // Public interface
    int GetEmbedDim() const { return embed_dim; }
    int GetNumHeads() const { return num_heads; }
    int GetSrcVocabSize() const { return src_vocab_size; }
    int GetTgtVocabSize() const { return tgt_vocab_size; }
    
private:
    // Helper functions
    Volume& GenerateSubsequentMask(int size);  // For causal masking in decoder
    int num_heads;
};

// Helper function to create a transformer
std::unique_ptr<TransformerCRTP> CreateTransformer(int src_vocab_size, int tgt_vocab_size, 
                                                  int embed_dim, int num_heads, 
                                                  int num_encoder_layers, int num_decoder_layers,
                                                  int ff_dim, int max_seq_len, 
                                                  double dropout_rate = 0.1);

} // namespace ConvNet

#endif