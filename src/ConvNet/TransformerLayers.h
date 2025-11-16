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
    EncoderLayerCRTP(ValueMap values) : self_attention(0, 0), feed_forward(0), dropout1(0.0), dropout2(0.0) { LoadImpl(values); }

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
    DecoderLayerCRTP(ValueMap values) : self_attention(0, 0), cross_attention(0, 0), feed_forward(0), dropout1(0.0), dropout2(0.0), dropout3(0.0) { LoadImpl(values); }

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
    PositionalEncodingCRTP(ValueMap values) : pe(0, 0, 0) { LoadImpl(values); }

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

// ViT Patch Embedding Layer
class ViTPatchEmbeddingCRTP : public LayerBaseCRTP<ViTPatchEmbeddingCRTP> {
private:
    friend class LayerBaseCRTP<ViTPatchEmbeddingCRTP>;

    // Core data
    int patch_size;      // Size of each patch (e.g., 16x16)
    int embed_dim;       // Embedding dimension
    int num_patches;     // Number of patches (calculated from image dimensions)

    // Projection weight matrix and bias
    Volume proj_weight;  // Weight matrix for linear projection
    Volume proj_bias;    // Bias vector

    // Positional embedding
    Volume pos_embed;    // Learnable positional embeddings

    // Cached values for forward/backward pass
    Volume output_activation;
    Volume input_activation;

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "vit_patch_embed"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }

public:
    ViTPatchEmbeddingCRTP(int patch_size, int embed_dim, int num_patches);
    ViTPatchEmbeddingCRTP(ValueMap values) : proj_weight(0, 0, 0), proj_bias(0, 0, 0), pos_embed(0, 0, 0) { LoadImpl(values); }

    // Public interface
    int GetPatchSize() const { return patch_size; }
    int GetEmbedDim() const { return embed_dim; }
    int GetNumPatches() const { return num_patches; }

    // Helper to create patches from input image
    Volume CreatePatches(const Volume& input);
};

// ViT Encoder (stack of transformer encoder layers)
class ViTEncoderCRTP : public LayerBaseCRTP<ViTEncoderCRTP> {
private:
    friend class LayerBaseCRTP<ViTEncoderCRTP>;

    // Core data
    int embed_dim;        // Embedding dimension
    int num_heads;        // Number of attention heads
    int ff_dim;           // Feed-forward dimension
    int num_layers;       // Number of encoder layers
    double dropout_rate;  // Dropout rate

    // Transformer encoder layers
    Vector<EncoderLayerCRTP> encoder_layers;

    // Class token
    Volume class_token;   // Learnable class token
    Volume class_token_expanded;  // Class token expanded to batch size

    // Cached values
    Volume output_activation;
    Volume input_activation;

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "vit_encoder"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }

public:
    ViTEncoderCRTP(int embed_dim, int num_heads, int ff_dim, int num_layers, double dropout_rate = 0.1);
    ViTEncoderCRTP(ValueMap values) { LoadImpl(values); }

    // Public interface
    int GetEmbedDim() const { return embed_dim; }
    int GetNumHeads() const { return num_heads; }
    int GetFFDim() const { return ff_dim; }
    int GetNumLayers() const { return num_layers; }
};

// ViT Classifier Head
class ViTClassifierCRTP : public LayerBaseCRTP<ViTClassifierCRTP> {
private:
    friend class LayerBaseCRTP<ViTClassifierCRTP>;

    // Core data
    int num_classes;      // Number of classes for classification
    int embed_dim;        // Input embedding dimension

    // Classification weight matrix and bias
    Volume classifier_weight;  // Weight matrix for classification
    Volume classifier_bias;    // Bias vector

    // Cached values
    Volume output_activation;
    Volume input_activation;

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "vit_classifier"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }

public:
    ViTClassifierCRTP(int num_classes, int embed_dim);
    ViTClassifierCRTP(ValueMap values) : classifier_weight(0, 0, 0), classifier_bias(0, 0, 0) { LoadImpl(values); }

    // Public interface
    int GetNumClasses() const { return num_classes; }
    int GetEmbedDim() const { return embed_dim; }
};

// Swin Transformer Patch Merging Layer
class SwinPatchMergingCRTP : public LayerBaseCRTP<SwinPatchMergingCRTP> {
private:
    friend class LayerBaseCRTP<SwinPatchMergingCRTP>;

    // Core data
    int input_resolution[2];  // [height, width] of input patches
    int dim;                  // Input dimension per patch
    int out_dim;              // Output dimension after merging

    // Linear projection for patch merging
    Volume reduction_weight;  // Weight matrix for reducing 4 patches to 1
    Volume reduction_bias;    // Bias vector

    // Cached values for forward/backward pass
    Volume output_activation;
    Volume input_activation;

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "swin_patch_merge"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }

public:
    SwinPatchMergingCRTP(int dim, int out_dim);
    SwinPatchMergingCRTP(ValueMap values) : reduction_weight(0, 0, 0), reduction_bias(0, 0, 0) { LoadImpl(values); }

    // Public interface
    int GetInputDim() const { return dim; }
    int GetOutputDim() const { return out_dim; }
};

// Window-based Multi-Head Self-Attention (W-MSA) Layer
class WindowAttentionCRTP : public LayerBaseCRTP<WindowAttentionCRTP> {
private:
    friend class LayerBaseCRTP<WindowAttentionCRTP>;

    // Core data
    int window_size;      // Window size (e.g., 7 for 7x7 windows)
    int num_heads;        // Number of attention heads
    int head_dim;         // Dimension per head
    int input_dim;        // Total input dimension

    // Weight matrices for Q, K, V projections
    Volume wq;            // Query weight matrix
    Volume wk;            // Key weight matrix
    Volume wv;            // Value weight matrix
    Volume wo;            // Output weight matrix

    // Bias vectors
    Volume bq;            // Query bias
    Volume bk;            // Key bias
    Volume bv;            // Value bias
    Volume bo;            // Output bias

    // Relative position bias
    Volume relative_position_bias_table;  // Bias lookup table for relative positions
    int relative_position_index[49][49];  // Precomputed index mapping (for 7x7 windows)

    // Cached values for forward/backward pass
    Volume output_activation;
    Volume input_activation;

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "window_attention"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }

public:
    WindowAttentionCRTP(int window_size, int num_heads, int input_dim);
    WindowAttentionCRTP(ValueMap values) : wq(0, 0, 0), wk(0, 0, 0), wv(0, 0, 0), wo(0, 0, 0),
                                          bq(0, 0, 0), bk(0, 0, 0), bv(0, 0, 0), bo(0, 0, 0),
                                          relative_position_bias_table(0, 0, 0) { LoadImpl(values); }

    // Public interface
    int GetWindowSize() const { return window_size; }
    int GetNumHeads() const { return num_heads; }
    int GetInputDim() const { return input_dim; }
};

// Swin Transformer Block
class SwinTransformerBlockCRTP : public LayerBaseCRTP<SwinTransformerBlockCRTP> {
private:
    friend class LayerBaseCRTP<SwinTransformerBlockCRTP>;

    // Core components
    int dim;                    // Input dimension
    int input_resolution[2];    // [height, width] of input feature map
    int num_heads;              // Number of attention heads
    int window_size;            // Window size for window attention
    int shift_size;             // Shift size for shifted windows (0 or window_size/2)
    int mlp_ratio;              // Ratio of MLP hidden dim to embedding dim
    bool mlp_bias;              // Use bias in MLP
    double mlp_dropout;         // Dropout rate in MLP

    // Sub-layers
    WindowAttentionCRTP window_attn;
    FullyConnLayerCRTP feed_forward;  // MLP layer
    LayerNormCRTP norm1;              // First layer norm
    LayerNormCRTP norm2;              // Second layer norm
    DropOutLayerCRTP dropout1;        // Dropouts
    DropOutLayerCRTP dropout2;

    // Cached values
    Volume output_activation;
    Volume input_activation;

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "swin_transformer_block"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }

public:
    SwinTransformerBlockCRTP(int dim, int input_resolution[2], int num_heads, int window_size=7,
                            int shift_size=0, int mlp_ratio=4, bool mlp_bias=true, double mlp_dropout=0.0);
    SwinTransformerBlockCRTP(ValueMap values) : window_attn(7, 0, 0), feed_forward(0),
                                               norm1(), norm2(), dropout1(0.0), dropout2(0.0) { LoadImpl(values); }

    // Public interface
    int GetDim() const { return dim; }
    int GetNumHeads() const { return num_heads; }
    int GetWindowSize() const { return window_size; }
    int GetShiftSize() const { return shift_size; }
};

// Masked Multi-Head Attention Layer for BERT
class MaskedMultiHeadAttentionCRTP : public LayerBaseCRTP<MaskedMultiHeadAttentionCRTP> {
private:
    friend class LayerBaseCRTP<MaskedMultiHeadAttentionCRTP>;

    // Core data (inherits from MultiHeadAttentionCRTP)
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
    Volume mask;        // Attention mask (for masking padded tokens)

    // Internal implementation methods
    Volume& ForwardImpl(Volume& input, bool is_training);
    void BackwardImpl();
    void InitImpl(int input_width, int input_height, int input_depth);
    Vector<ParametersAndGradients>& GetParametersAndGradientsImpl();
    String GetKeyImpl() const { return "masked_multihead_attention"; }
    void StoreImpl(ValueMap& map) const;
    void LoadImpl(const ValueMap& map);
    String ToStringImpl() const;
    Volume& GetOutputImpl() { return output_activation; }

public:
    MaskedMultiHeadAttentionCRTP(int embed_dim, int num_heads);
    MaskedMultiHeadAttentionCRTP(ValueMap values) : wq(0, 0, 0), wk(0, 0, 0), wv(0, 0, 0), wo(0, 0, 0),
                                                   bq(0, 0, 0), bk(0, 0, 0), bv(0, 0, 0), bo(0, 0, 0),
                                                   mask(0, 0, 0) { LoadImpl(values); }

    // Public interface
    int GetEmbedDim() const { return embed_dim; }
    int GetNumHeads() const { return num_heads; }
    int GetHeadDim() const { return head_dim; }

    // Set attention mask
    void SetMask(const Volume& mask_volume) { mask = mask_volume; }
};

} // namespace ConvNet

#endif