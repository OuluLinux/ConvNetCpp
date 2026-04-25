#ifndef _ConvNet_TransformerLayers_h_
#define _ConvNet_TransformerLayers_h_

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

    // Copy constructor
    MultiHeadAttentionCRTP(const MultiHeadAttentionCRTP& other)
        : embed_dim(other.embed_dim), num_heads(other.num_heads), head_dim(other.head_dim),
          wq(other.wq), wk(other.wk), wv(other.wv), wo(other.wo),
          bq(other.bq), bk(other.bk), bv(other.bv), bo(other.bo),
          output_activation(other.output_activation), input_activation(other.input_activation),
          scores(other.scores), attention_weights(other.attention_weights), output(other.output) {
        // Explicitly copy Vector members using deep copy
        for(int i = 0; i < other.queries.GetCount(); i++) {
            queries.Add() = other.queries[i];
        }
        for(int i = 0; i < other.keys.GetCount(); i++) {
            keys.Add() = other.keys[i];
        }
        for(int i = 0; i < other.values.GetCount(); i++) {
            values.Add() = other.values[i];
        }
        for(int i = 0; i < other.attention_scores.GetCount(); i++) {
            attention_scores.Add() = other.attention_scores[i];
        }
        for(int i = 0; i < other.attention_outputs.GetCount(); i++) {
            attention_outputs.Add() = other.attention_outputs[i];
        }
    }

    // Copy assignment operator
    MultiHeadAttentionCRTP& operator=(const MultiHeadAttentionCRTP& other) {
        if (this != &other) {
            embed_dim = other.embed_dim;
            num_heads = other.num_heads;
            head_dim = other.head_dim;
            wq = other.wq;
            wk = other.wk;
            wv = other.wv;
            wo = other.wo;
            bq = other.bq;
            bk = other.bk;
            bv = other.bv;
            bo = other.bo;
            output_activation = other.output_activation;
            input_activation = other.input_activation;
            scores = other.scores;
            attention_weights = other.attention_weights;
            output = other.output;

            // Explicitly copy Vector members using deep copy
            queries.Clear();
            for(int i = 0; i < other.queries.GetCount(); i++) {
                queries.Add() = other.queries[i];
            }
            keys.Clear();
            for(int i = 0; i < other.keys.GetCount(); i++) {
                keys.Add() = other.keys[i];
            }
            values.Clear();
            for(int i = 0; i < other.values.GetCount(); i++) {
                values.Add() = other.values[i];
            }
            attention_scores.Clear();
            for(int i = 0; i < other.attention_scores.GetCount(); i++) {
                attention_scores.Add() = other.attention_scores[i];
            }
            attention_outputs.Clear();
            for(int i = 0; i < other.attention_outputs.GetCount(); i++) {
                attention_outputs.Add() = other.attention_outputs[i];
            }
        }
        return *this;
    }

    // Public interface
    int GetEmbedDim() const { return embed_dim; }
    int GetNumHeads() const { return num_heads; }
    int GetHeadDim() const { return head_dim; }

    // Scaled Dot-Product Attention helper
    Volume& ScaledDotProductAttention(Volume& query, Volume& key, Volume& value,
                                     const Volume* mask = nullptr);

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % embed_dim % num_heads % head_dim;
        s % wq % wk % wv % wo;
        s % bq % bk % bv % bo;
        s % output_activation % input_activation;
        s % queries % keys % values % attention_scores % attention_outputs;
        s % scores % attention_weights % output;
    }

    typedef MultiHeadAttentionCRTP CLASSNAME;
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
    // Default constructor - initializes with default values
    EncoderLayerCRTP() : self_attention(0, 0), feed_forward(0), dropout1(0.0), dropout2(0.0) { }

    EncoderLayerCRTP(int embed_dim, int num_heads, int ff_dim, double dropout_rate = 0.1)
        : self_attention(embed_dim, num_heads),
          feed_forward(ff_dim),  // Assuming this takes neuron_count as parameter
          dropout1(dropout_rate), dropout2(dropout_rate) {
        // Initialize layer normalization parameters
        norm1_weights.Init(embed_dim, 1, 1);
        norm1_biases.Init(embed_dim, 1, 1);
        norm2_weights.Init(embed_dim, 1, 1);
        norm2_biases.Init(embed_dim, 1, 1);

        // Initialize to appropriate values for layer normalization
        norm1_weights.SetConst(1.0);
        norm1_biases.SetConst(0.0);
        norm2_weights.SetConst(1.0);
        norm2_biases.SetConst(0.0);
    }
    EncoderLayerCRTP(ValueMap values) : self_attention(0, 0), feed_forward(0), dropout1(0.0), dropout2(0.0) { LoadImpl(values); }

    // Copy constructor
    EncoderLayerCRTP(const EncoderLayerCRTP& other)
        : self_attention(other.self_attention), feed_forward(other.feed_forward),
          norm1_weights(other.norm1_weights), norm1_biases(other.norm1_biases),
          norm2_weights(other.norm2_weights), norm2_biases(other.norm2_biases),
          dropout1(other.dropout1), dropout2(other.dropout2),
          dropout_rate(other.dropout_rate),
          output_activation(other.output_activation), input_activation(other.input_activation) {
    }

    // Copy assignment operator
    EncoderLayerCRTP& operator=(const EncoderLayerCRTP& other) {
        if (this != &other) {
            self_attention = other.self_attention;
            feed_forward = other.feed_forward;
            norm1_weights = other.norm1_weights;
            norm1_biases = other.norm1_biases;
            norm2_weights = other.norm2_weights;
            norm2_biases = other.norm2_biases;
            dropout1 = other.dropout1;
            dropout2 = other.dropout2;
            dropout_rate = other.dropout_rate;
            output_activation = other.output_activation;
            input_activation = other.input_activation;
        }
        return *this;
    }

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % self_attention % feed_forward % dropout1 % dropout2
          % norm1_weights % norm1_biases % norm2_weights % norm2_biases;
    }

    // Public interface
    int GetEmbedDim() const { return self_attention.GetEmbedDim(); }
    int GetNumHeads() const { return self_attention.GetNumHeads(); }

    // Helper for layer normalization
    void ApplyLayerNorm(Volume& input, const Volume& gamma, const Volume& beta, int d_model, int seq_len);

    typedef EncoderLayerCRTP CLASSNAME;
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
    // Default constructor - initializes with default values
    DecoderLayerCRTP() : self_attention(0, 0), cross_attention(0, 0), feed_forward(0), dropout1(0.0), dropout2(0.0), dropout3(0.0) { }

    DecoderLayerCRTP(int embed_dim, int num_heads, int ff_dim, double dropout_rate = 0.1)
        : self_attention(embed_dim, num_heads),
          cross_attention(embed_dim, num_heads),  // Cross attention with encoder
          feed_forward(ff_dim),  // Assuming this takes neuron_count as parameter
          dropout1(dropout_rate), dropout2(dropout_rate), dropout3(dropout_rate) {
        // Initialize layer normalization parameters
        norm1_weights.Init(embed_dim, 1, 1);
        norm1_biases.Init(embed_dim, 1, 1);
        norm2_weights.Init(embed_dim, 1, 1);
        norm2_biases.Init(embed_dim, 1, 1);
        norm3_weights.Init(embed_dim, 1, 1);
        norm3_biases.Init(embed_dim, 1, 1);

        // Initialize to appropriate values
        norm1_weights.SetConst(1.0);
        norm1_biases.SetConst(0.0);
        norm2_weights.SetConst(1.0);
        norm2_biases.SetConst(0.0);
        norm3_weights.SetConst(1.0);
        norm3_biases.SetConst(0.0);
    }
    DecoderLayerCRTP(ValueMap values) : self_attention(0, 0), cross_attention(0, 0), feed_forward(0), dropout1(0.0), dropout2(0.0), dropout3(0.0) { LoadImpl(values); }

    // Copy constructor
    DecoderLayerCRTP(const DecoderLayerCRTP& other)
        : self_attention(other.self_attention), cross_attention(other.cross_attention), feed_forward(other.feed_forward),
          norm1_weights(other.norm1_weights), norm1_biases(other.norm1_biases),
          norm2_weights(other.norm2_weights), norm2_biases(other.norm2_biases),
          norm3_weights(other.norm3_weights), norm3_biases(other.norm3_biases),
          dropout1(other.dropout1), dropout2(other.dropout2), dropout3(other.dropout3),
          dropout_rate(other.dropout_rate),
          output_activation(other.output_activation), input_activation(other.input_activation) {
    }

    // Copy assignment operator
    DecoderLayerCRTP& operator=(const DecoderLayerCRTP& other) {
        if (this != &other) {
            self_attention = other.self_attention;
            cross_attention = other.cross_attention;
            feed_forward = other.feed_forward;
            norm1_weights = other.norm1_weights;
            norm1_biases = other.norm1_biases;
            norm2_weights = other.norm2_weights;
            norm2_biases = other.norm2_biases;
            norm3_weights = other.norm3_weights;
            norm3_biases = other.norm3_biases;
            dropout1 = other.dropout1;
            dropout2 = other.dropout2;
            dropout3 = other.dropout3;
            dropout_rate = other.dropout_rate;
            output_activation = other.output_activation;
            input_activation = other.input_activation;
        }
        return *this;
    }

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % self_attention % cross_attention % feed_forward % dropout1 % dropout2 % dropout3
          % norm1_weights % norm1_biases % norm2_weights % norm2_biases
          % norm3_weights % norm3_biases;
    }

    // Public interface
    int GetEmbedDim() const { return self_attention.GetEmbedDim(); }
    int GetNumHeads() const { return self_attention.GetNumHeads(); }

    // Helper for layer normalization
    void ApplyLayerNorm(Volume& input, const Volume& gamma, const Volume& beta, int d_model, int seq_len);

    typedef DecoderLayerCRTP CLASSNAME;
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

    // Serialization support for U++ containers
    void Serialize(Stream& s) {
        s % max_len % embed_dim % pe;
    }

    typedef PositionalEncodingCRTP CLASSNAME;
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

    // Output activation for forward pass
    Volume output_activation;

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

    typedef TransformerCRTP CLASSNAME;
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

    // Copy constructor
    ViTPatchEmbeddingCRTP(const ViTPatchEmbeddingCRTP& other)
        : patch_size(other.patch_size), embed_dim(other.embed_dim), num_patches(other.num_patches),
          proj_weight(other.proj_weight), proj_bias(other.proj_bias), pos_embed(other.pos_embed),
          output_activation(other.output_activation), input_activation(other.input_activation) {
    }

    // Copy assignment operator
    ViTPatchEmbeddingCRTP& operator=(const ViTPatchEmbeddingCRTP& other) {
        if (this != &other) {
            patch_size = other.patch_size;
            embed_dim = other.embed_dim;
            num_patches = other.num_patches;
            proj_weight = other.proj_weight;
            proj_bias = other.proj_bias;
            pos_embed = other.pos_embed;
            output_activation = other.output_activation;
            input_activation = other.input_activation;
        }
        return *this;
    }

    // Public interface
    int GetPatchSize() const { return patch_size; }
    int GetEmbedDim() const { return embed_dim; }
    int GetNumPatches() const { return num_patches; }

    // Helper to create patches from input image
    Volume CreatePatches(const Volume& input);

    typedef ViTPatchEmbeddingCRTP CLASSNAME;
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

    // Copy constructor
    ViTEncoderCRTP(const ViTEncoderCRTP& other)
        : embed_dim(other.embed_dim), num_heads(other.num_heads), ff_dim(other.ff_dim),
          num_layers(other.num_layers), dropout_rate(other.dropout_rate),
          class_token(other.class_token), class_token_expanded(other.class_token_expanded),
          output_activation(other.output_activation), input_activation(other.input_activation) {
        // Explicitly copy Vector members
        for(int i = 0; i < other.encoder_layers.GetCount(); i++) {
            encoder_layers.Add() = other.encoder_layers[i];
        }
    }

    // Copy assignment operator
    ViTEncoderCRTP& operator=(const ViTEncoderCRTP& other) {
        if (this != &other) {
            embed_dim = other.embed_dim;
            num_heads = other.num_heads;
            ff_dim = other.ff_dim;
            num_layers = other.num_layers;
            dropout_rate = other.dropout_rate;
            class_token = other.class_token;
            class_token_expanded = other.class_token_expanded;
            output_activation = other.output_activation;
            input_activation = other.input_activation;

            // Explicitly copy Vector members
            encoder_layers.Clear();
            for(int i = 0; i < other.encoder_layers.GetCount(); i++) {
                encoder_layers.Add() = other.encoder_layers[i];
            }
        }
        return *this;
    }

    // Public interface
    int GetEmbedDim() const { return embed_dim; }
    int GetNumHeads() const { return num_heads; }
    int GetFFDim() const { return ff_dim; }
    int GetNumLayers() const { return num_layers; }

    typedef ViTEncoderCRTP CLASSNAME;
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

    // Copy constructor
    ViTClassifierCRTP(const ViTClassifierCRTP& other)
        : num_classes(other.num_classes), embed_dim(other.embed_dim),
          classifier_weight(other.classifier_weight), classifier_bias(other.classifier_bias),
          output_activation(other.output_activation), input_activation(other.input_activation) {
    }

    // Copy assignment operator
    ViTClassifierCRTP& operator=(const ViTClassifierCRTP& other) {
        if (this != &other) {
            num_classes = other.num_classes;
            embed_dim = other.embed_dim;
            classifier_weight = other.classifier_weight;
            classifier_bias = other.classifier_bias;
            output_activation = other.output_activation;
            input_activation = other.input_activation;
        }
        return *this;
    }

    // Public interface
    int GetNumClasses() const { return num_classes; }
    int GetEmbedDim() const { return embed_dim; }

    typedef ViTClassifierCRTP CLASSNAME;
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

    // Copy constructor
    SwinPatchMergingCRTP(const SwinPatchMergingCRTP& other)
        : dim(other.dim), out_dim(other.out_dim),
          reduction_weight(other.reduction_weight), reduction_bias(other.reduction_bias),
          output_activation(other.output_activation), input_activation(other.input_activation) {
    }

    // Copy assignment operator
    SwinPatchMergingCRTP& operator=(const SwinPatchMergingCRTP& other) {
        if (this != &other) {
            dim = other.dim;
            out_dim = other.out_dim;
            reduction_weight = other.reduction_weight;
            reduction_bias = other.reduction_bias;
            output_activation = other.output_activation;
            input_activation = other.input_activation;
        }
        return *this;
    }

    // Public interface
    int GetInputDim() const { return dim; }
    int GetOutputDim() const { return out_dim; }

    typedef SwinPatchMergingCRTP CLASSNAME;
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

    // Copy constructor
    WindowAttentionCRTP(const WindowAttentionCRTP& other)
        : window_size(other.window_size), num_heads(other.num_heads), head_dim(other.head_dim), input_dim(other.input_dim),
          wq(other.wq), wk(other.wk), wv(other.wv), wo(other.wo),
          bq(other.bq), bk(other.bk), bv(other.bv), bo(other.bo),
          relative_position_bias_table(other.relative_position_bias_table),
          output_activation(other.output_activation), input_activation(other.input_activation) {
        // Copy the relative position index array
        for(int i = 0; i < 49; i++) {
            for(int j = 0; j < 49; j++) {
                relative_position_index[i][j] = other.relative_position_index[i][j];
            }
        }
    }

    // Copy assignment operator
    WindowAttentionCRTP& operator=(const WindowAttentionCRTP& other) {
        if (this != &other) {
            window_size = other.window_size;
            num_heads = other.num_heads;
            head_dim = other.head_dim;
            input_dim = other.input_dim;
            wq = other.wq;
            wk = other.wk;
            wv = other.wv;
            wo = other.wo;
            bq = other.bq;
            bk = other.bk;
            bv = other.bv;
            bo = other.bo;
            relative_position_bias_table = other.relative_position_bias_table;
            output_activation = other.output_activation;
            input_activation = other.input_activation;

            // Copy the relative position index array
            for(int i = 0; i < 49; i++) {
                for(int j = 0; j < 49; j++) {
                    relative_position_index[i][j] = other.relative_position_index[i][j];
                }
            }
        }
        return *this;
    }

    // Public interface
    int GetWindowSize() const { return window_size; }
    int GetNumHeads() const { return num_heads; }
    int GetInputDim() const { return input_dim; }

    typedef WindowAttentionCRTP CLASSNAME;
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

    // Copy constructor
    SwinTransformerBlockCRTP(const SwinTransformerBlockCRTP& other)
        : dim(other.dim), num_heads(other.num_heads), window_size(other.window_size),
          shift_size(other.shift_size), mlp_ratio(other.mlp_ratio), mlp_bias(other.mlp_bias),
          mlp_dropout(other.mlp_dropout),
          window_attn(other.window_attn), feed_forward(other.feed_forward),
          norm1(other.norm1), norm2(other.norm2),
          dropout1(other.dropout1), dropout2(other.dropout2),
          output_activation(other.output_activation), input_activation(other.input_activation) {
        input_resolution[0] = other.input_resolution[0];
        input_resolution[1] = other.input_resolution[1];
    }

    // Copy assignment operator
    SwinTransformerBlockCRTP& operator=(const SwinTransformerBlockCRTP& other) {
        if (this != &other) {
            dim = other.dim;
            num_heads = other.num_heads;
            window_size = other.window_size;
            shift_size = other.shift_size;
            mlp_ratio = other.mlp_ratio;
            mlp_bias = other.mlp_bias;
            mlp_dropout = other.mlp_dropout;
            window_attn = other.window_attn;
            feed_forward = other.feed_forward;
            norm1 = other.norm1;
            norm2 = other.norm2;
            dropout1 = other.dropout1;
            dropout2 = other.dropout2;
            output_activation = other.output_activation;
            input_activation = other.input_activation;
            input_resolution[0] = other.input_resolution[0];
            input_resolution[1] = other.input_resolution[1];
        }
        return *this;
    }

    // Public interface
    int GetDim() const { return dim; }
    int GetNumHeads() const { return num_heads; }
    int GetWindowSize() const { return window_size; }
    int GetShiftSize() const { return shift_size; }

    typedef SwinTransformerBlockCRTP CLASSNAME;
};

// Masked Multi-Head Attention Layer for BERT
class MaskedMultiHeadAttentionCRTP : public LayerBaseCRTP<MaskedMultiHeadAttentionCRTP> {
private:
    friend class LayerBaseCRTP<MaskedMultiHeadAttentionCRTP>;

    // Core data (inherits from MultiHeadAttentionCRTP)
    int embed_dim;      // Total embedding dimension
    int num_heads;      // Number of attention heads
    int head_dim;       // Dimension per head (embed_dim / num_heads)
    int input_depth;    // Input depth dimension (needed for weight matrix dimensions)

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
                                                   input_depth(0), mask(0, 0, 0) { LoadImpl(values); }

    // Copy constructor
    MaskedMultiHeadAttentionCRTP(const MaskedMultiHeadAttentionCRTP& other)
        : embed_dim(other.embed_dim), num_heads(other.num_heads), head_dim(other.head_dim), input_depth(other.input_depth),
          wq(other.wq), wk(other.wk), wv(other.wv), wo(other.wo),
          bq(other.bq), bk(other.bk), bv(other.bv), bo(other.bo),
          output_activation(other.output_activation), input_activation(other.input_activation),
          mask(other.mask) {
    }

    // Copy assignment operator
    MaskedMultiHeadAttentionCRTP& operator=(const MaskedMultiHeadAttentionCRTP& other) {
        if (this != &other) {
            embed_dim = other.embed_dim;
            num_heads = other.num_heads;
            head_dim = other.head_dim;
            input_depth = other.input_depth;
            wq = other.wq;
            wk = other.wk;
            wv = other.wv;
            wo = other.wo;
            bq = other.bq;
            bk = other.bk;
            bv = other.bv;
            bo = other.bo;
            output_activation = other.output_activation;
            input_activation = other.input_activation;
            mask = other.mask;
        }
        return *this;
    }

    // Public interface
    int GetEmbedDim() const { return embed_dim; }
    int GetNumHeads() const { return num_heads; }
    int GetHeadDim() const { return head_dim; }

    // Set attention mask
    void SetMask(const Volume& mask_volume) { mask = mask_volume; }

    typedef MaskedMultiHeadAttentionCRTP CLASSNAME;
};
} // namespace ConvNet

namespace Upp {
  template<> inline constexpr bool is_trivially_relocatable<ConvNet::MultiHeadAttentionCRTP> = true;
  template<> inline constexpr bool is_trivially_relocatable<ConvNet::EncoderLayerCRTP> = true;
  template<> inline constexpr bool is_trivially_relocatable<ConvNet::DecoderLayerCRTP> = true;
}

#endif
