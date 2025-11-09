#ifndef _ConvNet_GptLayers_h_
#define _ConvNet_GptLayers_h_

#include "ConvNet.h"
#include "TransformerLayers.h"

namespace ConvNet {

// GPT Model - Autoregressive Transformer Language Model
class GPTModel {
private:
    // Model configuration
    int vocab_size;
    int embed_dim;
    int num_heads;
    int num_layers;
    int ff_dim;
    int max_seq_len;
    double dropout_rate;

    // Core components (GPT uses decoder-only architecture with masked self-attention)
    Vector<DecoderLayerCRTP> decoder_layers;  // Multiple decoder layers (though simplified for GPT)
    PositionalEncodingCRTP positional_encoding;  // For positional information
    
    // Embedding layers
    Volume token_embeddings;  // Input token embeddings
    Volume output_weights;    // Output projection weights (often tied to input embeddings)

    // Cached values
    Vector<ParametersAndGradients> parameters;

public:
    GPTModel(int vocab_size, int embed_dim, int num_heads, 
             int num_layers, int ff_dim, int max_seq_len, 
             double dropout_rate = 0.1);
    
    // Forward pass for training (with teacher forcing)
    Volume& Forward(Volume& input_tokens, bool is_training = false);
    
    // Forward pass for generation (autoregressive)
    Volume& Generate(Volume& context, int max_new_tokens);
    
    // Get logits for next token prediction
    Volume& GetNextTokenLogits(const Vector<int>& context);
    
    // Sample next token from logits
    int SampleNextToken(const Vector<double>& logits, double temperature = 1.0, 
                       bool top_k = false, int k = 50, bool nucleus = false, double p = 0.9);
    
    // Get parameters for training
    Vector<ParametersAndGradients>& GetParameters();
    
    // Serialization
    void Store(ValueMap& map) const;
    void Load(const ValueMap& map);
    
    // Public interface
    int GetVocabSize() const { return vocab_size; }
    int GetEmbedDim() const { return embed_dim; }
    int GetNumHeads() const { return num_heads; }
    int GetNumLayers() const { return num_layers; }
    
    // Tokenization utilities (simplified)
    Vector<int> Tokenize(const String& text);
    String Detokenize(const Vector<int>& tokens);
    
private:
    // Helper functions
    Volume& PrepareInputs(const Vector<int>& token_ids);
    Volume& ApplyCausalMask(Volume& attention_scores);
    void UpdateOutputWeights();  // Tie output weights with input embeddings
};

// GPT Session - wraps GPT model with training utilities
class GPTSession {
private:
    std::unique_ptr<GPTModel> model;
    
    // Training parameters
    double learning_rate;
    int batch_size;
    int sequence_length;
    
    // Optimization
    // In a real implementation, we'd have an optimizer here

public:
    GPTSession(std::unique_ptr<GPTModel> gpt_model);
    
    // Training methods
    void TrainBatch(const Vector<Volume>& inputs, const Vector<Volume>& targets);
    double ComputeLoss(Volume& predictions, Volume& targets);
    
    // Generation methods
    Vector<int> GenerateText(const Vector<int>& context, int max_tokens, 
                            double temperature = 1.0, bool top_k = false, 
                            int k = 50, bool nucleus = false, double p = 0.9);
    
    // Helper methods
    double GetPerplexity(const Vector<int>& test_tokens);
    Vector<double> GetEmbeddings(const Vector<int>& tokens);
    
    // Accessor
    GPTModel& GetModel() { return *model; }
    const GPTModel& GetModel() const { return *model; }
};

// Helper function to create a GPT model
std::unique_ptr<GPTModel> CreateGPT(int vocab_size, int embed_dim, int num_heads, 
                                   int num_layers, int ff_dim, int max_seq_len, 
                                   double dropout_rate = 0.1);

// Helper function to create a GPT session
std::unique_ptr<GPTSession> CreateGPTSession(int vocab_size, int embed_dim, 
                                            int num_heads, int num_layers, 
                                            int ff_dim, int max_seq_len, 
                                            double dropout_rate = 0.1,
                                            double learning_rate = 3e-4);

} // namespace ConvNet

#endif