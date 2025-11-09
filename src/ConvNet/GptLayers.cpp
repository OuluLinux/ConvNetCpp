#include "GptLayers.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace ConvNet {

GPTModel::GPTModel(int vocab_size, int embed_dim, int num_heads, 
                   int num_layers, int ff_dim, int max_seq_len, 
                   double dropout_rate)
    : vocab_size(vocab_size), embed_dim(embed_dim), num_heads(num_heads),
      num_layers(num_layers), ff_dim(ff_dim), max_seq_len(max_seq_len),
      dropout_rate(dropout_rate) {
    
    // Initialize the transformer with equal source and target vocab sizes
    // For GPT, source and target are the same (autoregressive language modeling)
    transformer = CreateTransformer(vocab_size, vocab_size, embed_dim, num_heads,
                                   num_layers, num_layers, ff_dim, max_seq_len, 
                                   dropout_rate);
    
    // Initialize output weights (tied with input embeddings in many GPT implementations)
    output_weights.Init(vocab_size, embed_dim, 1);
    output_weights.FillRandom();
}

Volume& GPTModel::Forward(Volume& input_tokens, bool is_training) {
    // For GPT, we pass the same sequence as both source and target
    // This implements the causal language modeling objective
    
    // In a real implementation, we would:
    // 1. Create causal mask to prevent attending to future tokens
    // 2. Pass input through the transformer
    // 3. Apply final linear layer to convert embeddings to logits
    
    // For now, we'll use the base transformer forward
    // Note: This is simplified - real GPT would have specific handling for causal masking
    return transformer->Forward(input_tokens, input_tokens, is_training);
}

Volume& GPTModel::Generate(Volume& context, int max_new_tokens) {
    // Autoregressive generation: generate one token at a time
    // This is a simplified version - a full implementation would be more complex
    
    Volume current_context = context;  // Copy the input context
    
    for (int i = 0; i < max_new_tokens; i++) {
        // Get next token logits
        Volume& logits = GetNextTokenLogits({});  // This would need proper implementation
        
        // Sample next token
        // This would sample from the logits distribution
    }
    
    // Return the completed sequence
    return current_context;
}

Volume& GPTModel::GetNextTokenLogits(const Vector<int>& context) {
    // Convert context tokens to volume
    Volume context_volume = PrepareInputs(context);
    
    // Forward through transformer
    Volume& embeddings = transformer->Forward(context_volume, context_volume, false);
    
    // Apply final linear projection to vocab size
    // This would be a matrix multiplication: embeddings * output_weights^T
    // For now, return the embeddings as placeholder
    return embeddings;
}

int GPTModel::SampleNextToken(const Vector<double>& logits, double temperature, 
                              bool top_k, int k, bool nucleus, double p) {
    // Apply temperature scaling
    Vector<double> scaled_logits = logits;
    for (int i = 0; i < scaled_logits.GetCount(); i++) {
        scaled_logits[i] /= temperature;
    }
    
    // Apply softmax to get probabilities
    Vector<double> probs(scaled_logits.GetCount());
    double max_logit = scaled_logits[0];
    for (int i = 1; i < scaled_logits.GetCount(); i++) {
        if (scaled_logits[i] > max_logit) max_logit = scaled_logits[i];
    }
    
    // Subtract max for numerical stability
    double sum = 0.0;
    for (int i = 0; i < scaled_logits.GetCount(); i++) {
        probs[i] = exp(scaled_logits[i] - max_logit);
        sum += probs[i];
    }
    
    // Normalize
    for (int i = 0; i < probs.GetCount(); i++) {
        probs[i] /= sum;
    }
    
    // Apply top-k or nucleus sampling if requested
    if (top_k && k > 0) {
        // Get indices sorted by probability
        Vector<int> indices(probs.GetCount());
        for (int i = 0; i < indices.GetCount(); i++) indices[i] = i;
        
        // Sort indices by probability (descending)
        std::sort(indices.Begin(), indices.End(), 
                  [&probs](int a, int b) { return probs[a] > probs[b]; });
        
        // Zero out probabilities after top-k
        for (int i = k; i < probs.GetCount(); i++) {
            probs[indices[i]] = 0.0;
        }
        
        // Re-normalize
        sum = 0.0;
        for (int i = 0; i < probs.GetCount(); i++) {
            sum += probs[i];
        }
        for (int i = 0; i < probs.GetCount(); i++) {
            probs[i] /= sum;
        }
    }
    
    if (nucleus && p > 0.0 && p < 1.0) {
        // Sort indices by probability (descending)
        Vector<int> indices(probs.GetCount());
        for (int i = 0; i < indices.GetCount(); i++) indices[i] = i;
        
        std::sort(indices.Begin(), indices.End(), 
                  [&probs](int a, int b) { return probs[a] > probs[b]; });
        
        // Find top-p (nucleus) tokens
        double cumsum = 0.0;
        int cutoff_idx = 0;
        for (int i = 0; i < indices.GetCount(); i++) {
            cumsum += probs[indices[i]];
            if (cumsum >= p) {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Zero out probabilities after the cutoff
        for (int i = cutoff_idx; i < probs.GetCount(); i++) {
            probs[indices[i]] = 0.0;
        }
        
        // Re-normalize
        sum = 0.0;
        for (int i = 0; i < probs.GetCount(); i++) {
            sum += probs[i];
        }
        for (int i = 0; i < probs.GetCount(); i++) {
            probs[i] /= sum;
        }
    }
    
    // Sample from the probability distribution
    double rand_val = Randomf();
    double cumsum = 0.0;
    for (int i = 0; i < probs.GetCount(); i++) {
        cumsum += probs[i];
        if (rand_val <= cumsum) {
            return i;
        }
    }
    
    // Fallback (shouldn't happen with proper normalization)
    return probs.GetCount() - 1;
}

Vector<ParametersAndGradients>& GPTModel::GetParameters() {
    // Get transformer parameters
    parameters = transformer->GetParametersAndGradients();
    
    // Add output weights
    ParametersAndGradients output_weights_grad;
    output_weights_grad.volume = &output_weights;
    parameters.Add() = output_weights_grad;
    
    return parameters;
}

void GPTModel::Store(ValueMap& map) const {
    map.GetAdd("vocab_size") = vocab_size;
    map.GetAdd("embed_dim") = embed_dim;
    map.GetAdd("num_heads") = num_heads;
    map.GetAdd("num_layers") = num_layers;
    map.GetAdd("ff_dim") = ff_dim;
    map.GetAdd("max_seq_len") = max_seq_len;
    map.GetAdd("dropout_rate") = dropout_rate;
    
    // Store transformer parameters
    transformer->Store(map);
    
    // Store output weights
    // This would require a serialization method for Volume
}

void GPTModel::Load(const ValueMap& map) {
    vocab_size = map.GetValue(map.Find("vocab_size"));
    embed_dim = map.GetValue(map.Find("embed_dim"));
    num_heads = map.GetValue(map.Find("num_heads"));
    num_layers = map.GetValue(map.Find("num_layers"));
    ff_dim = map.GetValue(map.Find("ff_dim"));
    max_seq_len = map.GetValue(map.Find("max_seq_len"));
    dropout_rate = map.GetValue(map.Find("dropout_rate"));
    
    // Recreate the transformer with loaded parameters
    transformer = CreateTransformer(vocab_size, vocab_size, embed_dim, num_heads,
                                   num_layers, num_layers, ff_dim, max_seq_len, 
                                   dropout_rate);
    
    // Load transformer parameters
    transformer->Load(map);
    
void GPTModel::Serialize(Stream& s) {
    // Serialize model configuration
    s % vocab_size;
    s % embed_dim;
    s % num_heads;
    s % num_layers;
    s % ff_dim;
    s % max_seq_len;
    s % dropout_rate;
    
    // Serialize core components
    if (transformer) {
        s % (int)1;  // flag indicating transformer exists
        transformer->Serialize(s);
    } else {
        s % (int)0;  // flag indicating no transformer
    }
    
    // Serialize embedding layers
    s % token_embeddings;
    s % output_weights;
}

    // Load output weights
    // This would require a deserialization method for Volume
}

Vector<int> GPTModel::Tokenize(const String& text) {
    // Simplified tokenization - in practice, this would use a proper tokenizer
    // like BPE, WordPiece, etc.
    
    Vector<int> tokens;
    
    // For demo purposes, we'll just convert characters to their ASCII values
    // In a real implementation, you'd use a vocabulary lookup
    for (int i = 0; i < text.GetCount(); i++) {
        int token_id = (int)text[i];
        // Clamp to vocab size to avoid out-of-bounds
        token_id = min(token_id, vocab_size - 1);
        tokens.Add(token_id);
    }
    
    return tokens;
}

String GPTModel::Detokenize(const Vector<int>& tokens) {
    // Simplified detokenization
    String text;
    
    for (int i = 0; i < tokens.GetCount(); i++) {
        // Convert token ID back to character (for demo)
        char c = (char)min(255, max(0, tokens[i]));  // Clamp to valid ASCII range
        text.Cat(c);
    }
    
    return text;
}

Volume& GPTModel::PrepareInputs(const Vector<int>& token_ids) {
    // Convert token IDs to input volume
    // This would typically involve embedding lookup
    static Volume input_volume;
    
    // Initialize as [embed_dim, 1, sequence_length]
    input_volume.Init(embed_dim, 1, token_ids.GetCount());
    
    // In a real implementation, this would look up embeddings for each token
    // For now, we'll just fill with dummy values
    for (int pos = 0; pos < token_ids.GetCount(); pos++) {
        for (int dim = 0; dim < embed_dim; dim++) {
            // This is a placeholder - would use actual embeddings in practice
            input_volume.Set(dim, 0, pos, (double)token_ids[pos] / vocab_size);
        }
    }
    
    return input_volume;
}

GPTSession::GPTSession(std::unique_ptr<GPTModel> gpt_model) 
    : model(std::move(gpt_model)), learning_rate(3e-4), batch_size(16), sequence_length(256) {
}

void GPTSession::TrainBatch(const Vector<Volume>& inputs, const Vector<Volume>& targets) {
    // In a real implementation, this would:
    // 1. Forward pass through the model
    // 2. Compute loss against targets
    // 3. Backward pass to compute gradients
    // 4. Update parameters using optimizer
    
    // For now, this is a placeholder
    for (int i = 0; i < inputs.GetCount(); i++) {
        Volume& output = model->Forward(inputs[i], true);
        double loss = ComputeLoss(output, targets[i]);
        // In a complete implementation, we would perform backprop and parameter updates here
    }
}

double GPTSession::ComputeLoss(Volume& predictions, Volume& targets) {
    // Compute cross-entropy loss between predictions and targets
    // This is a simplified implementation
    
    int total_elements = predictions.GetSize();
    if (total_elements != targets.GetSize()) {
        throw std::runtime_error("Prediction and target dimensions don't match");
    }
    
    double loss = 0.0;
    for (int i = 0; i < total_elements; i++) {
        double pred = predictions.Get(i);
        double target = targets.Get(i);
        
        // Cross-entropy loss: -sum(target * log(prediction))
        // This is simplified - proper implementation would handle probability distributions
        double element_loss = -target * log(max(pred, 1e-8)); // Clamp to prevent log(0)
        loss += element_loss;
    }
    
    return loss / total_elements;
}

Vector<int> GPTSession::GenerateText(const Vector<int>& context, int max_tokens, 
                                    double temperature, bool top_k, int k, 
                                    bool nucleus, double p) {
    // Generate text autoregressively
    Vector<int> current_context = context;
    
    for (int i = 0; i < max_tokens; i++) {
        // Get logits for next token
        Volume& logits_volume = model->GetNextTokenLogits(current_context);
        
        // Convert volume to vector for sampling
        Vector<double> logits(logits_volume.GetSize());
        for (int j = 0; j < logits.GetCount(); j++) {
            logits[j] = logits_volume.Get(j);
        }
        
        // Sample next token
        int next_token = model->SampleNextToken(logits, temperature, top_k, k, nucleus, p);
        
        // Add to context
        current_context.Add(next_token);
        
        // Check for end-of-sequence token if applicable
        // (in a real model, this would be a specific token ID)
    }
    
    return current_context;
}

double GPTSession::GetPerplexity(const Vector<int>& test_tokens) {
    // Compute perplexity on test tokens
    // Perplexity = exp(average cross-entropy loss)
    
    if (test_tokens.GetCount() == 0) return INFINITY;
    
    double total_loss = 0.0;
    int seq_len = test_tokens.GetCount();
    
    // Sliding window approach to compute loss on all subsequences
    for (int i = 1; i < seq_len; i++) {
        Vector<int> context;
        for (int j = 0; j < i && j < sequence_length; j++) {
            context.Add(test_tokens[i - j - 1]);  // Add in reverse order
        }
        
        // Invert the context to get the correct order
        Vector<int> actual_context;
        for (int j = context.GetCount() - 1; j >= 0; j--) {
            actual_context.Add(context[j]);
        }
        
        // Get next token logits
        Volume& logits_volume = model->GetNextTokenLogits(actual_context);
        
        // Get target token
        int target_token = test_tokens[i];
        
        // Compute loss for this position
        double target_logit = logits_volume.Get(target_token);
        
        // Simplified loss calculation - a full implementation would use proper cross-entropy
        total_loss += -target_logit; // This is a placeholder
    }
    
    double avg_loss = total_loss / (seq_len - 1);
    return exp(avg_loss);
}

Vector<double> GPTSession::GetEmbeddings(const Vector<int>& tokens) {
    // Get embeddings for the given tokens
    // This would involve processing through the embedding layer of the model
    
    Volume input_volume = model->PrepareInputs(tokens);
    
    // Process through the embedding part of transformer (without full forward pass)
    // This is a simplified approach - real implementation would separate embedding computation
    Vector<double> embeddings;
    for (int i = 0; i < tokens.GetCount(); i++) {
        // For demonstration, just return the input values as "embeddings"
        embeddings.Add((double)tokens[i] / model->GetVocabSize());
    }
    
    return embeddings;
}

std::unique_ptr<GPTModel> CreateGPT(int vocab_size, int embed_dim, int num_heads, 
                                   int num_layers, int ff_dim, int max_seq_len, 
                                   double dropout_rate) {
    return std::make_unique<GPTModel>(vocab_size, embed_dim, num_heads, 
                                     num_layers, ff_dim, max_seq_len, dropout_rate);
}

std::unique_ptr<GPTSession> CreateGPTSession(int vocab_size, int embed_dim, 
                                            int num_heads, int num_layers, 
                                            int ff_dim, int max_seq_len, 
                                            double dropout_rate,
                                            double learning_rate) {
    auto model = CreateGPT(vocab_size, embed_dim, num_heads, num_layers, 
                          ff_dim, max_seq_len, dropout_rate);
    auto session = std::make_unique<GPTSession>(std::move(model));
    session->learning_rate = learning_rate;
    
    return session;
}

} // namespace ConvNet