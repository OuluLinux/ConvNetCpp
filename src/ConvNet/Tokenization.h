#ifndef _ConvNet_Tokenization_h_
#define _ConvNet_Tokenization_h_

#include "ConvNet.h"
#include <Core/Core.h>

namespace ConvNet {

class Tokenizer {
private:
    // Maps tokens (strings) to integers
    Upp::VectorMap<Upp::String, int> token_to_id;
    // Maps integers to tokens (strings) 
    Upp::VectorMap<int, Upp::String> id_to_token;
    
    // Special token IDs
    static const int START_TOKEN_ID = 0;
    static const int END_TOKEN_ID = 0;  // Using same as START for now, as in char-level approach
    static const int UNKNOWN_TOKEN_ID = -1;
    
public:
    Tokenizer();
    ~Tokenizer();
    
    // Build vocabulary from text
    void BuildVocabulary(const Upp::Vector<Upp::WString>& texts, int min_frequency = 1);
    
    // Tokenize a string into a sequence of token IDs
    Upp::Vector<int> Tokenize(const Upp::WString& text) const;
    
    // Convert a sequence of token IDs back to text
    Upp::WString Detokenize(const Upp::Vector<int>& token_ids) const;
    
    // Add a specific token to vocabulary
    void AddToken(const Upp::String& token);
    
    // Get token ID for a token
    int GetTokenId(const Upp::String& token) const;
    
    // Get token string for an ID
    Upp::String GetToken(int token_id) const;
    
    // Get vocabulary size
    int GetVocabSize() const;
    
    // Check if tokenizer has been initialized
    bool IsInitialized() const;
};

// SentencePiece-like tokenizer that breaks text into subword units
class SubwordTokenizer : public Tokenizer {
private:
    // BPE (Byte-Pair Encoding) or similar algorithm implementation
    // For now, we'll implement a simple character-level tokenizer
    // as a placeholder, but a real implementation would use BPE, WordPiece, etc.

public:
    SubwordTokenizer();
    
    // Build vocabulary using BPE or similar algorithm
    void BuildVocabulary(const Upp::Vector<Upp::WString>& texts, int min_frequency = 1, int vocab_size = 1000);
};

// Default character-level tokenizer (existing CharGen behavior)
class CharacterTokenizer : public Tokenizer {
public:
    CharacterTokenizer();
    
    // Build vocabulary from characters in the text
    void BuildVocabulary(const Upp::Vector<Upp::WString>& texts, int min_frequency = 1);
};

} // namespace ConvNet

#endif