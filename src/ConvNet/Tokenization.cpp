#include "Tokenization.h"

namespace ConvNet {

Tokenizer::Tokenizer() {
    // Initialize with special tokens - START_TOKEN_ID is 0 by default
    // Add special tokens - using 0 for START/END as in existing CharGen code
    AddToken("<START>");
    AddToken("<END>");
    AddToken("<UNK>");  // Unknown token
}

Tokenizer::~Tokenizer() {
}

void Tokenizer::BuildVocabulary(const Upp::Vector<Upp::WString>& texts, int min_frequency) {
    // Clear existing vocabulary (keeping special tokens at the beginning)
    Upp::String start_token = GetToken(0);
    Upp::String end_token = GetToken(1);
    Upp::String unk_token = GetToken(2);
    
    token_to_id.Clear();
    id_to_token.Clear();
    
    // Re-add special tokens
    token_to_id.Add(start_token, 0);
    id_to_token.Add(0, start_token);
    token_to_id.Add(end_token, 1);
    id_to_token.Add(1, end_token);
    token_to_id.Add(unk_token, 2);
    id_to_token.Add(2, unk_token);
    
    // Count token frequencies
    Upp::VectorMap<Upp::String, int> token_counts;
    
    for (const auto& text : texts) {
        // For a word-level tokenizer, we split by whitespace and punctuation
        Upp::String str = text.ToString(); // Convert to String
        
        Upp::String current_token;
        for(int i = 0; i < str.GetCount(); i++) {
            char c = str[i];
            // Check if character is a delimiter
            bool is_delimiter = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || 
                               c == '.' || c == ',' || c == '!' || c == '?' || 
                               c == ';' || c == ':' || c == '"' || c == '\'' ||
                               c == '[' || c == ']' || c == '{' || c == '}' ||
                               c == '(' || c == ')' || c == '<' || c == '>' ||
                               c == '/' || c == '\\' || c == '|' || c == '_' ||
                               c == '-' || c == '=' || c == '+' || c == '*' ||
                               c == '&' || c == '^' || c == '%' || c == '$' ||
                               c == '#' || c == '@' || c == '~' || c == '`');
            
            if(is_delimiter) {
                if(!current_token.IsEmpty()) {
                    token_counts.GetAdd(current_token, 0)++;
                    current_token.Clear();
                }
                // Add the delimiter as a separate token to preserve it
                Upp::String delim_token;
                delim_token.Cat(c);
                token_counts.GetAdd(delim_token, 0)++;
            } else {
                current_token.Cat(c);
            }
        }
        // Add the last token if there is one
        if (!current_token.IsEmpty()) {
            token_counts.GetAdd(current_token, 0)++;
        }
    }
    
    // Add tokens that meet the frequency threshold
    int next_id = GetVocabSize(); // Start after special tokens
    for (int i = 0; i < token_counts.GetCount(); i++) {
        if (token_counts[i] >= min_frequency) {
            Upp::String token = token_counts.GetKey(i);
            if (token_to_id.Find(token) == -1) {  // Don't add if already exists as special token
                token_to_id.Add(token, next_id);
                id_to_token.Add(next_id, token);
                next_id++;
            }
        }
    }
}

Upp::Vector<int> Tokenizer::Tokenize(const Upp::WString& text) const {
    Upp::Vector<int> token_ids;
    
    // For word-level tokenization, split by spaces and punctuation
    Upp::String str = text.ToString();
    Upp::String current_token;
    for(int i = 0; i < str.GetCount(); i++) {
        char c = str[i];
        // Check if character is a delimiter
        bool is_delimiter = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || 
                           c == '.' || c == ',' || c == '!' || c == '?' || 
                           c == ';' || c == ':' || c == '"' || c == '\'' ||
                           c == '[' || c == ']' || c == '{' || c == '}' ||
                           c == '(' || c == ')' || c == '<' || c == '>' ||
                           c == '/' || c == '\\' || c == '|' || c == '_' ||
                           c == '-' || c == '=' || c == '+' || c == '*' ||
                           c == '&' || c == '^' || c == '%' || c == '$' ||
                           c == '#' || c == '@' || c == '~' || c == '`');
        
        if(is_delimiter) {
            // Add the current token if there is one
            if(!current_token.IsEmpty()) {
                int token_id = GetTokenId(current_token);
                if (token_id != UNKNOWN_TOKEN_ID) {
                    token_ids.Add(token_id);
                } else {
                    token_ids.Add(GetTokenId("<UNK>"));  // Use unknown token
                }
                current_token.Clear();
            }
            // Add the delimiter as a separate token
            Upp::String delim_token;
            delim_token.Cat(c);
            int token_id = GetTokenId(delim_token);
            if (token_id != UNKNOWN_TOKEN_ID) {
                token_ids.Add(token_id);
            } else {
                token_ids.Add(GetTokenId("<UNK>"));  // Use unknown token
            }
        } else {
            current_token.Cat(c);
        }
    }
    // Add the last token if there is one
    if (!current_token.IsEmpty()) {
        int token_id = GetTokenId(current_token);
        if (token_id != UNKNOWN_TOKEN_ID) {
            token_ids.Add(token_id);
        } else {
            token_ids.Add(GetTokenId("<UNK>"));  // Use unknown token
        }
    }
    
    return token_ids;
}

Upp::WString Tokenizer::Detokenize(const Upp::Vector<int>& token_ids) const {
    Upp::WString result;
    
    for (int token_id : token_ids) {
        Upp::String token = GetToken(token_id);
        if (!token.IsEmpty()) {
            // For character-level tokenization, the token is typically a single character
            // Convert it to WString and append
            Upp::WString token_wstr = token.ToWString();
            result.Cat(token_wstr);
        }
    }
    
    return result;
}

void Tokenizer::AddToken(const Upp::String& token) {
    int next_id = GetVocabSize();
    token_to_id.Add(token, next_id);
    id_to_token.Add(next_id, token);
}

int Tokenizer::GetTokenId(const Upp::String& token) const {
    int idx = token_to_id.Find(token);
    if (idx != -1) {
        return token_to_id[idx];
    }
    return UNKNOWN_TOKEN_ID;
}

Upp::String Tokenizer::GetToken(int token_id) const {
    int idx = id_to_token.Find(token_id);
    if (idx != -1) {
        return id_to_token[idx];
    }
    return "";
}

int Tokenizer::GetVocabSize() const {
    return token_to_id.GetCount();
}

bool Tokenizer::IsInitialized() const {
    return token_to_id.GetCount() > 0;
}

SubwordTokenizer::SubwordTokenizer() : Tokenizer() {
    // Initialize subword tokenizer specific parameters
}

void SubwordTokenizer::BuildVocabulary(const Upp::Vector<Upp::WString>& texts, int min_frequency, int vocab_size) {
    // This is a simplified implementation
    // A full implementation would use algorithms like BPE or WordPiece
    
    // For now, just call the base implementation
    Tokenizer::BuildVocabulary(texts, min_frequency);
    
    // If vocabulary exceeds vocab_size, we could implement pruning here
    if (GetVocabSize() > vocab_size) {
        // In a real implementation, we'd apply vocabulary reduction techniques
        // For now, just issue a warning if needed
    }
}

CharacterTokenizer::CharacterTokenizer() : Tokenizer() {
    // Character tokenizer is just a specialized tokenizer
}

void CharacterTokenizer::BuildVocabulary(const Upp::Vector<Upp::WString>& texts, int min_frequency) {
    Tokenizer::BuildVocabulary(texts, min_frequency);
}

} // namespace ConvNet