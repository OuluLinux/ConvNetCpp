#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();

    // Test tokenizer functionality in isolation
    Vector<WString> sents;
    sents.Add(WString("hello"));
    sents.Add(WString("world"));
    
    LOG("Testing Tokenizer functionality:");
    SubwordTokenizer tokenizer;
    tokenizer.BuildVocabulary(sents, 1);
    
    LOG("Tokenizer vocab size: " << tokenizer.GetVocabSize());
    LOG("Full vocabulary: ");
    for(int i = 0; i < tokenizer.GetVocabSize(); i++) {
        LOG("  ID " << i << " -> '" << tokenizer.GetToken(i) << "'");
    }
    
    LOG("Tokenizing 'hello':");
    Vector<int> token_ids = tokenizer.Tokenize(WString("hello"));
    for (int i = 0; i < token_ids.GetCount(); i++) {
        LOG("    Char '" << WString(1, (wchar_t)sents[0][i]) << "' -> ID " << token_ids[i] 
           << " -> Token '" << tokenizer.GetToken(token_ids[i]) << "'");
    }
    
    WString detokenized = tokenizer.Detokenize(token_ids);
    LOG("Detokenized result: '" << detokenized << "'");
    LOG("Expected: 'hello'");
    LOG("Are they equal? " << (detokenized == WString("hello") ? "YES" : "NO"));
    
    if (detokenized != WString("hello")) {
        LOG("TEST FAILED: detokenization did not work properly");
    } else {
        LOG("TEST PASSED: detokenization works correctly");
    }
}