#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();

    // Test CharGen vocabulary setup functionality
    Vector<WString> sents;
    sents.Add(WString("hello"));
    sents.Add(WString("world"));

    VectorMap<int, int> letterToIndex;
    VectorMap<int, int> indexToLetter;
    Vector<WString> vocab;
    int input_size, output_size, epoch_size;

    // Simulate the InitVocab logic from CharGen
    WString txt = Join(sents, ""); // concat all
    VectorMap<int, int> d;
    for (int i = 0; i < txt.GetCount(); i++) {
        int txti = txt[i];
        d.GetAdd(txti, 0)++;
    }

    // filter by count threshold and create pointers
    letterToIndex.Clear();
    indexToLetter.Clear();
    vocab.Clear();

    int q = 1; // Start at 1 to reserve 0 for special tokens
    for (int i = 0; i < d.GetCount(); i++) {
        int ch = d.GetKey(i);
        if (d[i] >= 1) { // count threshold = 1
            letterToIndex.Add(ch, q);
            indexToLetter.Add(q, ch);
            vocab.Add().Cat(ch);
            q++;
        }
    }

    input_size = vocab.GetCount() + 1;
    output_size = vocab.GetCount() + 1;
    epoch_size = sents.GetCount();

    // Verify results
    LOG("CharGen vocabulary test:");
    LOG("  Vocabulary size: " << vocab.GetCount());
    LOG("  Input size: " << input_size);
    LOG("  Output size: " << output_size);
    LOG("  Epoch size: " << epoch_size);

    ASSERT(vocab.GetCount() == 7);  // h,e,l,o,w,r,d (duplicates removed)
    ASSERT(input_size == 8);        // vocab size + 1 (for START token)
    ASSERT(output_size == 8);       // vocab size + 1 (for END token)
    ASSERT(epoch_size == 2);        // 2 sentences

    // Test RecurrentSession with LSTM configuration including tokenization
    RecurrentSession ses;

    String model_str = "{\n"
        "\t\"generator\":\"lstm\",\n"
        "\t\"hidden_sizes\":[10,10],\n"
        "\t\"letter_size\":5,\n"
        "\t\"use_tokenization\":true,\n"  // Test with tokenization enabled
        "\t\"regc\":0.000001,\n"
        "\t\"learning_rate\":0.01,\n"
        "\t\"clipval\":5.0\n"
        "}";

    ValueMap js = ParseJSON(model_str);
    ses.Load(js);
    
    // Verify that tokenization was properly loaded
    LOG("Tokenization enabled in RecurrentSession: " << (ses.use_tokenization ? "true" : "false"));
    ASSERT(ses.use_tokenization == true);  // Should be true based on JSON config

    ses.SetInputSize(20);  // arbitrary size
    ses.SetOutputSize(20); // arbitrary size
    ses.Init();

    LOG("CharGen RecurrentSession test:");
    LOG("  Learning rate: " << ses.GetLearningRate());
    ASSERT(ses.GetLearningRate() == 0.01);

    // Test tokenizer functionality
    LOG("Testing Tokenizer functionality:");
    SubwordTokenizer tokenizer;
    tokenizer.BuildVocabulary(sents, 1);
    
    LOG("  Tokenizer vocab size: " << tokenizer.GetVocabSize());
    LOG("  Tokenizing 'hello':");
    
    Vector<int> token_ids = tokenizer.Tokenize(WString("hello"));
    for (int i = 0; i < token_ids.GetCount(); i++) {
        Upp::String token = tokenizer.GetToken(token_ids[i]);
        LOG("    Character '" << token << "' -> ID " << token_ids[i]);
    }
    
    WString detokenized = tokenizer.Detokenize(token_ids);
    Cout() << "  Detokenized result: '" << detokenized << "'" << '\n';
    Cout() << "  Expected: 'hello'" << '\n';
    Cout() << "  Are they equal? " << (detokenized == WString("hello") ? "YES" : "NO") << '\n';
    Cout() << "  Length of detokenized: " << detokenized.GetCount() << '\n';
    Cout() << "  Length of expected: " << WString("hello").GetCount() << '\n';
    
    if (detokenized != WString("hello")) {
        Cout() << "  Token IDs were: " << '\n';
        for (int i = 0; i < token_ids.GetCount(); i++) {
            Cout() << "    " << token_ids[i] << " -> token string '" << tokenizer.GetToken(token_ids[i]) << "'" 
                   << " -> as WString '" << tokenizer.GetToken(token_ids[i]).ToWString() << "'" << '\n';
        }
        
        // Let's also look at the full vocabulary
        Cout() << "Full vocabulary: " << '\n';
        for(int i = 0; i < tokenizer.GetVocabSize(); i++) {
            Upp::String token_str = tokenizer.GetToken(i);
            Cout() << "  ID " << i << " -> string '" << token_str << "'" 
                   << " -> WString '" << token_str.ToWString() << "'" << '\n';
        }
        
        // Print character by character comparison
        Cout() << "Character by character comparison:" << '\n';
        for(int i = 0; i < min(detokenized.GetCount(), (int)WString("hello").GetCount()); i++) {
            Cout() << "  Position " << i << ": got '" << detokenized[i] << "' (code " << (int)detokenized[i] 
                   << "), expected '" << WString("hello")[i] << "' (code " << (int)WString("hello")[i] << ")" << '\n';
        }
        
        Cout() << "TEST FAILED: detokenization did not work properly" << '\n';
    } else {
        Cout() << "TEST PASSED: detokenization works correctly" << '\n';
    }
    
    // The assertion that was causing the test to fail
    if (detokenized != WString("hello")) {
        LOG("TEST FAILED: detokenization did not work properly");
        // Exit with failure if needed
    } else {
        LOG("TEST PASSED: detokenization works correctly");
    }
    ASSERT(detokenized == WString("hello"));

    LOG("CharGen tests completed successfully!");
}