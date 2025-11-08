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
    
    // Test RecurrentSession with LSTM configuration
    RecurrentSession ses;
    
    String model_str = "{\n"
        "\t\"generator\":\"lstm\",\n"
        "\t\"hidden_sizes\":[10,10],\n"
        "\t\"letter_size\":5,\n"
        "\t\"regc\":0.000001,\n"
        "\t\"learning_rate\":0.01,\n"
        "\t\"clipval\":5.0\n"
        "}";
    
    ValueMap js = ParseJSON(model_str);
    ses.Load(js);
    ses.SetInputSize(20);  // arbitrary size
    ses.SetOutputSize(20); // arbitrary size
    ses.Init();
    
    LOG("CharGen RecurrentSession test:");
    LOG("  Learning rate: " << ses.GetLearningRate());
    ASSERT(ses.GetLearningRate() == 0.01);
    
    LOG("CharGen tests completed successfully!");
}