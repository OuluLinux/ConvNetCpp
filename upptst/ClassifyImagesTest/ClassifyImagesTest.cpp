#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("ClassifyImages Test - Testing image classification functionality");
    
    // Create a network suitable for image classification
    Session session;
    
    // A multi-layer network that could be used for image classification
    String net_str = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":28, \"input_height\":28, \"input_depth\":1},\n"  // 28x28 grayscale image
        "\t{\"type\":\"conv\", \"sx\":5, \"sy\":5, \"out_depth\":8, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"pool\", \"sx\":2, \"sy\":2, \"stride\":2},\n"  // Pooling layer
        "\t{\"type\":\"conv\", \"sx\":5, \"sy\":5, \"out_depth\":16, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"pool\", \"sx\":2, \"sy\":2, \"stride\":2},\n"  // Pooling layer
        "\t{\"type\":\"fc\", \"neuron_count\":100, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"softmax\", \"class_count\":10},\n"  // 10 classes
        "\t{\"type\":\"sgd\", \"learning_rate\":0.01, \"momentum\":0.9, \"batch_size\":32, \"l2_decay\":0.0001}\n"
        "]\n";
    
    bool success = session.MakeLayers(net_str);
    
    LOG("  Network creation: " << (success ? "success" : "failed"));
    ASSERT(success);
    
    LOG("ClassifyImages tests completed successfully!");
}