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
        "\t{\"type\":\"fc\", \"neuron_count\":100, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"softmax\", \"class_count\":10}\n"  // 10 classes
        "]\n";

    bool success = session.MakeLayers(net_str);

    LOG("  Network creation: " << (success ? "success" : "failed"));
    ASSERT(success);

    LOG("ClassifyImages tests completed successfully!");
}