#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();

    LOG("GridWorld Test - Testing network initialization with working data pattern");

    // Create a network suitable for reinforcement learning in grid world
    Session session;

    // A network that could be used for Q-learning in a grid environment
    String net_str =
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":16},\n"  // One-hot encoded state (4x4 grid)
        "\t{\"type\":\"fc\", \"neuron_count\":32, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":4},\n"  // 4 possible actions (up, down, left, right)
        "\t{\"type\":\"sgd\", \"learning_rate\":0.01, \"momentum\":0.1, \"batch_size\":1, \"l2_decay\":0.001}\n"
        "]\n";

    bool success = session.MakeLayers(net_str);

    LOG("  Network creation for GridWorld: " << (success ? "success" : "failed"));
    ASSERT(success);

    // Create simple test data (state -> action value prediction) following Regression1DTest pattern
    SessionData& d = session.Data();
    LOG("  Setting up training data...");
    d.BeginDataResult(16, 4, 4, 0);  // 16 state inputs, 4 samples, 4 action outputs

    // Create multiple samples of simple state representation (one-hot for position in 4x4 grid)
    for (int sample = 0; sample < 4; sample++) {
        for (int i = 0; i < 16; i++) {
            d.SetData(sample, i, (i == sample) ? 1.0 : 0.0);  // Different position for each sample
        }
        // Target action values (arbitrary for testing)
        for (int i = 0; i < 4; i++) {
            d.SetResult(sample, i, 0.25 * i + sample * 0.1);  // Different values for each action and sample
        }
    }

    d.EndData();
    LOG("GridWorld initialization with working data pattern completed successfully!");
}