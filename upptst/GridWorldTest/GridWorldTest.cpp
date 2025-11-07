#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("GridWorld Test - Testing reinforcement learning functionality");
    
    // Create a network suitable for reinforcement learning in grid world
    Session session;
    
    // A network that could be used for Q-learning in a grid environment
    String net_str = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":16},\n"  // One-hot encoded state (4x4 grid)
        "\t{\"type\":\"fc\", \"neuron_count\":32, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":64, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":4},\n"  // 4 possible actions (up, down, left, right)
        "\t{\"type\":\"adam\", \"learning_rate\":0.001, \"beta1\":0.9, \"beta2\":0.999, \"eps\":1e-8}\n"
        "]\n";
    
    bool success = session.MakeLayers(net_str);
    
    LOG("  Network creation for GridWorld: " << (success ? "success" : "failed"));
    ASSERT(success);
    
    // Create simple test data (state -> action value prediction)
    SessionData& d = session.Data();
    d.BeginDataResult(16, 1, 4, 0);  // 16 state inputs, 1 sample, 4 action outputs
    
    // Create a simple state representation (one-hot for position in 4x4 grid)
    for (int i = 0; i < 16; i++) {
        d.SetData(0, i, (i == 5) ? 1.0 : 0.0);  // Position at index 5
    }
    // Target action values (arbitrary for testing)
    for (int i = 0; i < 4; i++) {
        d.SetResult(0, i, 0.25 * i);  // Different values for each action
    }
    
    d.EndData();
    
    session.StartTraining();
    
    // Train for a few iterations
    for (int i = 0; i < 5; i++) {
        session.Tick();
    }
    
    session.StopTraining();
    
    // Test prediction with the same state
    Vector<double> state_input(16, 0.0);
    state_input[5] = 1.0;  // Same state as training
    
    Vector<double> q_values = session.Predict(state_input);
    
    LOG("  State input (position 5): [" << state_input[5] << "]");
    LOG("  Q-values for actions: [";
    for (int i = 0; i < q_values.GetCount(); i++) {
        LOG("  " << q_values[i] << (i < q_values.GetCount()-1 ? "," : ""));
    }
    LOG("  ]");
    
    ASSERT(q_values.GetCount() == 4);  // Should have Q-value for each action
    
    LOG("GridWorld tests completed successfully!");
}