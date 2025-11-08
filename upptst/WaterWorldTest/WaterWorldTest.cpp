#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("WaterWorld Test - Testing multi-agent reinforcement learning");
    
    // Create a network for multi-agent scenario
    Session agent_network;
    
    String multi_agent_net = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":8},\n"  // State: agent pos (2) + target pos (2) + other agent pos (2) + features (2)
        "\t{\"type\":\"fc\", \"neuron_count\":32, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":32, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":2},\n"  // 2D action (dx, dy)
        "\t{\"type\":\"adam\", \"learning_rate\":0.001, \"beta1\":0.9, \"beta2\":0.999, \"eps\":1e-8}\n"
        "]\n";
    
    bool success = agent_network.MakeLayers(multi_agent_net);
    
    LOG("  Multi-agent network creation: " << (success ? "success" : "failed"));
    ASSERT(success);
    
    // Create training data simulating water world environment
    SessionData& d = agent_network.Data();
    d.BeginDataResult(2, 2, 8, 0);  // 2 action outputs, 2 samples, 8 state inputs
    
    // Sample 1: [agent_x, agent_y, target_x, target_y, other_x, other_y, feature1, feature2]
    d.SetData(0, 0, 0.2).SetData(0, 1, 0.3).SetData(0, 2, 0.8).SetData(0, 3, 0.7);  // agent, target
    d.SetData(0, 4, 0.5).SetData(0, 5, 0.5).SetData(0, 6, 1.0).SetData(0, 7, 0.0);  // other agent, features
    d.SetResult(0, 0, 0.1).SetResult(0, 1, 0.2);  // action toward target
    
    // Sample 2: different configuration
    d.SetData(1, 0, 0.7).SetData(1, 1, 0.2).SetData(1, 2, 0.1).SetData(1, 3, 0.8);
    d.SetData(1, 4, 0.4).SetData(1, 5, 0.6).SetData(1, 6, 0.0).SetData(1, 7, 1.0);
    d.SetResult(1, 0, -0.2).SetResult(1, 1, 0.3);  // action toward target
    
    d.EndData();
    
    agent_network.StartTraining();
    
    // Train for a few iterations
    for (int i = 0; i < 10; i++) {
        agent_network.Tick();
    }
    
    agent_network.StopTraining();
    
    // Test prediction with a water world state
    Vector<double> state = {0.3, 0.4, 0.7, 0.8, 0.5, 0.2, 1.0, 0.0};  // agent at (0.3,0.4), target at (0.7,0.8), etc.
    Vector<double> action = agent_network.Predict(state);
    
    LOG("  State: [agent(0.3,0.4), target(0.7,0.8), other(0.5,0.2), features(1.0,0.0)]");
    LOG("  Action: [" << action[0] << ", " << action[1] << "]");
    
    ASSERT(action.GetCount() == 2);
    ASSERT(std::isfinite(action[0]));  // dx should be finite
    ASSERT(std::isfinite(action[1]));  // dy should be finite
    
    LOG("WaterWorld tests completed successfully!");
}