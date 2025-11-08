#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("ReinforcedLearning Test - Testing reinforcement learning concepts");
    
    // Create a network suitable for reinforcement learning
    Session q_network, target_network;
    
    String dqn_net = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":4},\n"  // State: e.g., position, velocity
        "\t{\"type\":\"fc\", \"neuron_count\":24, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":24, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":3},\n"  // 3 possible actions
        "\t{\"type\":\"adam\", \"learning_rate\":0.001, \"beta1\":0.9, \"beta2\":0.999, \"eps\":1e-8}\n"
        "]\n";
    
    bool q_success = q_network.MakeLayers(dqn_net);
    bool target_success = target_network.MakeLayers(dqn_net);
    
    LOG("  Q-network creation: " << (q_success ? "success" : "failed"));
    LOG("  Target network creation: " << (target_success ? "success" : "failed"));
    ASSERT(q_success && target_success);
    
    // Create experience replay data (state, action, reward, next_state)
    SessionData& d = q_network.Data();
    d.BeginDataResult(3, 2, 4, 0);  // 3 action outputs, 2 samples, 4 state inputs
    
    // Experience 1: state -> Q-values
    d.SetData(0, 0, 0.1).SetData(0, 1, 0.2).SetData(0, 2, 0.3).SetData(0, 3, 0.4);
    d.SetResult(0, 0, 0.5).SetResult(0, 1, 0.2).SetResult(0, 2, 0.8);  // Q-values for 3 actions
    
    // Experience 2: state -> Q-values  
    d.SetData(1, 0, 0.8).SetData(1, 1, 0.7).SetData(1, 2, 0.2).SetData(1, 3, 0.1);
    d.SetResult(1, 0, 0.1).SetResult(1, 1, 0.9).SetResult(1, 2, 0.3);  // Q-values for 3 actions
    
    d.EndData();
    
    q_network.StartTraining();
    
    // Train for a few iterations
    for (int i = 0; i < 10; i++) {
        q_network.Tick();
    }
    
    q_network.StopTraining();
    
    // Test prediction
    Vector<double> state = {0.4, 0.5, 0.1, 0.2};
    Vector<double> q_values = q_network.Predict(state);
    
    LOG("  State: [" << state[0] << ", " << state[1] << ", " << state[2] << ", " << state[3] << "]");
    LOG("  Q-values: [" << q_values[0] << ", " << q_values[1] << ", " << q_values[2] << "]");
    
    // Find best action (highest Q-value)
    int best_action = 0;
    double best_q = q_values[0];
    for (int i = 1; i < q_values.GetCount(); i++) {
        if (q_values[i] > best_q) {
            best_q = q_values[i];
            best_action = i;
        }
    }
    
    LOG("  Best action: " << best_action << " (Q-value: " << best_q << ")");
    
    ASSERT(q_values.GetCount() == 3);
    for (int i = 0; i < q_values.GetCount(); i++) {
        ASSERT(std::isfinite(q_values[i]));
    }
    
    LOG("ReinforcedLearning tests completed successfully!");
}