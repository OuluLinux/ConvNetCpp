#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("TemporalDifference Test - Testing temporal difference learning");
    
    // Create a network for value function approximation in TD learning
    Session value_network;
    
    String td_net = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":5},\n"  // State features
        "\t{\"type\":\"fc\", \"neuron_count\":16, \"activation\":\"tanh\"},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":16, \"activation\":\"tanh\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":1},\n"  // Value estimate
        "\t{\"type\":\"adam\", \"learning_rate\":0.01, \"beta1\":0.9, \"beta2\":0.999, \"eps\":1e-8}\n"
        "]\n";
    
    bool success = value_network.MakeLayers(td_net);
    
    LOG("  TD Network creation: " << (success ? "success" : "failed"));
    ASSERT(success);
    
    // Create temporal difference learning data (state -> value)
    SessionData& d = value_network.Data();
    d.BeginDataResult(1, 3, 5, 0);  // 1 value output, 3 samples, 5 state inputs
    
    // Sample states with their true values (for TD training)
    // In TD learning, we train on (V(s_t) ≈ r_t + γ*V(s_t+1))
    d.SetData(0, 0, 0.1).SetData(0, 1, 0.2).SetData(0, 2, 0.3).SetData(0, 3, 0.4).SetData(0, 4, 0.5);
    d.SetResult(0, 0, 0.7);  // Value of state 1
    
    d.SetData(1, 0, 0.2).SetData(1, 1, 0.3).SetData(1, 2, 0.4).SetData(1, 3, 0.5).SetData(1, 4, 0.6);
    d.SetResult(1, 0, 0.5);  // Value of state 2
    
    d.SetData(2, 0, 0.3).SetData(2, 1, 0.4).SetData(2, 2, 0.5).SetData(2, 3, 0.6).SetData(2, 4, 0.7);
    d.SetResult(2, 0, 0.3);  // Value of state 3
    
    d.EndData();
    
    value_network.StartTraining();
    
    // Train for a few iterations
    for (int i = 0; i < 15; i++) {
        value_network.Tick();
    }
    
    value_network.StopTraining();
    
    // Test value prediction for a state
    Vector<double> state = {0.15, 0.25, 0.35, 0.45, 0.55};
    Vector<double> value = value_network.Predict(state);
    
    LOG("  State: [" << state[0] << ", ..., " << state[4] << "]");
    LOG("  Estimated value: " << value[0]);
    
    ASSERT(value.GetCount() == 1);
    ASSERT(std::isfinite(value[0]));
    
    LOG("TemporalDifference tests completed successfully!");
}