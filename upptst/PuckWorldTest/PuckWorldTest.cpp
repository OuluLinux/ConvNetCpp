#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("PuckWorld Test - Testing reinforcement learning in continuous control");
    
    // Create a network suitable for continuous control (like PuckWorld)
    Session session;
    
    // A network for policy or value function in continuous control
    String net_str = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":6},\n"  // State: puck x,y + vel x,y + agent x,y
        "\t{\"type\":\"fc\", \"neuron_count\":32, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":32, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":2},\n"  // 2D action (dx, dy)
        "\t{\"type\":\"adam\", \"learning_rate\":0.001, \"beta1\":0.9, \"beta2\":0.999, \"eps\":1e-8}\n"
        "]\n";
    
    bool success = session.MakeLayers(net_str);
    
    LOG("  Network creation for PuckWorld: " << (success ? "success" : "failed"));
    ASSERT(success);
    
    // Create simple test data representing puck world state
    SessionData& d = session.Data();
    d.BeginDataResult(2, 2, 6, 0);  // 2 action outputs, 2 samples, 6 state inputs
    
    // Sample 1: puck at (0.2, 0.3) moving to (0.25, 0.35), agent at (0.8, 0.7)
    d.SetData(0, 0, 0.2).SetData(0, 1, 0.3).SetData(0, 2, 0.05).SetData(0, 3, 0.05).SetData(0, 4, 0.8).SetData(0, 5, 0.7);
    d.SetResult(0, 0, -0.1).SetResult(0, 1, -0.2);  // Action to move toward puck
    
    // Sample 2: puck at (0.7, 0.8) moving to (0.65, 0.75), agent at (0.1, 0.2)
    d.SetData(1, 0, 0.7).SetData(1, 1, 0.8).SetData(1, 2, -0.05).SetData(1, 3, -0.05).SetData(1, 4, 0.1).SetData(1, 5, 0.2);
    d.SetResult(1, 0, 0.3).SetResult(1, 1, 0.4);  // Action to move toward puck
    
    d.EndData();
    
    session.StartTraining();
    
    // Train for a few iterations
    for (int i = 0; i < 10; i++) {
        session.Tick();
    }
    
    session.StopTraining();
    
    // Test prediction with a state
    Vector<double> state = {0.5, 0.5, 0.01, 0.01, 0.3, 0.7};  // puck at (0.5,0.5), agent at (0.3,0.7)
    Vector<double> action = session.Predict(state);
    
    LOG("  State: [puck(0.5,0.5), vel(0.01,0.01), agent(0.3,0.7)]");
    LOG("  Action: [" << action[0] << ", " << action[1] << "]");
    
    ASSERT(action.GetCount() == 2);
    ASSERT(std::isfinite(action[0]));  // dx should be finite
    ASSERT(std::isfinite(action[1]));  // dy should be finite
    
    LOG("PuckWorld tests completed successfully!");
}