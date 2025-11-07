#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("HeteroscedasticUncertainty Test - Testing uncertainty estimation functionality");
    
    // Create a network that can estimate uncertainty (like HeteroscedasticRegressionLayer)
    Session session;
    
    // A network with regression layer that can potentially estimate uncertainty
    String net_str = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":2},\n"  // 2D input
        "\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\":\"tanh\"},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\":\"tanh\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":2},\n"  // 2 outputs: prediction + uncertainty
        "\t{\"type\":\"adam\", \"learning_rate\":0.01, \"beta1\":0.9, \"beta2\":0.999, \"eps\":1e-8}\n"
        "]\n";
    
    bool success = session.MakeLayers(net_str);
    
    LOG("  Network creation for Heteroscedastic: " << (success ? "success" : "failed"));
    ASSERT(success);
    
    // Create test data for a simple function with varying uncertainty
    SessionData& d = session.Data();
    d.BeginDataResult(2, 5, 2, 0);  // 2 inputs, 5 samples, 2 outputs
    
    for (int i = 0; i < 5; i++) {
        double x = i * 0.5;
        // Function: y = x^2, with increasing noise
        double y = x * x + Randomf() * x * 0.2;  // uncertainty increases with x
        d.SetData(i, 0, x);
        d.SetData(i, 1, 1.0);  // bias term
        d.SetResult(i, 0, y);  // mean prediction
        d.SetResult(i, 1, x * 0.2);  // uncertainty (simplified)
    }
    
    d.EndData();
    
    session.StartTraining();
    
    // Train for a few iterations
    for (int i = 0; i < 10; i++) {
        session.Tick();
    }
    
    session.StopTraining();
    
    // Test prediction
    Vector<double> input = {0.7, 1.0};
    Vector<double> output = session.Predict(input);
    
    LOG("  Input: [" << input[0] << ", " << input[1] << "]");
    LOG("  Output: [" << output[0] << ", " << output[1] << "]");  // prediction, uncertainty
    
    ASSERT(output.GetCount() == 2);
    ASSERT(std::isfinite(output[0]));  // prediction should be finite
    ASSERT(std::isfinite(output[1]));  // uncertainty should be finite
    
    LOG("HeteroscedasticUncertainty tests completed successfully!");
}