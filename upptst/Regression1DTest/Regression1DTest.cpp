#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    // Test the network structure from Regression1D example
    Session ses;
    
    String t = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":1},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"relu\"},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"sigmoid\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":1},\n"
        "\t{\"type\":\"adadelta\", \"learning_rate\":0.01, \"momentum\":0, \"batch_size\":1, \"l2_decay\":0.001}\n"
        "]\n";
    
    bool success = ses.MakeLayers(t);
    
    LOG("Regression1D network setup test:");
    LOG("  Network creation success: " << (success ? "true" : "false"));
    ASSERT(success);
    ASSERT(ses.GetLayerCount() == 5);  // Should have 5 layers
    
    // Test training with simple y = 2*x function
    SessionData& d = ses.Data();
    d.BeginDataResult(1, 4, 1, 0);  // 1 input, 4 samples, 1 output
    
    for (int i = 0; i < 4; i++) {
        double x = i + 1;
        double y = 2 * x;
        d.SetData(i, 0, x);
        d.SetResult(i, 0, y);
    }
    
    d.EndData();
    
    ses.StartTraining();
    
    // Train for a few iterations
    for (int iter = 0; iter < 10; iter++) {
        ses.Tick();
    }
    
    ses.StopTraining();
    
    // Test prediction
    Vector<double> input = {2.0};
    Vector<double> output = ses.Predict(input);
    
    LOG("Regression1D prediction test:");
    LOG("  Input: " << input[0]);
    LOG("  Expected output (around 4.0): " << output[0]);
    
    // Should be close to 4.0 (since y = 2*x)
    // Allow some tolerance since training might not be perfect in just 10 iterations
    ASSERT(abs(output[0] - 4.0) < 2.0);  // Allow up to 2.0 error for basic functionality test
    
    LOG("Regression1D tests completed successfully!");
}