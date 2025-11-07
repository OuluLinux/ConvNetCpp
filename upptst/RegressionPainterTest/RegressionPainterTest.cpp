#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("RegressionPainter Test - Testing image regression functionality");
    
    // Test a network structure similar to what might be used for image regression
    Session session;
    
    // A more complex network that could be used for image tasks
    String net_str = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":8, \"input_height\":8, \"input_depth\":1},\n"  // 8x8 input
        "\t{\"type\":\"fc\", \"neuron_count\":32, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":16, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":64},\n"  // 64 outputs for 8x8 image
        "\t{\"type\":\"adadelta\", \"learning_rate\":0.001, \"momentum\":0, \"batch_size\":1, \"l2_decay\":0.001}\n"
        "]\n";
    
    bool success = session.MakeLayers(net_str);
    
    LOG("  Network creation: " << (success ? "success" : "failed"));
    ASSERT(success);
    
    // Create test data - a simple pattern that the network could learn
    SessionData& d = session.Data();
    d.BeginDataResult(64, 1, 64, 0);  // 64 inputs (8x8), 1 sample, 64 outputs (8x8)
    
    // Create a simple 8x8 input image (flattened)
    for (int i = 0; i < 64; i++) {
        double input_val = sin(i * 0.3);  // Simple pattern
        d.SetData(0, i, input_val);
        
        // Create a target output that's related to the input
        double output_val = input_val * 0.5 + 0.2;  // Simple transformation
        d.SetResult(0, i, output_val);
    }
    
    d.EndData();
    
    session.StartTraining();
    
    // Train for a few iterations
    for (int i = 0; i < 20; i++) {
        session.Tick();
    }
    
    session.StopTraining();
    
    // Create a test input based on the same pattern
    Vector<double> test_input;
    for (int i = 0; i < 64; i++) {
        test_input.Add(sin((i + 1) * 0.3));
    }
    
    Vector<double> prediction = session.Predict(test_input);
    
    LOG("  Input sample [0]: " << test_input[0]);
    LOG("  Output sample [0]: " << prediction[0]);
    LOG("  Prediction vector size: " << prediction.GetCount());
    
    ASSERT(prediction.GetCount() == 64);  // Should have 64 outputs
    
    // Verify that all outputs are finite values
    for (int i = 0; i < prediction.GetCount(); i++) {
        ASSERT(std::isfinite(prediction[i]));
    }
    
    LOG("RegressionPainter tests completed successfully!");
}