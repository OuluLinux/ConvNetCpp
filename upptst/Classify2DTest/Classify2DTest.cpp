#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    // Test the network structure from Classify2D example
    Session session;
    
    String t = "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":2},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":6, \"activation\": \"tanh\"},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":2, \"activation\": \"tanh\"},\n"
        "\t{\"type\":\"softmax\", \"class_count\":2},\n"
        "\t{\"type\":\"sgd\", \"learning_rate\":0.01, \"momentum\":0.1, \"batch_size\":10, \"l2_decay\":0.001}\n"
        "]\n";

    bool success = session.MakeLayers(t);
    
    LOG("Classify2D network setup test:");
    LOG("  Network creation success: " << (success ? "true" : "false"));
    ASSERT(success);
    ASSERT(session.GetLayerCount() == 7);  // Should have 7 layers (input + fc + tanh + fc + tanh + fc_for_softmax + softmax)
    
    // Test with simple binary classification data
    SessionData& d = session.Data();
    d.BeginData(2, 6, 2);  // 2 input features, 6 samples, 2 possible labels
    
    // Class 0: points with x1 < 0
    d.SetData(0, 0, -1.0).SetData(0, 1, 0.5).SetLabel(0, 0);
    d.SetData(1, 0, -0.5).SetData(1, 1, -1.0).SetLabel(1, 0);
    d.SetData(2, 0, -2.0).SetData(2, 1, 1.5).SetLabel(2, 0);
    
    // Class 1: points with x1 > 0
    d.SetData(3, 0, 1.0).SetData(3, 1, 0.5).SetLabel(3, 1);
    d.SetData(4, 0, 0.5).SetData(4, 1, -1.0).SetLabel(4, 1);
    d.SetData(5, 0, 2.0).SetData(5, 1, 1.5).SetLabel(5, 1);
    
    d.EndData();
    
    session.StartTraining();
    
    // Train for a few iterations
    for (int i = 0; i < 50; i++) {
        session.Tick();
    }
    
    session.StopTraining();
    
    // Test that the network can process inputs and produce outputs
    Vector<double> input1 = {-1.0, 0.0};  // Should be class 0
    Vector<double> output1 = session.Predict(input1);
    
    Vector<double> input2 = {1.0, 0.0};   // Should be class 1
    Vector<double> output2 = session.Predict(input2);
    
    LOG("Classify2D prediction test:");
    LOG("  Input 1 [" << input1[0] << ", " << input1[1] << "] -> Output ["
         << output1[0] << ", " << output1[1] << "]");
    LOG("  Input 2 [" << input2[0] << ", " << input2[1] << "] -> Output ["
         << output2[0] << ", " << output2[1] << "]");
    
    // Verify that the network produces valid probability outputs (sums to ~1)
    double sum1 = output1[0] + output1[1];
    double sum2 = output2[0] + output2[1];
    
    ASSERT(abs(sum1 - 1.0) < 0.01);  // Softmax outputs should sum to ~1
    ASSERT(abs(sum2 - 1.0) < 0.01);  // Softmax outputs should sum to ~1
    
    LOG("Classify2D tests completed successfully!");
}