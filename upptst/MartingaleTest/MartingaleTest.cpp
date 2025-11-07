#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("Martingale Test - Testing financial prediction functionality");
    
    // Create a network that could be used for financial time series prediction
    Session session;
    
    // A network suitable for time series prediction
    String net_str = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":10},\n"  // 10 time steps
        "\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":10, \"activation\":\"relu\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":1},\n"  // Predict next value
        "\t{\"type\":\"adam\", \"learning_rate\":0.001, \"beta1\":0.9, \"beta2\":0.999, \"eps\":1e-8}\n"
        "]\n";
    
    bool success = session.MakeLayers(net_str);
    
    LOG("  Network creation for Martingale: " << (success ? "success" : "failed"));
    ASSERT(success);
    
    // Create simple time series data (random walk)
    SessionData& d = session.Data();
    d.BeginDataResult(10, 3, 1, 0);  // 10 inputs, 3 samples, 1 output
    
    for (int s = 0; s < 3; s++) {  // 3 different sequences
        double value = 100.0;  // Starting value
        for (int i = 0; i < 10; i++) {
            d.SetData(s, i, value);
            value += Randomf() * 2.0 - 1.0;  // Small random change
        }
        d.SetResult(s, 0, value);  // Next value in sequence
    }
    
    d.EndData();
    
    session.StartTraining();
    
    // Train for a few iterations
    for (int i = 0; i < 10; i++) {
        session.Tick();
    }
    
    session.StopTraining();
    
    // Test prediction with a new sequence
    Vector<double> series_input(10);
    double val = 105.0;
    for (int i = 0; i < 10; i++) {
        series_input[i] = val;
        val += Randomf() * 1.0 - 0.5;
    }
    
    Vector<double> prediction = session.Predict(series_input);
    
    LOG("  Time series input [0]: " << series_input[0]);
    LOG("  Predicted next value: " << prediction[0]);
    
    ASSERT(prediction.GetCount() == 1);
    ASSERT(std::isfinite(prediction[0]));
    
    LOG("Martingale tests completed successfully!");
}