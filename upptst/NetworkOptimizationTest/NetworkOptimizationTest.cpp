#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("NetworkOptimization Test - Testing different optimization algorithms");
    
    // Test different optimization methods
    Session sgd_session, adam_session, adagrad_session;
    
    String net_template = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":2},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":10, \"activation\":\"tanh\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":1},\n"
        "\t{\"optimizer_placeholder\"}\n"
        "]\n";
    
    // Test SGD optimizer
    String sgd_net = net_template;
    sgd_net.Replace("\"optimizer_placeholder\"", 
        "{\"type\":\"sgd\", \"learning_rate\":0.01, \"momentum\":0.1, \"batch_size\":10, \"l2_decay\":0.001}");
    
    bool sgd_success = sgd_session.MakeLayers(sgd_net);
    LOG("  SGD network creation: " << (sgd_success ? "success" : "failed"));
    ASSERT(sgd_success);
    
    // Test Adam optimizer
    String adam_net = net_template;
    adam_net.Replace("\"optimizer_placeholder\"", 
        "{\"type\":\"adam\", \"learning_rate\":0.01, \"beta1\":0.9, \"beta2\":0.999, \"eps\":1e-8}");
    
    bool adam_success = adam_session.MakeLayers(adam_net);
    LOG("  Adam network creation: " << (adam_success ? "success" : "failed"));
    ASSERT(adam_success);
    
    // Test Adagrad optimizer
    String adagrad_net = net_template;
    adagrad_net.Replace("\"optimizer_placeholder\"", 
        "{\"type\":\"adagrad\", \"learning_rate\":0.01, \"batch_size\":1, \"l2_decay\":0.001}");
    
    bool adagrad_success = adagrad_session.MakeLayers(adagrad_net);
    LOG("  Adagrad network creation: " << (adagrad_success ? "success" : "failed"));
    ASSERT(adagrad_success);
    
    // Create simple test data for all optimizers
    Vector<double> test_inputs[3] = {
        {0.5, 0.3},  // For SGD
        {0.5, 0.3},  // For Adam  
        {0.5, 0.3}   // For Adagrad
    };
    
    Vector<Session*> sessions = {&sgd_session, &adam_session, &adagrad_session};
    
    for (int i = 0; i < 3; i++) {
        SessionData& d = sessions[i]->Data();
        d.BeginDataResult(1, 2, 2, 0);  // 1 output, 2 samples, 2 inputs
        
        d.SetData(0, 0, 0.5).SetData(0, 1, 0.3).SetResult(0, 0, 0.8);
        d.SetData(1, 0, 0.2).SetData(1, 1, 0.7).SetResult(1, 0, 0.9);
        
        d.EndData();
        
        sessions[i]->StartTraining();
        
        // Train each for a few iterations
        for (int j = 0; j < 5; j++) {
            sessions[i]->Tick();
        }
        
        sessions[i]->StopTraining();
        
        // Test prediction
        Vector<double> output = sessions[i]->Predict(test_inputs[i]);
        LOG("  Optimizer " << i << " output: " << output[0]);
        ASSERT(output.GetCount() == 1);
        ASSERT(std::isfinite(output[0]));
    }
    
    LOG("NetworkOptimization tests completed successfully!");
}