#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("TrainerBenchmark Test - Testing different training algorithms");
    
    // Create networks with different trainers to benchmark
    Session sgd_session, adam_session, adadelta_session;
    
    String base_net = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":3},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":15, \"activation\":\"tanh\"},\n"
        "\t{\"type\":\"regression\", \"neuron_count\":1},\n"
        "\t{\"trainer_placeholder\"}\n"
        "]\n";
    
    // SGD trainer
    String sgd_net = base_net;
    sgd_net.Replace("\"trainer_placeholder\"", 
        "{\"type\":\"sgd\", \"learning_rate\":0.01, \"momentum\":0.9, \"batch_size\":1, \"l2_decay\":0.001}");
    bool sgd_success = sgd_session.MakeLayers(sgd_net);
    
    // Adam trainer  
    String adam_net = base_net;
    adam_net.Replace("\"trainer_placeholder\"", 
        "{\"type\":\"adam\", \"learning_rate\":0.001, \"beta1\":0.9, \"beta2\":0.999, \"eps\":1e-8}");
    bool adam_success = adam_session.MakeLayers(adam_net);
    
    // Adadelta trainer
    String adadelta_net = base_net;
    adadelta_net.Replace("\"trainer_placeholder\"", 
        "{\"type\":\"adadelta\", \"learning_rate\":0.1, \"momentum\":0, \"batch_size\":1, \"l2_decay\":0.001}");
    bool adadelta_success = adadelta_session.MakeLayers(adadelta_net);
    
    LOG("  SGD network: " << (sgd_success ? "✓" : "✗"));
    LOG("  Adam network: " << (adam_success ? "✓" : "✗"));
    LOG("  Adadelta network: " << (adadelta_success ? "✓" : "✗"));
    
    ASSERT(sgd_success && adam_success && adadelta_success);
    
    // Create identical training data for all networks
    Vector<double> inputs[3] = {{0.1, 0.2, 0.3}, {0.1, 0.2, 0.3}, {0.1, 0.2, 0.3}};
    Vector<Session*> sessions = {&sgd_session, &adam_session, &adadelta_session};
    String trainer_names[] = {"SGD", "Adam", "Adadelta"};
    
    for (int i = 0; i < 3; i++) {
        SessionData& d = sessions[i]->Data();
        d.BeginDataResult(3, 4, 1, 0);  // 3 inputs, 4 samples, 1 output
        
        for (int j = 0; j < 4; j++) {
            d.SetData(j, 0, 0.1 + j*0.1).SetData(j, 1, 0.2 + j*0.1).SetData(j, 2, 0.3 - j*0.05);
            d.SetResult(j, 0, 0.5 + j*0.05);  // Simple target function
        }
        
        d.EndData();
        
        sessions[i]->StartTraining();
        
        // Train each for same number of iterations
        for (int iter = 0; iter < 20; iter++) {
            sessions[i]->Tick();
        }
        
        sessions[i]->StopTraining();
        
        // Test final prediction
        Vector<double> output = sessions[i]->Predict(inputs[i]);
        LOG("  " << trainer_names[i] << " final prediction: " << output[0]);
        
        ASSERT(output.GetCount() == 1);
        ASSERT(std::isfinite(output[0]));
    }
    
    LOG("TrainerBenchmark tests completed successfully!");
}