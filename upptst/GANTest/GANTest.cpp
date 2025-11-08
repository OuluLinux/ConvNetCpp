#include <Core/Core.h>
#include <ConvNet/ConvNet.h>

using namespace Upp;
using namespace ConvNet;

CONSOLE_APP_MAIN
{
    SeedRandom();
    
    LOG("GAN Test - Testing core network functionality");
    
    // Test basic network creation similar to GAN architectures
    Session gen, disc;
    
    // Generator network: takes random input and generates data-like output
    String gen_net = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":1},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":10, \"activation\":\"tanh\"},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":1, \"activation\":\"tanh\"}\n"
        "]\n";
    
    // Discriminator network: takes data and determines if it's real or fake
    String disc_net = 
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":1},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":10, \"activation\":\"tanh\"},\n"
        "\t{\"type\":\"softmax\", \"class_count\":2}\n"
        "]\n";
    
    bool gen_success = gen.MakeLayers(gen_net);
    bool disc_success = disc.MakeLayers(disc_net);
    
    LOG("  Generator network creation: " << (gen_success ? "success" : "failed"));
    LOG("  Discriminator network creation: " << (disc_success ? "success" : "failed"));
    
    ASSERT(gen_success);
    ASSERT(disc_success);
    
    // Test forward pass through generator
    Vector<double> noise_input = {Randomf()};
    Vector<double> generated_output = gen.Predict(noise_input);
    
    LOG("  Generator input: " << noise_input[0]);
    LOG("  Generator output: " << generated_output[0]);
    
    ASSERT(std::isfinite(generated_output[0]));
    
    // Test forward pass through discriminator with real data
    Vector<double> real_data = {0.8}; // Simulated real data
    Vector<double> disc_real_output = disc.Predict(real_data);
    
    LOG("  Discriminator with real data: [" << real_data[0] << "] -> [" 
         << disc_real_output[0] << ", " << disc_real_output[1] << "]");
    
    // Discriminator output should be valid probabilities (sum to ~1)
    double disc_sum = disc_real_output[0] + disc_real_output[1];
    ASSERT(abs(disc_sum - 1.0) < 0.01);
    
    // Test forward pass through discriminator with generated data
    Vector<double> fake_data;
    fake_data.SetCount(generated_output.GetCount());
    for(int i = 0; i < generated_output.GetCount(); i++) {
        fake_data[i] = generated_output[i];
    } // Use generated output as fake data
    Vector<double> disc_fake_output = disc.Predict(fake_data);
    
    LOG("  Discriminator with fake data: [" << fake_data[0] << "] -> [" 
         << disc_fake_output[0] << ", " << disc_fake_output[1] << "]");
    
    // Discriminator output should be valid probabilities (sum to ~1)
    double disc_fake_sum = disc_fake_output[0] + disc_fake_output[1];
    ASSERT(abs(disc_fake_sum - 1.0) < 0.01);
    
    LOG("GAN tests completed successfully!");
}