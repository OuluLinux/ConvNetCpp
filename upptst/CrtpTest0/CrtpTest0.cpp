#include <Core/Core.h>
#include <ConvNet/ConvNet.h>
#include <ConvNet/CrtpLayers.h>

using namespace Upp;
using namespace ConvNet;

using namespace Upp;

CONSOLE_APP_MAIN
{
    // Test the new CRTP architecture
    LOG("Testing CRTP Architecture");
    
    try {
        // Test Input Layer
        ConvNet::InputLayer inputLayer;
        inputLayer.Init(28, 28, 1); // 28x28 grayscale image
        LOG("Input layer created and initialized");
        
        // Test Fully Connected Layer
        ConvNet::FullyConnLayer fcLayer;
        fcLayer.neuron_count = 10;
        fcLayer.bias_pref = 0.0;
        fcLayer.l1_decay_mul = 0.0;
        fcLayer.l2_decay_mul = 1.0;
        fcLayer.Init(1, 1, 784); // Input from flattened 28x28
        LOG("Fully connected layer created and initialized");
        
        // Test basic network flow with CRTP layers
        ConvNet::Volume inputVolume;
        inputVolume.Init(28, 28, 1, 0.5); // Initialize with some sample data
        
        // Forward pass through input layer
        auto& output1 = inputLayer.Forward(inputVolume);
        LOG("Input layer forward pass completed");
        
        // Flatten the output for FC layer (simplified)
        ConvNet::Volume flattened;
        flattened.Init(1, 1, 784, 0.5); // Simulate flattened input
        
        // Forward pass through FC layer
        auto& output2 = fcLayer.Forward(flattened);
        LOG("Fully connected layer forward pass completed");
        
        // Test type-erased wrapper
        LOG("Testing type-erased wrapper...");
        ConvNet::InputLayer inputForWrapper{};
        inputForWrapper.Init(28, 28, 1);
        ConvNet::LayerWrapper wrappedInputLayer(std::move(inputForWrapper));
        LOG("Wrapped input layer created and initialized: " << wrappedInputLayer.ToString());
        
        ConvNet::FullyConnLayer fcForWrapper{};
        fcForWrapper.neuron_count = 10;
        fcForWrapper.bias_pref = 0.0;
        fcForWrapper.l1_decay_mul = 0.0;
        fcForWrapper.l2_decay_mul = 1.0;
        fcForWrapper.Init(1, 1, 784);
        ConvNet::LayerWrapper wrappedFcLayer(std::move(fcForWrapper));
        LOG("Wrapped fully connected layer created and initialized: " << wrappedFcLayer.ToString());
        
        LOG("CRTP Architecture test completed successfully!");
    }
    catch (const std::exception& e) {
        LOG("Error in CRTP test: " << e.what());
        exit(1);
    }
    catch (...) {
        LOG("Unknown error in CRTP test");
        exit(1);
    }
}