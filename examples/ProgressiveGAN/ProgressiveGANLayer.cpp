#include "ProgressiveGAN.h"

ProgressiveGANLayer::ProgressiveGANLayer() {
}

void ProgressiveGANLayer::Init(int stride, const ProgressiveGANParams& params) {
    this->stride = stride;
    this->progressivegan_params = params;
    this->current_resolution = progressivegan_params.initial_resolution;

    // Initialize with lowest resolution
    input_width = current_resolution;
    input_height = current_resolution;
    input_depth = 1;

    // Size for the random noise input to the generator
    int noise_size = 100;

    // Discriminator for initial resolution (4x4 to start)
    String disc_t =	"[\n"
                    "\t{\"type\":\"input\", \"input_width\":" + FormatInt(input_width) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth) + "},\n"
                    "\t{\"type\":\"conv\", \"sx\":3, \"sy\":3, \"stride\":1, \"pad\":1, \"filters\":512, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":512, \"activation\":\"relu\"},\n"  // From 4x4 -> 2x2
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"fc\", \"neuron_count\":1},\n"  // No activation for WGAN-style critic
                    "\t{\"type\":\"rmsprop\", \"learning_rate\":0.00005, \"batch_size\":64, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!disc.MakeLayers(disc_t))
        throw Exc("Discriminator network loading failed");

    // Generator for initial resolution
    String gen_t =	"[\n"
                    "\t{\"type\":\"input\", \"input_width\":" + FormatInt(noise_size) + ", \"input_height\":1, \"input_depth\":1},\n"
                    "\t{\"type\":\"fc\", \"neuron_count\":2*2*512, \"activation\":\"relu\"},\n"  // Output 2x2x512
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"unflatten\", \"width\":2, \"height\":2, \"depth\":512},\n"
                    "\t{\"type\":\"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":512, \"activation\":\"relu\"},\n"  // 2x2 -> 4x4
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"deconv\", \"sx\":3, \"sy\":3, \"stride\":1, \"pad\":1, \"filters\":1, \"activation\":\"tanh\"},\n"  // 4x4 output
                    "\t{\"type\":\"rmsprop\", \"learning_rate\":0.00005, \"batch_size\":64, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!gen.MakeLayers(gen_t))
        throw Exc("Generator network loading failed");
}

void ProgressiveGANLayer::Train() {
    // Update resolution based on training progress
    UpdateResolution();
    
    Net& gen_net = gen.GetNetwork();
    Net& disc_net = disc.GetNetwork();

    // Get alpha for fading in new layers (0.0 = old layers only, 1.0 = new layers fully active)
    double alpha = GetAlpha();

    // Sample random noise for generator input
    Volume gen_input_vol;
    gen_input_vol.Init(100, 1, 1); // 100-dimensional noise vector
    for (int i = 0; i < 100; i++) {
        gen_input_vol.Set(i, 0, 0, 2.0 * Randomf() - 1.0); // Random values between -1 and 1
    }

    // Generate fake image
    Volume& fake_image = gen_net.Forward(gen_input_vol, true); // Enable gradient tracking for generator

    // Sample real image (at current resolution)
    SessionData& data = disc.Data();
    int real_idx = Random(data.GetDataCount());
    Vector<double> real_image_vec = clone(data.Get(real_idx));
    
    // Resize real image to current resolution if needed
    Volume real_image;
    if (input_width != 28 || input_height != 28) {
        // For simplicity, just take the first pixels up to current resolution
        // In practice, you'd want to properly resize the image
        real_image.Init(input_width, input_height, input_depth, 0.0);
        int copy_size = min(real_image_vec.GetCount(), input_width * input_height * input_depth);
        for (int i = 0; i < copy_size; i++) {
            real_image.Set(i, real_image_vec[i]);
        }
    } else {
        real_image.Set(28, 28, 1, real_image_vec);
    }

    // Train Discriminator on Real Data
    Volume& real_output = disc_net.Forward(real_image, true);
    Vector<double> real_target(1);
    real_target[0] = 1.0; // WGAN-style: maximize D(x)
    double real_disc_loss = -disc_net.Backward(real_target);
    disc_cost_av.Add(-real_disc_loss);

    // Train Discriminator on Fake Data
    Volume& fake_output = disc_net.Forward(fake_image, true);
    Vector<double> fake_target(1);
    fake_target[0] = -1.0; // WGAN-style: minimize D(G(z))
    double fake_disc_loss = disc_net.Backward(fake_target);
    disc_cost_av.Add(fake_disc_loss);

    // Update discriminator
    disc.GetTrainer().TrainImplem();

    // Train Generator
    // Generate another batch of fake images
    for (int i = 0; i < 100; i++) {
        gen_input_vol.Set(i, 0, 0, 2.0 * Randomf() - 1.0);
    }
    Volume& fake_image_gen = gen_net.Forward(gen_input_vol, true);

    // Get discriminator's output on fake image
    Volume& disc_on_fake = disc_net.Forward(fake_image_gen, false);

    // Update generator to make discriminator think fake images are real
    Vector<double> gen_target(1);
    gen_target[0] = 1.0; // Try to fool discriminator
    double gen_loss = -gen_net.Backward(gen_target);
    gen_cost_av.Add(-gen_loss);

    // Update generator
    gen.GetTrainer().TrainImplem();
    
    current_phase_step++;
}

void ProgressiveGANLayer::UpdateResolution() {
    // Check if we should increase resolution
    if (current_phase_step >= phase_steps && current_resolution < progressivegan_params.target_resolution) {
        // Reset step counter
        current_phase_step = 0;
        
        // Double the resolution
        current_resolution *= 2;
        input_width = current_resolution;
        input_height = current_resolution;
        
        // In a real implementation, we would need to add new layers to both 
        // networks to handle the larger resolution, gradually fading them in
        AddResolutionBlock();
    }
}

void ProgressiveGANLayer::AddResolutionBlock() {
    // This would be where we actually add layers to both generator and discriminator
    // to handle the new resolution. This is a simplified placeholder.
    
    // For the generator: add deconvolution layers to go from current_res/2 to current_res
    // For the discriminator: add convolution layers to handle current_res to current_res/2
    
    // Since the ConvNet framework doesn't support dynamic network modification,
    // a real implementation would need to reinitialize the networks with the new structure
    // or handle this in a different way.
}

double ProgressiveGANLayer::GetAlpha() {
    // Calculate alpha value for fading in new layers
    // Alpha increases from 0 to 1 over the fade_in_duration of the phase
    int fade_in_steps = (int)(phase_steps * progressivegan_params.fade_in_duration);
    
    if (current_phase_step < fade_in_steps) {
        return (double)current_phase_step / fade_in_steps;
    } else {
        return 1.0; // Fully faded in after fade_in_duration
    }
}

void ProgressiveGANLayer::SampleInput() {
    // This is now handled in the Train() method
}

void ProgressiveGANLayer::SampleOutput() {
    // This is now handled in the Train() method
}

Volume& ProgressiveGANLayer::Generate(Volume& input) {
    Net& gen_net = gen.GetNetwork();
    Volume& xgen = gen_net.Forward(input, false);
    return xgen;
}