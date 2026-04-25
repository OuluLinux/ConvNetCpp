#include "SAGAN.h"

SAGANLayer::SAGANLayer() {
}

void SAGANLayer::Init(int stride, const SAGANParams& params) {
    this->stride = stride;
    this->sagan_params = params;

    // Input dimensions - standard for image generation tasks
    input_width = 32;  // Using 32x32 for computational efficiency
    input_height = 32;
    input_depth = 1;

    int latent_size = sagan_params.latent_dim;

    // Discriminator for SAGAN - includes self-attention layers conceptually
    // In a real SAGAN, there would be attention layers, but using available layers
    String disc_t =	"[\n"
                    "\t{\"type\":\"input\", \"input_width\":" + FormatInt(input_width) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth) + "},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":64, \"activation\":\"relu\"},\n"  // 32x32 -> 16x16
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":128, \"activation\":\"relu\"},\n"  // 16x16 -> 8x8
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":256, \"activation\":\"relu\"},\n"  // 8x8 -> 4x4
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":1, \"pad\":0, \"filters\":512, \"activation\":\"relu\"},\n"  // 4x4 -> 1x1 (with appropriate padding/filter size)
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"fc\", \"neuron_count\":1},\n"  // No activation for WGAN-style critic
                    "\t{\"type\":\"adam\", \"learning_rate\":" + FormatDouble(sagan_params.learning_rate) + ", \"beta1\":0.0, \"beta2\":0.99, \"batch_size\":16, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!disc.MakeLayers(disc_t))
        throw Exc("Discriminator network loading failed");

    // Generator for SAGAN - includes conceptually where attention would be applied
    String gen_t =	"[\n"
                    "\t{\"type\":\"input\", \"input_width\":" + FormatInt(latent_size) + ", \"input_height\":1, \"input_depth\":1},\n"  // Latent vector z
                    "\t{\"type\":\"fc\", \"neuron_count\":4*4*512, \"activation\":\"relu\"},\n"  // Output features for 4x4 resolution
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"unflatten\", \"width\":4, \"height\":4, \"depth\":512},\n"
                    "\t{\"type\":\"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":256, \"activation\":\"relu\"},\n"  // 4x4 -> 8x8
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":128, \"activation\":\"relu\"},\n"  // 8x8 -> 16x16
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":1, \"activation\":\"tanh\"},\n"  // 16x16 -> 32x32
                    "\t{\"type\":\"adam\", \"learning_rate\":" + FormatDouble(sagan_params.learning_rate) + ", \"beta1\":0.0, \"beta2\":0.99, \"batch_size\":16, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!gen.MakeLayers(gen_t))
        throw Exc("Generator network loading failed");
}

void SAGANLayer::Train() {
    Net& gen_net = gen.GetNetwork();
    Net& disc_net = disc.GetNetwork();

    // Sample random latent vector for generator input
    Volume gen_input_vol;
    gen_input_vol.Init(sagan_params.latent_dim, 1, 1); // 128-dimensional latent vector
    for (int i = 0; i < sagan_params.latent_dim; i++) {
        gen_input_vol.Set(i, 0, 0, 2.0 * Randomf() - 1.0); // Random values between -1 and 1
    }

    // Generate fake image
    Volume& fake_image = gen_net.Forward(gen_input_vol, true); // Enable gradient tracking for generator

    // Sample real image
    SessionData& data = disc.Data();
    int real_idx = Random(data.GetDataCount());
    Vector<double> real_image_vec = clone(data.Get(real_idx));
    
    // Resize real image to match generator output if needed
    Volume real_image;
    if (input_width * input_height != real_image_vec.GetCount()) {
        // For simplicity, just take the required number of pixels
        // In practice, you'd want to properly resize the image
        real_image.Init(input_width, input_height, input_depth, 0.0);
        int copy_size = min(real_image_vec.GetCount(), input_width * input_height * input_depth);
        for (int i = 0; i < copy_size; i++) {
            real_image.Set(i, real_image_vec[i]);
        }
    } else {
        real_image.Set(input_width, input_height, input_depth, real_image_vec);
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
    for (int i = 0; i < sagan_params.latent_dim; i++) {
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
}

void SAGANLayer::SampleInput() {
    // This is now handled in the Train() method
}

void SAGANLayer::SampleOutput() {
    // This is now handled in the Train() method
}

Volume& SAGANLayer::Generate(Volume& input) {
    Net& gen_net = gen.GetNetwork();
    Volume& xgen = gen_net.Forward(input, false);
    return xgen;
}