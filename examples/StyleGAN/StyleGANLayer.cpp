#include "StyleGAN.h"

StyleGANLayer::StyleGANLayer() {
}

void StyleGANLayer::Init(int stride, const StyleGANParams& params) {
    this->stride = stride;
    this->stylegan_params = params;

    // Input dimensions - using a standard resolution for training
    input_width = stylegan_params.target_resolution;
    input_height = stylegan_params.target_resolution;
    input_depth = 1;

    // StyleGAN uses a mapping network to transform latent vectors to w-space
    // Then an intermediate generator that receives style inputs at different scales
    int latent_size = stylegan_params.latent_dim;

    // Discriminator for StyleGAN - processes regular images
    String disc_t =	"[\n"
                    "\t{\"type\":\"input\", \"input_width\":" + IntStr(input_width) + ", \"input_height\":" + IntStr(input_height) + ", \"input_depth\":" + IntStr(input_depth) + "},\n"
                    "\t{\"type\":\"conv\", \"sx\":3, \"sy\":3, \"stride\":1, \"pad\":1, \"filters\":64, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":128, \"activation\":\"relu\"},\n"  // Downsample to 16x16
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":256, \"activation\":\"relu\"},\n"  // Downsample to 8x8
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":512, \"activation\":\"relu\"},\n"  // Downsample to 4x4
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"fc\", \"neuron_count\":1},\n"  // No activation for WGAN-style critic
                    "\t{\"type\":\"adam\", \"learning_rate\":" + DoubleStr(stylegan_params.learning_rate) + ", \"beta1\":0.0, \"beta2\":0.99, \"batch_size\":16, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!disc.MakeLayers(disc_t))
        throw Exc("Discriminator network loading failed");

    // Generator for StyleGAN - simplified version
    // In a real StyleGAN, the generator would have an intermediate network that receives
    // style inputs at different resolutions, but for a simplified implementation:
    String gen_t =	"[\n"
                    "\t{\"type\":\"input\", \"input_width\":" + IntStr(latent_size) + ", \"input_height\":1, \"input_depth\":1},\n"  // Latent vector z
                    "\t{\"type\":\"fc\", \"neuron_count\":4*4*512, \"activation\":\"relu\"},\n"  // Output features for 4x4 resolution
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"unflatten\", \"width\":4, \"height\":4, \"depth\":512},\n"
                    "\t{\"type\":\"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":256, \"activation\":\"relu\"},\n"  // 4x4 -> 8x8
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":128, \"activation\":\"relu\"},\n"  // 8x8 -> 16x16
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":1, \"activation\":\"tanh\"},\n"  // 16x16 -> 32x32
                    "\t{\"type\":\"adam\", \"learning_rate\":" + DoubleStr(stylegan_params.learning_rate) + ", \"beta1\":0.0, \"beta2\":0.99, \"batch_size\":16, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!gen.MakeLayers(gen_t))
        throw Exc("Generator network loading failed");
}

void StyleGANLayer::Train() {
    Net& gen_net = gen.GetNetwork();
    Net& disc_net = disc.GetNetwork();

    // Sample random latent vector for generator input
    Volume gen_input_vol;
    gen_input_vol.Init(stylegan_params.latent_dim, 1, 1); // 512-dimensional latent vector
    for (int i = 0; i < stylegan_params.latent_dim; i++) {
        gen_input_vol.Set(i, 0, 0, 2.0 * Randomf() - 1.0); // Random values between -1 and 1
    }

    // Generate fake image
    Volume& fake_image = gen_net.Forward(gen_input_vol, true); // Enable gradient tracking for generator

    // Sample real image
    SessionData& data = disc.Data();
    int real_idx = Random(data.GetDataCount());
    Vector<double> real_image_vec = data.Get(real_idx);
    
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
    for (int i = 0; i < stylegan_params.latent_dim; i++) {
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

void StyleGANLayer::SampleInput() {
    // This is now handled in the Train() method
}

void StyleGANLayer::SampleOutput() {
    // This is now handled in the Train() method
}

Volume& StyleGANLayer::Generate(Volume& input) {
    Net& gen_net = gen.GetNetwork();
    Volume& xgen = gen_net.Forward(input, false);
    return xgen;
}