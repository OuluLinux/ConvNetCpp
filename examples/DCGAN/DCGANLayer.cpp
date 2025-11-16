#include "DCGAN.h"

DCGANLayer::DCGANLayer() {
}

void DCGANLayer::Init(int stride) {
    this->stride = stride;

    // Input dimensions for MNIST images (can be adjusted for other datasets)
    input_width  = 28;
    input_height = 28;
    input_depth  = 1;

    // Size for the random noise input to the generator
    int noise_size = 100;

    // DCGAN Discriminator: Convolutional network that takes in images
    String disc_t =	"[\n"
                    "\t{\"type\":\"input\", \"input_width\":" + IntStr(input_width) + ", \"input_height\":" + IntStr(input_height) + ", \"input_depth\":" + IntStr(input_depth) + "},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":64, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":128, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":256, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":1, \"pad\":0, \"filters\":512, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
                    "\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":128, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!disc.MakeLayers(disc_t))
        throw Exc("Discriminator network loading failed");

    // DCGAN Generator: Transposed convolutional network that generates images from noise
    String gen_t =	"[\n"
                    "\t{\"type\":\"input\", \"input_width\":" + IntStr(noise_size) + ", \"input_height\":1, \"input_depth\":1},\n"
                    "\t{\"type\":\"fc\", \"neuron_count\":7*7*256, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"unflatten\", \"width\":7, \"height\":7, \"depth\":256},\n"
                    "\t{\"type\":\"deconv\", \"sx\":5, \"sy\":5, \"stride\":2, \"pad\":2, \"filters\":128, \"activation\":\"relu\"},\n"  // 7x7 -> 14x14
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"deconv\", \"sx\":5, \"sy\":5, \"stride\":2, \"pad\":2, \"filters\":64, \"activation\":\"relu\"},\n"  // 14x14 -> 28x28
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"deconv\", \"sx\":3, \"sy\":3, \"stride\":1, \"pad\":1, \"filters\":1, \"activation\":\"tanh\"},\n"  // Final output layer
                    "\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":128, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!gen.MakeLayers(gen_t))
        throw Exc("Generator network loading failed");
}

void DCGANLayer::Train() {
    Net& gen_net = gen.GetNetwork();
    Net& disc_net = disc.GetNetwork();

    // Sample random noise for generator input
    Volume gen_input_vol;
    gen_input_vol.Init(100, 1, 1); // 100-dimensional noise vector
    for (int i = 0; i < 100; i++) {
        gen_input_vol.Set(i, 0, 0, 2.0 * Randomf() - 1.0); // Random values between -1 and 1
    }

    // Generate fake image
    Volume& fake_image = gen_net.Forward(gen_input_vol, true); // Enable gradient tracking for generator

    // Sample real image
    SessionData& data = disc.Data();
    int real_idx = Random(data.GetDataCount());
    const Vector<double>& real_image_vec = data.Get(real_idx); // Use reference to avoid copy
    Volume real_image;
    real_image.Set(28, 28, 1, real_image_vec);

    // Train Discriminator
    // Discriminator on real data (should output 1)
    Volume& real_output = disc_net.Forward(real_image, true);
    Vector<double> real_target(1);
    real_target[0] = 0.9; // Label smoothing: use 0.9 instead of 1
    double real_disc_loss = disc_net.Backward(real_target);
    disc_cost_av.Add(real_disc_loss);

    // Discriminator on fake data (should output 0)
    Volume& fake_output_from_gen = gen_net.Forward(gen_input_vol, false); // Generate fake data with no gradient tracking
    Volume& fake_output = disc_net.Forward(fake_output_from_gen, true); // Discriminator on fake data
    Vector<double> fake_target(1);
    fake_target[0] = 0.1; // Label smoothing: use 0.1 instead of 0
    double fake_disc_loss = disc_net.Backward(fake_target);
    disc_cost_av.Add(fake_disc_loss);

    // Update discriminator
    disc.GetTrainer().TrainImplem();

    // Train Generator (update generator to fool discriminator)
    // Generate another batch of fake images
    for (int i = 0; i < 100; i++) {
        gen_input_vol.Set(i, 0, 0, 2.0 * Randomf() - 1.0);
    }
    Volume& fake_image_gen = gen_net.Forward(gen_input_vol, true); // Generate with gradient tracking

    // Get discriminator's output on fake image
    Volume& disc_on_fake = disc_net.Forward(fake_image_gen, false); // No gradient tracking on discriminator

    // Update generator to make discriminator think fake images are real
    Vector<double> gen_target(1);
    gen_target[0] = 0.9; // Try to fool discriminator
    double gen_loss = gen_net.Backward(gen_target);
    gen_cost_av.Add(gen_loss);

    // Update generator
    gen.GetTrainer().TrainImplem();
}

void DCGANLayer::SampleInput() {
    // This is now handled in the Train() method
}

void DCGANLayer::SampleOutput() {
    // This is now handled in the Train() method
}

Volume& DCGANLayer::Generate(Volume& input) {
    Net& gen_net = gen.GetNetwork();
    Volume& xgen = gen_net.Forward(input, false);
    return xgen;
}