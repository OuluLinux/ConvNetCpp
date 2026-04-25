#include "WGAN.h"

WGANLayer::WGANLayer() {
}

void WGANLayer::Init(int stride, const WGANParams& params) {
    this->stride = stride;
    this->wgan_params = params;

    // Input dimensions for MNIST images (can be adjusted for other datasets)
    input_width  = 28;
    input_height = 28;
    input_depth  = 1;

    // Size for the random noise input to the generator
    int noise_size = 100;

    // WGAN Discriminator (Critic): Uses linear activation (no sigmoid) to output unbounded values
    String disc_t =	"[\n"
                    "\t{\"type\":\"input\", \"input_width\":" + FormatInt(input_width) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth) + "},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":64, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":128, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":256, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"conv\", \"sx\":4, \"sy\":4, \"stride\":1, \"pad\":0, \"filters\":512, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"fc\", \"neuron_count\":1},\n"  // No activation for WGAN critic (linear)
                    "\t{\"type\":\"rmsprop\", \"learning_rate\":0.00005, \"batch_size\":64, \"l2_decay\":0.0001}\n"  // Lower learning rate for WGAN
                    "]\n";

    if (!disc.MakeLayers(disc_t))
        throw Exc("Discriminator network loading failed");

    // WGAN Generator: Similar to DCGAN but with different learning parameters
    String gen_t =	"[\n"
                    "\t{\"type\":\"input\", \"input_width\":" + FormatInt(noise_size) + ", \"input_height\":1, \"input_depth\":1},\n"
                    "\t{\"type\":\"fc\", \"neuron_count\":7*7*256, \"activation\":\"relu\"},\n"
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"unflatten\", \"width\":7, \"height\":7, \"depth\":256},\n"
                    "\t{\"type\":\"deconv\", \"sx\":5, \"sy\":5, \"stride\":2, \"pad\":2, \"filters\":128, \"activation\":\"relu\"},\n"  // 7x7 -> 14x14
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"deconv\", \"sx\":5, \"sy\":5, \"stride\":2, \"pad\":2, \"filters\":64, \"activation\":\"relu\"},\n"  // 14x14 -> 28x28
                    "\t{\"type\":\"lrn\", \"k\":2, \"n\":5, \"alpha\":0.0001, \"beta\":0.75},\n"
                    "\t{\"type\":\"deconv\", \"sx\":3, \"sy\":3, \"stride\":1, \"pad\":1, \"filters\":1, \"activation\":\"tanh\"},\n"  // Final output layer
                    "\t{\"type\":\"rmsprop\", \"learning_rate\":0.00005, \"batch_size\":64, \"l2_decay\":0.0001}\n"  // Lower learning rate for WGAN
                    "]\n";

    if (!gen.MakeLayers(gen_t))
        throw Exc("Generator network loading failed");
}

void WGANLayer::Train() {
    Net& gen_net = gen.GetNetwork();
    Net& disc_net = disc.GetNetwork();

    // Clip discriminator weights to enforce Lipschitz constraint
    for (auto& param : disc.GetNetwork().GetParametersAndGradients()) {
        Volume& vol = *param.volume;
        for (int i = 0; i < vol.GetLength(); i++) {
            double val = vol.Get(i);
            if (val > wgan_params.clip_value) {
                vol.Set(i, wgan_params.clip_value);
            } else if (val < -wgan_params.clip_value) {
                vol.Set(i, -wgan_params.clip_value);
            }
        }
    }

    // Train Discriminator (Critic) more often than Generator
    for (int critic_iter = 0; critic_iter < wgan_params.critic_iter; critic_iter++) {
        // Sample random noise for generator input
        Volume gen_input_vol;
        gen_input_vol.Init(100, 1, 1); // 100-dimensional noise vector
        for (int i = 0; i < 100; i++) {
            gen_input_vol.Set(i, 0, 0, 2.0 * Randomf() - 1.0); // Random values between -1 and 1
        }

        // Generate fake image
        Volume& fake_image = gen_net.Forward(gen_input_vol, false); // No gradient tracking for generator during discriminator training

        // Sample real image
        SessionData& data = disc.Data();
        int real_idx = Random(data.GetDataCount());
        const Vector<double>& real_image_vec = data.Get(real_idx); // Use reference to avoid copy
        Volume real_image;
        real_image.Set(28, 28, 1, real_image_vec);

        // Train Discriminator on Real Data
        Volume& real_output = disc_net.Forward(real_image, true);
        // For WGAN, we want to maximize D(x) where x is real data
        // So we set target to 1 (though in practice WGAN uses -1 for real)
        Vector<double> real_target(1);
        real_target[0] = 1.0; // In WGAN, this would be used differently
        double real_disc_loss = -disc_net.Backward(real_target); // Negative because WGAN maximizes D(x)
        disc_cost_av.Add(-real_disc_loss); // Store positive value for display

        // Train Discriminator on Fake Data
        Volume& fake_output = disc_net.Forward(fake_image, true);
        Vector<double> fake_target(1);
        fake_target[0] = -1.0; // In WGAN, minimize D(G(z)), so target is -1
        double fake_disc_loss = disc_net.Backward(fake_target);
        disc_cost_av.Add(fake_disc_loss);

        // Update discriminator
        disc.GetTrainer().TrainImplem();
    }

    // Train Generator (only once every few critic iterations)
    {
        // Generate noise input
        Volume gen_input_vol;
        gen_input_vol.Init(100, 1, 1);
        for (int i = 0; i < 100; i++) {
            gen_input_vol.Set(i, 0, 0, 2.0 * Randomf() - 1.0);
        }

        // Generate fake image with gradient tracking
        Volume& fake_image_gen = gen_net.Forward(gen_input_vol, true);

        // Get critic's output on fake image
        Volume& disc_on_fake = disc_net.Forward(fake_image_gen, true);

        // Update generator to make critic think fake images are real
        // In WGAN, we want to maximize D(G(z)), so target is +1
        Vector<double> gen_target(1);
        gen_target[0] = 1.0; // Try to fool critic
        double gen_loss = -gen_net.Backward(gen_target); // Negative because we want to maximize D(G(z))
        gen_cost_av.Add(-gen_loss); // Store positive value for display

        // Update generator
        gen.GetTrainer().TrainImplem();
    }
}

void WGANLayer::SampleInput() {
    // This is now handled in the Train() method
}

void WGANLayer::SampleOutput() {
    // This is now handled in the Train() method
}

Volume& WGANLayer::Generate(Volume& input) {
    Net& gen_net = gen.GetNetwork();
    Volume& xgen = gen_net.Forward(input, false);
    return xgen;
}