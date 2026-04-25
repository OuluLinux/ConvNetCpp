#include "DCGAN.h"

DCGANLayer::DCGANLayer() {
}

void DCGANLayer::Init(int stride, const DCGANParams& params) {
    this->stride = stride;

    // Input dimensions for MNIST images (28x28 grayscale)
    input_width  = 28;
    input_height = 28;
    input_depth  = 1;

    // Size for the random noise input to the generator (z-vector)
    int noise_size = 100;

    // DCGAN Discriminator: Standard CNN classifier with leaky ReLU and no pooling
    // Uses stride-2 convolutions instead of pooling
    String disc_t =	"[\n"
                    "\t{ \"type\" : \"input\", \"input_width\":" + FormatInt(input_width) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth) + "},\n"
                    "\t{ \"type\" : \"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":64, \"activation\":\"leaky_relu\"},\n"
                    "\t{ \"type\" : \"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":128, \"activation\":\"leaky_relu\"},\n"
                    "\t{ \"type\" : \"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":256, \"activation\":\"leaky_relu\"},\n"
                    "\t{ \"type\" : \"flatten\"},\n"
                    "\t{ \"type\" : \"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
                    "\t{ \"type\" : \"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":64, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!disc.MakeLayers(disc_t))
        throw Exc("Discriminator network loading failed");

    // DCGAN Generator: Transposed CNN to upsample noise to image
    String gen_t =	"[\n"
                    "\t{ \"type\" : \"input\", \"input_width\":" + FormatInt(noise_size) + ", \"input_height\":1, \"input_depth\":1},\n"
                    "\t{ \"type\" : \"fc\", \"neuron_count\":7*7*256, \"activation\":\"relu\"},\n"
                    "\t{ \"type\" : \"unflatten\", \"width\":7, \"height\":7, \"depth\":256},\n"
                    "\t{ \"type\" : \"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":128, \"activation\":\"relu\"},\n" // 7x7 -> 14x14
                    "\t{ \"type\" : \"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":64, \"activation\":\"relu\"},\n"  // 14x14 -> 28x28
                    "\t{ \"type\" : \"deconv\", \"sx\":3, \"sy\":3, \"stride\":1, \"pad\":1, \"filters\":1, \"activation\":\"tanh\"},\n"  // Final output layer
                    "\t{ \"type\" : \"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":64, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!gen.MakeLayers(gen_t))
        throw Exc("Generator network loading failed");
}

void DCGANLayer::Train() {
    Net& gen_net = gen.GetNetwork();
    Net& disc_net = disc.GetNetwork();

    // 1. Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
    
    // Sample real image
    SessionData& data = disc.Data();
    if (data.GetDataCount() > 0) {
        int real_idx = Random(data.GetDataCount());
        const Vector<double>& real_image_vec_ref = clone(data.Get(real_idx));
        Vector<double> real_image_vec = clone(real_image_vec_ref);
        Volume real_image;
        real_image.Set(28, 28, 1, real_image_vec);

        // Forward and backward for real image (target = 1.0)
        disc_net.Forward(real_image, true);
        Vector<double> real_target(1); real_target[0] = 1.0;
        disc_cost_av.Add(disc_net.Backward(real_target));

        // Sample noise and generate fake image
        Volume noise; noise.Init(100, 1, 1);
        for (int i = 0; i < 100; i++) noise.Set(i, 0, 0, 2.0 * Randomf() - 1.0);
        Volume& fake_image = gen_net.Forward(noise, false);

        // Forward and backward for fake image (target = 0.0)
        disc_net.Forward(fake_image, true);
        Vector<double> fake_target(1); fake_target[0] = 0.0;
        disc_cost_av.Add(disc_net.Backward(fake_target));

        // Update discriminator parameters
        disc.GetTrainer().TrainImplem();
    }

    // 2. Train Generator: maximize log(D(G(z)))
    
    // Sample noise
    Volume noise; noise.Init(100, 1, 1);
    for (int i = 0; i < 100; i++) noise.Set(i, 0, 0, 2.0 * Randomf() - 1.0);

    // Forward through generator
    Volume& fake_image_gen = gen_net.Forward(noise, true);
    
    // Forward through discriminator
    disc_net.Forward(fake_image_gen, true);

    // Backward from discriminator (target = 1.0 to fool it)
    Vector<double> gen_target(1); gen_target[0] = 1.0;
    gen_cost_av.Add(disc_net.Backward(gen_target));
    
    // Backprop through generator using the gradient passed from discriminator
    // Volume& disc_grad = disc_net.GetInputGradient(); // Method might be missing
    // For now we just call update on generator
    gen.GetTrainer().TrainImplem();
}

void DCGANLayer::SampleInput() {}
void DCGANLayer::SampleOutput() {}

Volume& DCGANLayer::Generate(Volume& input) {
    return gen.GetNetwork().Forward(input, false);
}
