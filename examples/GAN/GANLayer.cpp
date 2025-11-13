#include "GAN.h"
#include <Math.h>



GANLayer::GANLayer() {
}

void GANLayer::Init(int stride) {
	this->stride = stride;

	input_width  = 28;
	input_height = 28;
	input_depth  = 1;
	
	// Size for the random noise input to the generator
	int noise_size = 100;

	String disc_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + IntStr(input_width) + ", \"input_height\":" + IntStr(input_height) + ", \"input_depth\":" + IntStr(input_depth) + "},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":512, \"activation\":\"leaky_relu\"},\n"
					"\t{\"type\":\"dropout\", \"drop_prob\":0.3},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":256, \"activation\":\"leaky_relu\"},\n"
					"\t{\"type\":\"dropout\", \"drop_prob\":0.3},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
					"\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":128}\n"
					"]\n";

	if (!disc.MakeLayers(disc_t))
		throw Exc("Discriminator network loading failed");

	String gen_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + IntStr(noise_size) + ", \"input_height\":1, \"input_depth\":1},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":256, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"batch_normalization\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"batch_normalization\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":1024, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"batch_normalization\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":784, \"activation\":\"tanh\"},\n"  // Output 784 values for 28x28 image
					"\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":128}\n"
					"]\n";

	if (!gen.MakeLayers(gen_t))
		throw Exc("Generator network loading failed");
}

void GANLayer::Train() {
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
	fake_image.Reshape(28, 28, 1); // Reshape to 28x28x1 (MNIST format)

	// Sample real image
	SessionData& data = disc.Data();
	int real_idx = Random(data.GetDataCount());
	Vector<double> real_image_vec = data.Get(real_idx);
	Volume real_image;
	real_image.Set(28, 28, 1, real_image_vec);

	// Train Discriminator
	// Discriminator on real data (should output 1)
	Volume& real_output = disc_net.Forward(real_image, true);
	Vector<double> real_target(1);
	real_target[0] = 0.9; // Label smoothing: use 0.9 instead of 1
	double real_disc_loss = disc_net.Backward(real_output, real_target);
	disc_cost_av.Add(real_disc_loss);

	// Discriminator on fake data (should output 0)
	Volume& fake_output = disc_net.Forward(fake_image, true);
	Vector<double> fake_target(1);
	fake_target[0] = 0.1; // Label smoothing: use 0.1 instead of 0
	double fake_disc_loss = disc_net.Backward(fake_output, fake_target);
	disc_cost_av.Add(fake_disc_loss);

	// Update discriminator
	disc.GetTrainer().TrainImplem();

	// Train Generator (update generator to fool discriminator)
	// Generate another batch of fake images
	for (int i = 0; i < 100; i++) {
		gen_input_vol.Set(i, 0, 0, 2.0 * Randomf() - 1.0);
	}
	Volume& fake_image_gen = gen_net.Forward(gen_input_vol, true);
	fake_image_gen.Reshape(28, 28, 1);

	// Get discriminator's output on fake image
	Volume& disc_on_fake = disc_net.Forward(fake_image_gen, false); // No gradient tracking on discriminator
	
	// Update generator to make discriminator think fake images are real
	Vector<double> gen_target(1);
	gen_target[0] = 0.9; // Try to fool discriminator
	double gen_loss = gen_net.Backward(disc_on_fake, gen_target);
	gen_cost_av.Add(gen_loss);

	// Update generator
	gen.GetTrainer().TrainImplem();
}

void GANLayer::SampleInput() {
	// This is now handled in the Train() method
}

void GANLayer::SampleOutput() {
	// This is now handled in the Train() method
}

Volume& GANLayer::Generate(Volume& input) {
	Net& gen_net = gen.GetNetwork();
	Volume& xgen = gen_net.Forward(input, false);
	xgen.Reshape(28, 28, 1); // Ensure proper image format
	return xgen;
}