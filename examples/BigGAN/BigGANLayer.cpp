#include "BigGAN.h"

BigGANLayer::BigGANLayer() {
}

void BigGANLayer::Init(int stride, int noise_size, int num_classes, int img_width, int img_height, int img_depth) {
	this->stride = stride;
	this->noise_size = noise_size;
	this->num_classes = num_classes;
	this->input_width = img_width;
	this->input_height = img_height;
	this->input_depth = img_depth;

	// BigGAN architecture - needs to handle high-resolution images with self-attention
	// The model uses spectral normalization in discriminator and self-attention layers
	
	// Discriminator network with self-attention
	String disc_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + FormatInt(input_width) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth) + "},\n"
					"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":64, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"conv\", \"filter_width\":4, \"filter_height\":4, \"filter_count\":128, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":256, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":256, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":512, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"selfattention\", \"filter_count\":512},\n"  // Self-attention layer is key to BigGAN
					"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":512, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"globalavgpool\"},\n"  // Global average pooling
					"\t{\"type\":\"conditional_batch_norm\"},\n"  // Conditional batch normalization
					"\t{\"type\":\"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
					"\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":16, \"l2_decay\":0.0001}\n"  // Small batch for BigGAN
					"]\n";

	if (!disc.MakeLayers(disc_t))
		throw Exc("Discriminator network loading failed");

	// Generator network with self-attention and conditioning
	String gen_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + FormatInt(noise_size + num_classes) + ", \"input_height\":1, \"input_depth\":1},\n"  // Combined noise and condition
					"\t{\"type\":\"linear_projection\", \"output_size\":4096},\n"  // Project to intermediate representation
					"\t{\"type\":\"biggan_generator_block\", \"out_channels\":512, \"input_size\":4, \"output_size\":4},\n"  // Initial block
					"\t{\"type\":\"biggan_generator_block\", \"out_channels\":256, \"input_size\":4, \"output_size\":8},\n"
					"\t{\"type\":\"biggan_generator_block\", \"out_channels\":128, \"input_size\":8, \"output_size\":16},\n"
					"\t{\"type\":\"biggan_generator_block\", \"out_channels\":64, \"input_size\":16, \"output_size\":32},\n"
					"\t{\"type\":\"selfattention\", \"filter_count\":64},\n"  // Self-attention at 32x32
					"\t{\"type\":\"biggan_generator_block\", \"out_channels\":32, \"input_size\":32, \"output_size\":64},\n"
					"\t{\"type\":\"biggan_generator_block\", \"out_channels\":16, \"input_size\":64, \"output_size\":128},\n"
					"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":3, \"stride\":1, \"pad\":1, \"activation\":\"tanh\"},\n"  // Output layer
					"\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":16, \"l2_decay\":0.0001}\n"  // Small batch for BigGAN
					"]\n";

	if (!gen.MakeLayers(gen_t))
		throw Exc("Generator network loading failed");
}

void BigGANLayer::Train() {
	Net& gen_net = gen.GetNetwork();
	Net& disc_net = disc.GetNetwork();

	// Sample random noise for generator input
	Volume gen_input_vol;
	gen_input_vol.Init(noise_size + num_classes, 1, 1); // Combined input: noise + condition

	// Fill with random noise (truncated normal distribution to implement truncation trick)
	for (int i = 0; i < noise_size; i++) {
		double noise_val = 2.0 * Randomf() - 1.0; // Random values between -1 and 1
		// Apply truncation trick - limit the noise to a range based on truncation factor
		noise_val = max(-truncation_factor, min(truncation_factor, noise_val));
		gen_input_vol.Set(i, 0, 0, noise_val);
	}

	// Add conditioning information (one-hot encoded class label)
	int condition_label = Random(num_classes); // Random class for this training step
	for (int i = noise_size; i < noise_size + num_classes; i++) {
		if (i == noise_size + condition_label) {
			gen_input_vol.Set(i, 0, 0, 1.0);  // 1 for the condition class
		} else {
			gen_input_vol.Set(i, 0, 0, 0.0);  // 0 for other classes
		}
	}

	// Generate fake image with conditioning
	Volume& fake_image = gen_net.Forward(gen_input_vol, true); // Enable gradient tracking for generator

	// Sample real image
	SessionData& data = disc.Data();
	int real_idx = Random(data.GetDataCount());
	const Vector<double>& real_image_vec = clone(data.Get(real_idx)); // Use reference to avoid copy

	// Get the corresponding label for this real image
	int real_label = data.GetLabel(real_idx) % num_classes; // Assuming labels are stored in SessionData

	// Train Discriminator
	// Discriminator on real data (should output 1)
	Volume real_input(input_width, input_height, input_depth);
	for (int w = 0; w < input_width; w++) {
		for (int h = 0; h < input_height; h++) {
			for (int d = 0; d < input_depth; d++) {
				real_input.Set(w, h, d, real_image_vec[w*input_height*input_depth + h*input_depth + d]);
			}
		}
	}

	// Discriminator on real data
	Volume& real_output = disc_net.Forward(real_input, true);
	Vector<double> real_target(1);
	real_target[0] = 0.9; // Label smoothing: use 0.9 instead of 1
	double real_disc_loss = disc_net.Backward(real_target);
	disc_cost_av.Add(real_disc_loss);

	// Discriminator on fake data (should output 0)
	// Discriminator on fake data
	Volume& fake_output = disc_net.Forward(fake_image, true); // Discriminator on fake data
	Vector<double> fake_target(1);
	fake_target[0] = 0.1; // Label smoothing: use 0.1 instead of 0
	double fake_disc_loss = disc_net.Backward(fake_target);
	disc_cost_av.Add(fake_disc_loss);

	// Update discriminator
	disc.GetTrainer().TrainImplem();

	// Train Generator (update generator to fool discriminator)
	// Generate another batch of fake images with conditioning
	for (int i = 0; i < noise_size; i++) {
		double noise_val = 2.0 * Randomf() - 1.0; // Random values between -1 and 1
		// Apply truncation trick - limit the noise to a range based on truncation factor
		noise_val = max(-truncation_factor, min(truncation_factor, noise_val));
		gen_input_vol.Set(i, 0, 0, noise_val);
	}
	// Reset conditioning
	for (int i = noise_size; i < noise_size + num_classes; i++) {
		if (i == noise_size + condition_label) {
			gen_input_vol.Set(i, 0, 0, 1.0);  // 1 for the condition class
		} else {
			gen_input_vol.Set(i, 0, 0, 0.0);  // 0 for other classes
		}
	}

	// Generate fake image with conditioning (with gradient tracking)
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

void BigGANLayer::SampleInput() {
	// This is now handled in the Train() method
}

void BigGANLayer::SampleOutput() {
	// This is now handled in the Train() method
}

Volume& BigGANLayer::Generate(Volume& noise_input, int condition_label) {
	Net& gen_net = gen.GetNetwork();

	// Combine noise and conditioning information
	Volume input_with_condition(noise_size + num_classes, 1, 1);

	// Fill with noise input
	for (int i = 0; i < noise_size; i++) {
		double noise_val = clone(noise_input.Get(i, 0, 0));
		// Apply truncation trick if needed
		noise_val = max(-truncation_factor, min(truncation_factor, noise_val));
		input_with_condition.Set(i, 0, 0, noise_val);
	}

	// Add conditioning information (one-hot encoded class label)
	for (int i = noise_size; i < noise_size + num_classes; i++) {
		if (i == noise_size + condition_label) {
			input_with_condition.Set(i, 0, 0, 1.0);  // 1 for the condition class
		} else {
			input_with_condition.Set(i, 0, 0, 0.0);  // 0 for other classes
		}
	}

	Volume& xgen = gen_net.Forward(input_with_condition, false);
	return xgen;
}