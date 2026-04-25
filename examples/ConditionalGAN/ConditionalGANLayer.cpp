#include "ConditionalGAN.h"

ConditionalGANLayer::ConditionalGANLayer() {
}

void ConditionalGANLayer::Init(int stride, int num_classes, int noise_size) {
	this->stride = stride;
	this->num_classes = num_classes;
	this->noise_size = noise_size;

	input_width  = 28;
	input_height = 28;
	input_depth  = 1;

	// Conditional GAN architecture requires conditioning information to be fed
	// to both the generator and discriminator

	// Discriminator network - takes image and conditioning information
	String disc_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + FormatInt(input_width + num_classes) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth) + "},\n"  // Concatenated image and condition
					"\t{\"type\":\"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"dropout\", \"drop_prob\":0.3},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":256, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"dropout\", \"drop_prob\":0.3},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
					"\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":128, \"l2_decay\":0.0001}\n"
					"]\n";

	if (!disc.MakeLayers(disc_t))
		throw Exc("Discriminator network loading failed");

	// Generator network - takes noise and conditioning information
	String gen_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + FormatInt(noise_size + num_classes) + ", \"input_height\":1, \"input_depth\":1},\n"  // Combined noise and condition
					"\t{\"type\":\"fc\", \"neuron_count\":256, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":512, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":1024, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":" + FormatInt(input_width * input_height * input_depth) + ", \"activation\":\"tanh\"},\n"  // Output image size
					"\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":128, \"l2_decay\":0.0001}\n"
					"]\n";

	if (!gen.MakeLayers(gen_t))
		throw Exc("Generator network loading failed");
}

void ConditionalGANLayer::Train() {
	Net& gen_net = gen.GetNetwork();
	Net& disc_net = disc.GetNetwork();

	// Sample random noise for generator input
	Volume gen_input_vol;
	gen_input_vol.Init(noise_size + num_classes, 1, 1); // Combined input: noise + condition
	
	// Fill with random noise
	for (int i = 0; i < noise_size; i++) {
		gen_input_vol.Set(i, 0, 0, 2.0 * Randomf() - 1.0); // Random values between -1 and 1
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
	
	// Create conditioned real image input (concatenate real image with condition)
	// Reshape real_image to match the expected input format for discriminator
	Volume conditioned_real_image(input_width + num_classes, input_height, input_depth);
	
	// Fill the image part
	for (int w = 0; w < input_width; w++) {
		for (int h = 0; h < input_height; h++) {
			for (int d = 0; d < input_depth; d++) {
				conditioned_real_image.Set(w, h, d, real_image_vec[w*input_height*input_depth + h*input_depth + d]);
			}
		}
	}
	
	// Add conditioning information to the real image
	for (int h = 0; h < input_height; h++) {
		for (int d = 0; d < input_depth; d++) {
			for (int c = 0; c < num_classes; c++) {
				int w = input_width + c;  // After the image data
				if (c == real_label) {
					conditioned_real_image.Set(w, h, d, 1.0);  // 1 for the condition class
				} else {
					conditioned_real_image.Set(w, h, d, 0.0);  // 0 for other classes
				}
			}
		}
	}

	// Train Discriminator
	// Discriminator on conditioned real data (should output 1)
	Volume& real_output = disc_net.Forward(conditioned_real_image, true);
	Vector<double> real_target(1);
	real_target[0] = 0.9; // Label smoothing: use 0.9 instead of 1
	double real_disc_loss = disc_net.Backward(real_target);
	disc_cost_av.Add(real_disc_loss);

	// Discriminator on conditioned fake data (should output 0)
	// Create conditioned fake image input (concatenate fake image with condition)
	Volume conditioned_fake_image(input_width + num_classes, input_height, input_depth);
	
	// Fill the image part with fake image
	for (int w = 0; w < input_width; w++) {
		for (int h = 0; h < input_height; h++) {
			for (int d = 0; d < input_depth; d++) {
				conditioned_fake_image.Set(w, h, d, fake_image.Get(w, h, d));
			}
		}
	}
	
	// Add conditioning information to the fake image (same as generator condition)
	for (int h = 0; h < input_height; h++) {
		for (int d = 0; d < input_depth; d++) {
			for (int c = 0; c < num_classes; c++) {
				int w = input_width + c;  // After the image data
				if (c == condition_label) {
					conditioned_fake_image.Set(w, h, d, 1.0);  // 1 for the condition class
				} else {
					conditioned_fake_image.Set(w, h, d, 0.0);  // 0 for other classes
				}
			}
		}
	}
	
	// Discriminator on conditioned fake data
	Volume& fake_output = disc_net.Forward(conditioned_fake_image, true); // Discriminator on fake data
	Vector<double> fake_target(1);
	fake_target[0] = 0.1; // Label smoothing: use 0.1 instead of 0
	double fake_disc_loss = disc_net.Backward(fake_target);
	disc_cost_av.Add(fake_disc_loss);

	// Update discriminator
	disc.GetTrainer().TrainImplem();

	// Train Generator (update generator to fool discriminator)
	// Generate another batch of conditioned fake images
	for (int i = 0; i < noise_size; i++) {
		gen_input_vol.Set(i, 0, 0, 2.0 * Randomf() - 1.0);
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

	// Create conditioned fake image input for discriminator
	Volume conditioned_fake_image_gen(input_width + num_classes, input_height, input_depth);
	
	// Fill the image part with new fake image
	for (int w = 0; w < input_width; w++) {
		for (int h = 0; h < input_height; h++) {
			for (int d = 0; d < input_depth; d++) {
				conditioned_fake_image_gen.Set(w, h, d, fake_image_gen.Get(w, h, d));
			}
		}
	}
	
	// Add conditioning information (same as generator condition)
	for (int h = 0; h < input_height; h++) {
		for (int d = 0; d < input_depth; d++) {
			for (int c = 0; c < num_classes; c++) {
				int w = input_width + c;  // After the image data
				if (c == condition_label) {
					conditioned_fake_image_gen.Set(w, h, d, 1.0);  // 1 for the condition class
				} else {
					conditioned_fake_image_gen.Set(w, h, d, 0.0);  // 0 for other classes
				}
			}
		}
	}

	// Get discriminator's output on conditioned fake image
	Volume& disc_on_fake = disc_net.Forward(conditioned_fake_image_gen, false); // No gradient tracking on discriminator

	// Update generator to make discriminator think fake images are real
	Vector<double> gen_target(1);
	gen_target[0] = 0.9; // Try to fool discriminator
	double gen_loss = gen_net.Backward(gen_target);
	gen_cost_av.Add(gen_loss);

	// Update generator
	gen.GetTrainer().TrainImplem();
}

void ConditionalGANLayer::SampleInput() {
	// This is now handled in the Train() method
}

void ConditionalGANLayer::SampleOutput() {
	// This is now handled in the Train() method
}

Volume& ConditionalGANLayer::Generate(Volume& noise_input, int condition_label) {
	Net& gen_net = gen.GetNetwork();
	
	// Combine noise and conditioning information
	Volume input_with_condition(noise_size + num_classes, 1, 1);
	
	// Fill with noise input
	for (int i = 0; i < noise_size; i++) {
		input_with_condition.Set(i, 0, 0, noise_input.Get(i, 0, 0));
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