#include "CycleGAN.h"

CycleGANLayer::CycleGANLayer() {
}

void CycleGANLayer::Init(int stride, const CycleGANParams& params) {
	this->stride = stride;

	// Set the dimensions for input images (this can be adjusted for different datasets)
	input_width  = 64;  // Example size, can be adjusted
	input_height = 64;
	input_depth  = 3;   // RGB images

	// Size for the image input (e.g., 64x64x3 = 12288)
	int img_size = input_width * input_height * input_depth;

	// Discriminator for domain X (e.g., horses)
	String disc_X_t =	"[\n"
						"\t{\"type\":\"input\", \"input_width\":" + FormatInt(input_width) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth) + "},\n"
						"\t{\"type\":\"conv\", \"filter_width\":4, \"filter_height\":4, \"filter_count\":64, \"stride\":2, \"pad\":1, \"activation\":\"leakyrelu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":4, \"filter_height\":4, \"filter_count\":128, \"stride\":2, \"pad\":1, \"activation\":\"leakyrelu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":4, \"filter_height\":4, \"filter_count\":256, \"stride\":2, \"pad\":1, \"activation\":\"leakyrelu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":4, \"filter_height\":4, \"filter_count\":512, \"stride\":2, \"pad\":1, \"activation\":\"leakyrelu\"},\n"
						"\t{\"type\":\"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
						"\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":1, \"l2_decay\":0.0001}\n"
						"]\n";

	if (!disc_X.MakeLayers(disc_X_t))
		throw Exc("Discriminator X network loading failed");

	// Discriminator for domain Y (e.g., zebras)
	String disc_Y_t =	"[\n"
						"\t{\"type\":\"input\", \"input_width\":" + FormatInt(input_width) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth) + "},\n"
						"\t{\"type\":\"conv\", \"filter_width\":4, \"filter_height\":4, \"filter_count\":64, \"stride\":2, \"pad\":1, \"activation\":\"leakyrelu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":4, \"filter_height\":4, \"filter_count\":128, \"stride\":2, \"pad\":1, \"activation\":\"leakyrelu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":4, \"filter_height\":4, \"filter_count\":256, \"stride\":2, \"pad\":1, \"activation\":\"leakyrelu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":4, \"filter_height\":4, \"filter_count\":512, \"stride\":2, \"pad\":1, \"activation\":\"leakyrelu\"},\n"
						"\t{\"type\":\"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
						"\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":1, \"l2_decay\":0.0001}\n"
						"]\n";

	if (!disc_Y.MakeLayers(disc_Y_t))
		throw Exc("Discriminator Y network loading failed");

	// Generator X to Y (e.g., horse to zebra)
	String gen_XtoY_t =	"[\n"
						"\t{\"type\":\"input\", \"input_width\":" + FormatInt(input_width) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth) + "},\n"
						"\t{\"type\":\"conv\", \"filter_width\":7, \"filter_height\":7, \"filter_count\":32, \"stride\":1, \"pad\":3, \"activation\":\"relu\"},\n"  // Input layer
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":64, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"  // Downsample
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n" // Downsample
						// Residual blocks
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1},\n"
						"\t{\"type\":\"add\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1},\n"
						"\t{\"type\":\"add\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1},\n"
						"\t{\"type\":\"add\"},\n"
						// Upsample
						"\t{\"type\":\"tconv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":64, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"tconv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":32, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":7, \"filter_height\":7, \"filter_count\":3, \"stride\":1, \"pad\":3, \"activation\":\"tanh\"},\n"  // Output layer
						"\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":1, \"l2_decay\":0.0001}\n"
						"]\n";

	if (!gen_XtoY.MakeLayers(gen_XtoY_t))
		throw Exc("Generator X to Y network loading failed");

	// Generator Y to X (e.g., zebra to horse)
	String gen_YtoX_t =	"[\n"
						"\t{\"type\":\"input\", \"input_width\":" + FormatInt(input_width) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth) + "},\n"
						"\t{\"type\":\"conv\", \"filter_width\":7, \"filter_height\":7, \"filter_count\":32, \"stride\":1, \"pad\":3, \"activation\":\"relu\"},\n"  // Input layer
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":64, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"  // Downsample
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n" // Downsample
						// Residual blocks
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1},\n"
						"\t{\"type\":\"add\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1},\n"
						"\t{\"type\":\"add\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1},\n"
						"\t{\"type\":\"add\"},\n"
						// Upsample
						"\t{\"type\":\"tconv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":64, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"tconv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":32, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":7, \"filter_height\":7, \"filter_count\":3, \"stride\":1, \"pad\":3, \"activation\":\"tanh\"},\n"  // Output layer
						"\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":1, \"l2_decay\":0.0001}\n"
						"]\n";

	if (!gen_YtoX.MakeLayers(gen_YtoX_t))
		throw Exc("Generator Y to X network loading failed");
}

void CycleGANLayer::Train() {
	Net& gen_XtoY_net = gen_XtoY.GetNetwork();
	Net& gen_YtoX_net = gen_YtoX.GetNetwork();
	Net& disc_X_net = disc_X.GetNetwork();
	Net& disc_Y_net = disc_Y.GetNetwork();

	// Sample real images from domains X and Y
	SessionData& data_X = disc_X.Data();
	SessionData& data_Y = disc_Y.Data();
	
	// Get real images (assuming they are RGB images of format width x height x 3)
	int real_X_idx = Random(data_X.GetDataCount());
	int real_Y_idx = Random(data_Y.GetDataCount());
	
	const Vector<double>& real_X_vec = clone(data_X.Get(real_X_idx));
	const Vector<double>& real_Y_vec = clone(data_Y.Get(real_Y_idx));
	
	Volume real_X_image(input_width, input_height, input_depth);
	Volume real_Y_image(input_width, input_height, input_depth);
	
	// Fill the volume with data - assume the vector has been flattened from width*height*depth values
	for (int i = 0; i < input_width * input_height * input_depth; i++) {
		real_X_image.Set(i % input_width, (i / input_width) % input_height, i / (input_width * input_height), real_X_vec[i]);
		real_Y_image.Set(i % input_width, (i / input_width) % input_height, i / (input_width * input_height), real_Y_vec[i]);
	}

	// =============== Train Discriminators ===============
	// Discriminator X: should distinguish real X images from fake X images (generated from Y)
	
	// Discriminator X on real X data (should output 1)
	Volume& real_X_output = disc_X_net.Forward(real_X_image, true);
	Vector<double> real_X_target(1);
	real_X_target[0] = 0.9; // Label smoothing: use 0.9 instead of 1
	double real_disc_X_loss = disc_X_net.Backward(real_X_target);
	disc_X_cost_av.Add(real_disc_X_loss);

	// Generate fake X from Y (Y -> X)
	Volume& fake_X_from_Y = gen_YtoX_net.Forward(real_Y_image, false); // No gradient tracking for generator during discriminator training
	
	// Discriminator X on fake X data (should output 0)
	Volume& fake_X_output = disc_X_net.Forward(fake_X_from_Y, true);
	Vector<double> fake_X_target(1);
	fake_X_target[0] = 0.1; // Label smoothing: use 0.1 instead of 0
	double fake_disc_X_loss = disc_X_net.Backward(fake_X_target);
	disc_X_cost_av.Add(fake_disc_X_loss);

	// Update discriminator X
	disc_X.GetTrainer().TrainImplem();

	// Discriminator Y: should distinguish real Y images from fake Y images (generated from X)
	
	// Discriminator Y on real Y data (should output 1)
	Volume& real_Y_output = disc_Y_net.Forward(real_Y_image, true);
	Vector<double> real_Y_target(1);
	real_Y_target[0] = 0.9; // Label smoothing: use 0.9 instead of 1
	double real_disc_Y_loss = disc_Y_net.Backward(real_Y_target);
	disc_Y_cost_av.Add(real_disc_Y_loss);

	// Generate fake Y from X (X -> Y)
	Volume& fake_Y_from_X = gen_XtoY_net.Forward(real_X_image, false); // No gradient tracking for generator during discriminator training
	
	// Discriminator Y on fake Y data (should output 0)
	Volume& fake_Y_output = disc_Y_net.Forward(fake_Y_from_X, true);
	Vector<double> fake_Y_target(1);
	fake_Y_target[0] = 0.1; // Label smoothing: use 0.1 instead of 0
	double fake_disc_Y_loss = disc_Y_net.Backward(fake_Y_target);
	disc_Y_cost_av.Add(fake_disc_Y_loss);

	// Update discriminator Y
	disc_Y.GetTrainer().TrainImplem();

	// =============== Train Generators (with cycle consistency) ===============
	// Train generators to fool discriminators and maintain cycle consistency
	
	// Generate fake images (with gradient tracking for generators)
	Volume& fake_Y = gen_XtoY_net.Forward(real_X_image, true);  // X -> Y
	Volume& fake_X = gen_YtoX_net.Forward(real_Y_image, true);  // Y -> X

	// Try to fool discriminators
	// Generator XtoY should fool Discriminator Y
	Volume& disc_Y_on_fake_Y = disc_Y_net.Forward(fake_Y, false);  // No gradient tracking on discriminator
	Vector<double> gen_XtoY_target(1);
	gen_XtoY_target[0] = 0.9; // Try to fool discriminator
	double gen_XtoY_adv_loss = gen_XtoY_net.Backward(gen_XtoY_target);
	gen_XtoY_cost_av.Add(gen_XtoY_adv_loss);

	// Generator YtoX should fool Discriminator X
	Volume& disc_X_on_fake_X = disc_X_net.Forward(fake_X, false);  // No gradient tracking on discriminator
	Vector<double> gen_YtoX_target(1);
	gen_YtoX_target[0] = 0.9; // Try to fool discriminator
	double gen_YtoX_adv_loss = gen_YtoX_net.Backward(gen_YtoX_target);
	gen_YtoX_cost_av.Add(gen_YtoX_adv_loss);

	// Calculate cycle consistency losses
	// Forward cycle: X -> Y -> X
	Volume& reconstructed_X = gen_YtoX_net.Forward(fake_Y, true);  // Y -> X (using gradients for generator update)
	Vector<double> cycle_X_target(input_width * input_height * input_depth);
	// Compute L1 distance between original and reconstructed
	for (int i = 0; i < input_width * input_height * input_depth; i++) {
		double orig_val = real_X_image.Get(i % input_width, (i / input_width) % input_height, i / (input_width * input_height));
		double recon_val = reconstructed_X.Get(i % input_width, (i / input_width) % input_height, i / (input_width * input_height));
		cycle_X_target[i] = orig_val - recon_val;  // Gradient of L1 loss is sign of difference
	}
	double cycle_X_loss = gen_YtoX_net.Backward(cycle_X_target) * lambda_cycle;
	gen_YtoX_cost_av.Add(cycle_X_loss);

	// Backward cycle: Y -> X -> Y
	Volume& reconstructed_Y = gen_XtoY_net.Forward(fake_X, true);  // X -> Y (using gradients for generator update)
	Vector<double> cycle_Y_target(input_width * input_height * input_depth);
	// Compute L1 distance between original and reconstructed
	for (int i = 0; i < input_width * input_height * input_depth; i++) {
		double orig_val = real_Y_image.Get(i % input_width, (i / input_width) % input_height, i / (input_width * input_height));
		double recon_val = reconstructed_Y.Get(i % input_width, (i / input_width) % input_height, i / (input_width * input_height));
		cycle_Y_target[i] = orig_val - recon_val;  // Gradient of L1 loss is sign of difference
	}
	double cycle_Y_loss = gen_XtoY_net.Backward(cycle_Y_target) * lambda_cycle;
	gen_XtoY_cost_av.Add(cycle_Y_loss);

	// Update generators
	gen_XtoY.GetTrainer().TrainImplem();
	gen_YtoX.GetTrainer().TrainImplem();
}

void CycleGANLayer::SampleInput() {
	// This is now handled in the Train() method
}

void CycleGANLayer::SampleOutput() {
	// This is now handled in the Train() method
}

Volume& CycleGANLayer::GenerateXtoY(Volume& input) {
	Net& gen_XtoY_net = gen_XtoY.GetNetwork();
	Volume& xgen = gen_XtoY_net.Forward(input, false);
	return xgen;
}

Volume& CycleGANLayer::GenerateYtoX(Volume& input) {
	Net& gen_YtoX_net = gen_YtoX.GetNetwork();
	Volume& xgen = gen_YtoX_net.Forward(input, false);
	return xgen;
}