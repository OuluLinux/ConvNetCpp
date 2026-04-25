#include "DiffusionModel.h"

DiffusionModelLayer::DiffusionModelLayer() {
}

void DiffusionModelLayer::Init(int stride, int timesteps, int img_width, int img_height, int img_depth) {
	this->stride = stride;
	this->timesteps = timesteps;
	this->input_width = img_width;
	this->input_height = img_height;
	this->input_depth = img_depth;

	// Set up the noise schedule (betas, alphas, alpha_bars)
	// Linear beta schedule as used in the original DDPM paper
	betas.SetCount(timesteps);
	alphas.SetCount(timesteps);
	alpha_bars.SetCount(timesteps);

	// Linear schedule from beta_start=0.0001 to beta_end=0.02
	double beta_start = 0.0001;
	double beta_end = 0.02;
	for (int i = 0; i < timesteps; i++) {
		betas[i] = beta_start + (double)i / (timesteps - 1) * (beta_end - beta_start);
		alphas[i] = 1.0 - betas[i];
		// Calculate cumulative alpha_bar (alpha_t_bar = prod_{s=1}^t (1 - beta_s))
		if (i == 0) {
			alpha_bars[i] = alphas[i];
		} else {
			alpha_bars[i] = alpha_bars[i-1] * alphas[i];
		}
	}

	// Diffusion model architecture - U-Net is common for diffusion models
	// This implementation uses a U-Net-like architecture to predict noise
	String model_t =	"[\n"
						"\t{\"type\":\"input\", \"input_width\":" + FormatInt(input_width) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth + 1) + "},\n"  // +1 for timestep embedding
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"downsample\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":256, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":256, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":256, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"downsample\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":512, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":512, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":512, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"downsample\"},\n"
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":512, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"tconv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":512, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"  // Upsample
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":256, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"tconv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":256, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"  // Upsample
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":1, \"pad\":1, \"activation\":\"relu\"},\n"
						"\t{\"type\":\"tconv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":128, \"stride\":2, \"pad\":1, \"activation\":\"relu\"},\n"  // Upsample
						"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":" + FormatInt(input_depth) + ", \"stride\":1, \"pad\":1, \"activation\":\"tanh\"},\n"  // Output same depth as input
						"\t{\"type\":\"adam\", \"learning_rate\":0.0002, \"beta1\":0.5, \"batch_size\":16, \"l2_decay\":0.0001}\n"
						"]\n";

	if (!model.MakeLayers(model_t))
		throw Exc("Diffusion model network loading failed");
}

void DiffusionModelLayer::Train() {
	Net& net = model.GetNetwork();

	// Sample real image
	SessionData& data = model.Data();
	int real_idx = Random(data.GetDataCount());
	const Vector<double>& real_image_vec = clone(data.Get(real_idx)); // Use reference to avoid copy

	// Convert to Volume
	Volume real_image(input_width, input_height, input_depth);
	for (int w = 0; w < input_width; w++) {
		for (int h = 0; h < input_height; h++) {
			for (int d = 0; d < input_depth; d++) {
				real_image.Set(w, h, d, real_image_vec[w*input_height*input_depth + h*input_depth + d]);
			}
		}
	}

	// Sample random timestep t
	int t = Random(timesteps);

	// Calculate the noise to add based on the schedule
	double sqrt_alpha_bar_t = sqrt(alpha_bars[t]);
	double sqrt_one_minus_alpha_bar_t = sqrt(1.0 - alpha_bars[t]);

	// Generate random noise
	Volume noise(input_width, input_height, input_depth);
	for (int i = 0; i < input_width * input_height * input_depth; i++) {
		noise.Set(i % input_width, (i / input_width) % input_height, i / (input_width * input_height), Randomf() * 2.0 - 1.0); // Random values between -1 and 1
	}

	// Add noise to the image according to the forward process
	Volume noised_image(input_width, input_height, input_depth);
	for (int w = 0; w < input_width; w++) {
		for (int h = 0; h < input_height; h++) {
			for (int d = 0; d < input_depth; d++) {
				double original_pixel = clone(real_image.Get(w, h, d));
				double noise_pixel = clone(noise.Get(w, h, d));
				double noised_pixel = sqrt_alpha_bar_t * original_pixel + sqrt_one_minus_alpha_bar_t * noise_pixel;
				// Clamp to [-1, 1] range
				noised_pixel = max(-1.0, min(1.0, noised_pixel));
				noised_image.Set(w, h, d, noised_pixel);
			}
		}
	}

	// Create input for the model: concat image and timestep embedding
	// For simplicity, we'll add a channel that represents the timestep
	Volume input_with_timestep(input_width, input_height, input_depth + 1);
	for (int w = 0; w < input_width; w++) {
		for (int h = 0; h < input_height; h++) {
			for (int d = 0; d < input_depth; d++) {
				input_with_timestep.Set(w, h, d, noised_image.Get(w, h, d));
			}
			// Add normalized timestep as an additional channel
			input_with_timestep.Set(w, h, input_depth, (double)t / timesteps);
		}
	}

	// Forward pass through the model
	Volume& predicted_noise = net.Forward(input_with_timestep, true); // Enable gradient tracking

	// Calculate loss: MSE between predicted noise and actual noise
	Vector<double> target_noise(input_width * input_height * input_depth);
	for (int i = 0; i < input_width * input_height * input_depth; i++) {
		double actual_noise = noise.Get(i % input_width, (i / input_width) % input_height, i / (input_width * input_height));
		double predicted = predicted_noise.Get(i % input_width, (i / input_width) % input_height, i / (input_width * input_height));
		// Loss is difference between predicted and actual noise
		target_noise[i] = actual_noise - predicted;
	}

	// Backward pass
	double loss = net.Backward(target_noise);
	loss_av.Add(loss);

	// Update model
	model.GetTrainer().TrainImplem();
}

void DiffusionModelLayer::SampleInput() {
	// This is now handled in the Train() method
}

void DiffusionModelLayer::SampleOutput() {
	// This is now handled in the Train() method
}

Volume& DiffusionModelLayer::Generate(int condition_label) {
	Net& net = model.GetNetwork();

	// Start with random noise
	Volume x(input_width, input_height, input_depth);
	for (int i = 0; i < input_width * input_height * input_depth; i++) {
		x.Set(i % input_width, (i / input_width) % input_height, i / (input_width * input_height), Randomf() * 2.0 - 1.0); // Random values between -1 and 1
	}

	// Denoise iteratively from t=T to t=0
	for (int t = timesteps-1; t >= 0; t--) {
		x = DenoiseStep(x, t, condition_label);
	}

	// Return the final denoised image
	return x;
}

Volume& DiffusionModelLayer::DenoiseStep(Volume& x, int t, int condition_label) {
	Net& net = model.GetNetwork();

	// Create input for the model: concat image and timestep embedding
	Volume input_with_timestep(input_width, input_height, input_depth + 1);
	for (int w = 0; w < input_width; w++) {
		for (int h = 0; h < input_height; h++) {
			for (int d = 0; d < input_depth; d++) {
				input_with_timestep.Set(w, h, d, x.Get(w, h, d));
			}
			// Add normalized timestep as an additional channel
			input_with_timestep.Set(w, h, input_depth, (double)t / timesteps);
		}
	}

	// Predict noise
	Volume& predicted_noise = net.Forward(input_with_timestep, false); // No gradient tracking during sampling

	// Get parameters for this timestep
	double alpha_t = alphas[t];
	double alpha_bar_t = alpha_bars[t];
	double beta_t = betas[t];

	// Calculate coefficients for the reverse process
	double coeff1 = 1.0 / sqrt(alpha_t);
	double coeff2 = (1.0 - alpha_t) / sqrt(1.0 - alpha_bar_t);

	// Reverse process: denoise step
	Volume result(input_width, input_height, input_depth);
	for (int w = 0; w < input_width; w++) {
		for (int h = 0; h < input_height; h++) {
			for (int d = 0; d < input_depth; d++) {
				double current_pixel = clone(x.Get(w, h, d));
				double predicted_noise_pixel = clone(predicted_noise.Get(w, h, d));
				
				// Apply reverse process formula
				double mean = coeff1 * (current_pixel - coeff2 * predicted_noise_pixel);
				
				// Add variance if not at first step
				double variance = 0.0;
				if (t > 0) {
					// Add small random noise
					variance = sqrt(betas[t]) * (Randomf() * 2.0 - 1.0);
				}
				
				double new_pixel = mean + variance;
				// Clamp to [-1, 1] range
				new_pixel = max(-1.0, min(1.0, new_pixel));
				result.Set(w, h, d, new_pixel);
			}
		}
	}

	// Update x with the result
	for (int w = 0; w < input_width; w++) {
		for (int h = 0; h < input_height; h++) {
			for (int d = 0; d < input_depth; d++) {
				x.Set(w, h, d, result.Get(w, h, d));
			}
		}
	}

	return x;
}