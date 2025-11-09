#include "GAN.h"



GANLayer::GANLayer() {
	
}

void GANLayer::Init(int stride) {
	this->stride = stride;
	
	input_width  = 28;
	input_height = 28;
	input_depth  = 1;
	
	
	#if 0
	String disc_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + IntStr(input_width) + ", \"input_height\":" + IntStr(input_height) + ", \"input_depth\":" + IntStr(input_depth) + "},\n"
					"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":1, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"dropout\", \"drop_prob\":0.3},\n"
					"\t{\"type\":\"pool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":1, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"dropout\", \"drop_prob\":0.3},\n"
					"\t{\"type\":\"pool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":1, \"stride\":1, \"pad\":2},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
					"\t{\"type\":\"adadelta\", \"learning_rate\":0.001, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";
	#else
	String disc_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + IntStr(input_width) + ", \"input_height\":" + IntStr(input_height) + ", \"input_depth\":" + IntStr(input_depth) + "},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
					"\t{\"type\":\"adadelta\", \"learning_rate\":0.001, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";
	#endif
	
	if (!disc.MakeLayers(disc_t))
		throw Exc("Discriminator network loading failed");
	
	
	
	#if 0
	String gen_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + IntStr(input_width) + ", \"input_height\":" + IntStr(input_height) + ", \"input_depth\":" + IntStr(input_depth) + "},\n"
					"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":4, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"pool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"conv\", \"width\":5, \"height\":5, \"filter_count\":16, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"pool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"unpool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"deconv\", \"width\":5, \"height\":5, \"filter_count\":8, \"stride\":1, \"pad\":2, \"activation\":\"relu\"},\n"
					"\t{\"type\":\"unpool\", \"width\":2, \"height\":2, \"stride\":2},\n"
					"\t{\"type\":\"deconv\", \"width\":5, \"height\":5, \"filter_count\":1, \"stride\":1, \"pad\":2, \"activation\":\"tanh\"},\n"
					"\t{\"type\":\"adadelta\", \"learning_rate\":0.001, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";
	#else
	String gen_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + IntStr(input_width) + ", \"input_height\":" + IntStr(input_height) + ", \"input_depth\":" + IntStr(input_depth) + "},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":50, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":784},\n"
					"\t{\"type\":\"adadelta\", \"learning_rate\":0.001, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";
	#endif
	
	if (!gen.MakeLayers(gen_t))
		throw Exc("Generator network loading failed");
	
	
	
}

void GANLayer::Train() {
	tmp_ret.SetCount(1);

	Net& gen_net = gen.GetNetwork();
	Net& disc_net = disc.GetNetwork();

	// Train Discriminator based on selected loss type
	for (int k = 0; k < 1; k++) {  // Reduced from 5 to 1 for initial stability
		// Train discriminator on real data
		SampleOutput(); // Sample real data
		Volume& real_output = disc_net.Forward(tmp_input, true);
		
		switch(loss_type) {
			case GANLossType::BINARY_CROSS_ENTROPY:
			{
				// Standard GAN: maximize log(D(x)) + log(1 - D(G(z)))
				// For real data, we want discriminator to output 1 (real)
				tmp_ret[0] = real_output.Get(0) - 1; // Discriminator loss for real data: want output = 1
				double real_disc_cost = disc_net.Backward(tmp_ret);
				disc_cost_av.Add(real_disc_cost);
				
				// Train discriminator on fake data (label = 0)
				SampleInput(); // Sample random input for generator
				Volume& fake_input = gen_net.Forward(tmp_input, false); // Generate fake data with no gradient tracking
				Volume& fake_output = disc_net.Forward(fake_input, true); // Discriminator on fake data
				// For fake data, we want discriminator to output 0 (fake)
				tmp_ret[0] = fake_output.Get(0) - 0; // Discriminator loss for fake data: want output = 0
				double fake_disc_cost = disc_net.Backward(tmp_ret);
				disc_cost_av.Add(fake_disc_cost);
				break;
			}
			case GANLossType::LEAST_SQUARES:
			{
				// LSGAN: use least squares loss instead of log loss
				// For real data, use label 1, for fake data use label 0
				// L = 0.5 * (D(x) - 1)^2 + 0.5 * (D(G(z)))^2
				// Gradient for real: D(x) - 1
				// Gradient for fake: D(G(z)) - 0
				tmp_ret[0] = real_output.Get(0) - 1; // Gradient for real data with LSGAN loss
				double real_disc_cost = disc_net.Backward(tmp_ret);
				disc_cost_av.Add(real_disc_cost);
				
				// Train discriminator on fake data
				SampleInput(); // Sample random input for generator
				Volume& fake_input = gen_net.Forward(tmp_input, false); // Generate fake data with no gradient tracking
				Volume& fake_output = disc_net.Forward(fake_input, true); // Discriminator on fake data
				tmp_ret[0] = fake_output.Get(0) - 0; // Gradient for fake data with LSGAN loss
				double fake_disc_cost = disc_net.Backward(tmp_ret);
				disc_cost_av.Add(fake_disc_cost);
				break;
			}
			case GANLossType::WASSERSTEIN:
			{
				// WGAN: use Wasserstein loss (distance between real and fake distributions)
				// For real data, we want discriminator to output high values (maximize)
				// For fake data, we want discriminator to output low values (minimize)
				// Gradient for real: -1 (we want to increase discriminator output on real)
				// Gradient for fake: +1 (we want to decrease discriminator output on fake)
				tmp_ret[0] = -1; // WGAN gradient for real data (we want to increase disc output)
				double real_disc_cost = disc_net.Backward(tmp_ret);
				disc_cost_av.Add(-real_output.Get(0)); // Use negative of discriminator output as loss for WGAN
				
				// Train discriminator on fake data
				SampleInput(); // Sample random input for generator
				Volume& fake_input = gen_net.Forward(tmp_input, false); // Generate fake data with no gradient tracking
				Volume& fake_output = disc_net.Forward(fake_input, true); // Discriminator on fake data
				tmp_ret[0] = 1; // WGAN gradient for fake data (we want to decrease disc output)
				double fake_disc_cost = disc_net.Backward(tmp_ret);
				disc_cost_av.Add(fake_output.Get(0)); // Use discriminator output as loss for fake data
				break;
			}
		}

		// Update discriminator weights
		disc.GetTrainer().TrainImplem();
	}

	// Train Generator based on selected loss type
	SampleInput(); // Sample random input for generator
	Volume& gen_input = gen_net.Forward(tmp_input, true); // Generate fake data with gradient tracking
	Volume& gen_output = disc_net.Forward(gen_input, false); // Discriminator on generated data

	switch(loss_type) {
		case GANLossType::BINARY_CROSS_ENTROPY:
		case GANLossType::LEAST_SQUARES:
		{
			// For generator: we want the discriminator to think generated data is real (output ~1)
			// For BCE and LSGAN, we want discriminator to output 1
			tmp_ret[0] = gen_output.Get(0) - 1; // Generator wants discriminator to output 1 for generated data
			break;
		}
		case GANLossType::WASSERSTEIN:
		{
			// For WGAN generator: we want to maximize discriminator output on fake data
			// So gradient should be -1 to minimize the negative of discriminator output
			tmp_ret[0] = -1; // WGAN gradient for generator (trying to maximize disc output)
			break;
		}
	}

	double gen_disc_loss = disc_net.Backward(tmp_ret);

	// Get gradients from discriminator input layer to pass to generator
	const Vector<double>& disc_input_grads = disc.GetInput()->output_activation.GetGradients();
	tmp_ret.SetCount(disc_input_grads.GetCount());
	for(int i = 0; i < disc_input_grads.GetCount(); i++)
		tmp_ret[i] = -disc_input_grads[i]; // Negate gradients for generator training

	// Backpropagate through generator
	double gen_cost = gen_net.Backward(tmp_ret);
	gen.GetTrainer().TrainImplem();
	gen_cost_av.Add(gen_cost);
}

void GANLayer::SampleInput() {
	tmp_input.Init(input_width, input_height, input_depth);
}

void GANLayer::SampleOutput() {
	if (label == -1) label = Random(10);
	SessionData& d = disc.Data();
	while (true) {
		int i = data_iter;
		data_iter++;
		if (data_iter >= d.GetDataCount()) data_iter = 0;
		
		int l = d.GetLabel(i);
		if (l == label) {
			const Vector<double>& data = d.Get(i);
			tmp_input.Set(input_width, input_height, input_depth, data);
			break;
		}
	}
}

Volume& GANLayer::Generate(Volume& input) {
	Net& gen_net = gen.GetNetwork();
	Volume& xgen = gen_net.Forward(input, false);
	return xgen;
}



