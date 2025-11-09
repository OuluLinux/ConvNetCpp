void GANLayer::Train() {
	tmp_ret.SetCount(1);

	Net& gen_net = gen.GetNetwork();
	Net& disc_net = disc.GetNetwork();

	// Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
	// Train k times on discriminator for every 1 time on generator (common practice)
	for (int k = 0; k < 1; k++) {  // Reduced from 5 to 1 for initial stability
		// Train discriminator on real data (label = 1)
		SampleOutput(); // Sample real data
		Volume& real_output = disc_net.Forward(tmp_input, true);
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

		// Update discriminator weights
		disc.GetTrainer().TrainImplem();
	}

	// Train Generator: maximize log(D(G(z))) - make discriminator think generated data is real
	SampleInput(); // Sample random input for generator
	Volume& gen_input = gen_net.Forward(tmp_input, true); // Generate fake data with gradient tracking
	Volume& gen_output = disc_net.Forward(gen_input, false); // Discriminator on generated data

	// We want the discriminator to think generated data is real (output ~1)
	tmp_ret[0] = gen_output.Get(0) - 1; // Generator wants discriminator to output 1 for generated data
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