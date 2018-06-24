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
	
	// forward and backward a generated (negative) example
	for (int k = 0; k < 5; k++) {
		SampleInput();
		Volume& xgen = gen_net.Forward(tmp_input, false);
		Volume& sgen = disc_net.Forward(xgen, true);
		tmp_ret[0] = sgen.Get(0) - 0;
		double cost = disc_net.Backward(tmp_ret);
		disc_cost_av.Add(cost);
		
		// forward and backward a real (positive) example
		SampleOutput();
		Volume& sdata = disc_net.Forward(tmp_input, true);
		tmp_ret[0] = sgen.Get(0) - 1;
		cost = disc_net.Backward(tmp_ret);
		disc_cost_av.Add(cost);
		disc.GetTrainer().TrainImplem();
	}
	
	// backward the generator
	SampleInput();
	Volume& xgen = gen_net.Forward(tmp_input, true);
	Volume& sgen = disc_net.Forward(xgen, true);
	
	#if 0
	// Original Simple GAN
	tmp_ret[0] = sgen.Get(0) - 0;
	double disc_loss = disc_net.Backward(tmp_ret);
	const Vector<double>& disc_in_grads = disc.GetInput()->output_activation.GetGradients();
	tmp_ret.SetCount(disc_in_grads.GetCount());
	for(int i = 0; i < disc_in_grads.GetCount(); i++)
		tmp_ret[i] = -disc_in_grads[i]; // negate
	double gen_cost = gen_net.Backward(tmp_ret);
	#elif 0
	// Works better
	tmp_ret[0] = sgen.Get(0) - 1;
	double disc_loss = disc_net.Backward(tmp_ret);
	const Vector<double>& disc_in_grads = disc.GetInput()->output_activation.GetGradients();
	double gen_cost = gen_net.Backward(disc_in_grads);
	#else
	// Combination of both
	{
		tmp_ret[0] = sgen.Get(0) - 0;
		double disc_loss = disc_net.Backward(tmp_ret);
		const Vector<double>& disc_in_grads = disc.GetInput()->output_activation.GetGradients();
		tmp_ret2.SetCount(disc_in_grads.GetCount());
		for(int i = 0; i < disc_in_grads.GetCount(); i++)
			tmp_ret2[i] = -disc_in_grads[i]; // negate
	}
	{
		tmp_ret[0] = sgen.Get(0) - 1;
		double disc_loss = disc_net.Backward(tmp_ret);
		const Vector<double>& disc_in_grads = disc.GetInput()->output_activation.GetGradients();
		for(int i = 0; i < disc_in_grads.GetCount(); i++)
			tmp_ret2[i] = (tmp_ret2[i] + disc_in_grads[i]) * 0.5;
	}
	double gen_cost = gen_net.Backward(tmp_ret2);
	#endif
	
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



