#include "SimpleGAN.h"

#ifdef flagUSE_CONVNET


SimpleGAN::SimpleGAN()
{
	pdf = pdf1;
	sample = sample1;
	
	input_width  = 1;
	input_height = 1;
	input_depth  = 1;
	
	String disc_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + IntStr(input_width) + ", \"input_height\":" + IntStr(input_height) + ", \"input_depth\":" + IntStr(input_depth) + "},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":30, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":30, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
					"\t{\"type\":\"adadelta\", \"learning_rate\":0.0001, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";
	
	if (!disc.MakeLayers(disc_t))
		throw Exc("Discriminator network loading failed");
	
	
	String gen_t =	"[\n"
					"\t{\"type\":\"input\", \"input_width\":" + IntStr(input_width) + ", \"input_height\":" + IntStr(input_height) + ", \"input_depth\":" + IntStr(input_depth) + "},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":30, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":30, \"activation\": \"tanh\"},\n"
					"\t{\"type\":\"fc\", \"neuron_count\":1},\n"
					"\t{\"type\":\"adadelta\", \"learning_rate\":0.0001, \"batch_size\":50, \"l1_decay\":0.001, \"l2_decay\":0.001}\n"
					"]\n";
	
	if (!gen.MakeLayers(gen_t))
		throw Exc("Generator network loading failed");
	
	
	Title("Simple GAN");
	SetRect(0, 0, 400, 360);
	
	Thread::Start(THISBACK(Trainer));
}

void SimpleGAN::Trainer() {
	stopped = false;
	running = true;
	
	TimeStop ts;
	while (!Thread::IsShutdownThreads() && running) {
		
		Step();
		
		if (ts.Elapsed() >= 1000/60) {
			PostCallback(THISBACK(Refresh0));
			ts.Reset();
		}
	}
	
	stopped = true;
}

// do a single training step
void SimpleGAN::Step() {
	
	Net& gen_net = gen.GetNetwork();
	Net& disc_net = disc.GetNetwork();
	tmp_ret.SetCount(1);
	
	// forward and backward a generated (negative) example
	for (int k = 0; k < 5; k++) {
		tmp_input.Init(input_width, input_height, input_depth, 0);
		tmp_input.Set(0, Randomf());
		Volume& xgen = gen_net.Forward(tmp_input, false);
		Volume& sgen = disc_net.Forward(xgen, true);
		tmp_ret[0] = sgen.Get(0) - 0;
		double cost = disc_net.Backward(tmp_ret);
		
		
		// forward and backward a real (positive) example
		tmp_input.Set(0, sample());
		Volume& sdata = disc_net.Forward(tmp_input, true);
		tmp_ret[0] = sgen.Get(0) - 1;
		cost = disc_net.Backward(tmp_ret);
		disc.GetTrainer().TrainImplem();
	}
	
	// backward the generator
	tmp_input.Init(input_width, input_height, input_depth, 0);
	tmp_input.Set(0, Randomf());
	Volume& xgen = gen_net.Forward(tmp_input, true);
	Volume& sgen = disc_net.Forward(xgen, true);
	tmp_ret[0] = sgen.Get(0) - 0;
	double disc_loss = disc_net.Backward(tmp_ret);
	const Vector<double>& disc_in_grads = disc.GetInput()->output_activation.GetGradients();
	tmp_ret.SetCount(disc_in_grads.GetCount());
	for(int i = 0; i < disc_in_grads.GetCount(); i++)
		tmp_ret[i] = -disc_in_grads[i]; // negate
	double gen_cost = gen_net.Backward(tmp_ret);
	gen.GetTrainer().TrainImplem();
	
	
	
}

void SimpleGAN::Paint(Draw& w) {
	Net& gen_net = gen.GetNetwork();
	Net& disc_net = disc.GetNetwork();
	
	Size sz = GetSize();
	ImageDraw id(sz);
	
	id.DrawRect(sz, White);
	
	id.DrawLine(orix0, orih, orix1, orih, 2, Black);
	id.DrawLine(orix0, transh, orix1, transh, 2, Black);
	
	// draw the true distribution
	double t = 0;
	double prevp = 0;
	while (t <= 1) {
		double p = pdf(t);
		if (t > 0) {
			id.DrawLine(
				(t - dt)*(orix1 - orix0) + orix0,
				transh - prevp,
				t*(orix1 - orix0) + orix0,
				transh - p,
				2,
				Blue);
		}
		prevp = p;
		t += dt;
	}
	
	// draw the discriminator
	t = 0;
	prevp = 0;
	tmp_input.Init(input_width, input_height, input_depth, 0);
	while (t <= 1) {
		tmp_input.Set(0, t);
		Volume& sdata = disc_net.Forward(tmp_input, true);
		double p = sdata.Get(0) * 50;
		if (t > 0) {
			id.DrawLine(
				(t - dt)*(orix1 - orix0) + orix0,
				transh - prevp,
				t*(orix1 - orix0) + orix0,
				transh - p,
				2,
				Green);
		}
		
		prevp = p;
		t += dt;
	}
	
	
	// draw the generating distribution arrows
	t = 0;
	while (t <= 1) {
		tmp_input.Set(0, t);
		Volume& xgen = gen_net.Forward(tmp_input, true);
		double x = xgen.Get(0);
		
		id.DrawLine(
			t*(orix1-orix0) + orix0,
			orih,
			x*(orix1-orix0) + orix0,
			transh,
			2,
			Red);
		
		t += dt;
	}
	
	w.DrawImage(0,0,id);
}

#endif
