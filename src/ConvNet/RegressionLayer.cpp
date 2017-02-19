#include "Layers.h"


namespace ConvNet {

RegressionLayer::RegressionLayer() {
	
}

void RegressionLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	int input_count = input_width * input_height * input_depth;
	output_depth = input_count;
	output_width = 1;
	output_height = 1;
}

Volume& RegressionLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input;
	return input; // identity function
}

void RegressionLayer::Backward() {
	throw NotImplementedException();
}

double RegressionLayer::Backward(const Vector<double>& y) {
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	
	input.ZeroGradients(); // zero out the gradient of input Vol
	double loss = 0.0;
	
	for (int i = 0; i < output_depth; i++) {
		double dy = input.Get(i) - y[i];
		input.SetGradient(i,  dy);
		loss += 0.5 * dy * dy;
	}
	
	return loss;
}

double RegressionLayer::Backward(double y) {
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	input.ZeroGradients(); // zero out the gradient of input Vol
	double loss = 0.0;
	
	// lets hope that only one number is being regressed
	double dy = input.Get(0) - y;
	input.SetGradient(0, dy);
	loss += 0.5 * dy * dy;
	
	return loss;
}

void Backward() {
	throw NotImplementedException();
}

}
