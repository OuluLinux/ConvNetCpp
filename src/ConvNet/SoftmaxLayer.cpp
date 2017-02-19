#include "Layers.h"


namespace ConvNet {

SoftmaxLayer::SoftmaxLayer(int class_count) {
	if (class_count < 1)
		throw ArgumentException("SoftmaxLayer class_count must be a positive integer");
	this->class_count = class_count;
}

void SoftmaxLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	int input_count = input_width * input_height * input_depth;
	
	output_depth = input_count;
	output_width = 1;
	output_height = 1;
}

Volume& SoftmaxLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	
	output_activation.Init(1, 1, output_depth, 0.0);
	
	// compute max activation
	double amax = input.Get(0);
	for (int i = 1; i < output_depth; i++) {
		if (input.Get(i) > amax) {
			amax = input.Get(i);
		}
	}
	
	// compute exponentials (carefully to not blow up)
	es.SetCount(output_depth, 0.0);
	
	double esum = 0.0;
	
	for (int i = 0; i < output_depth; i++) {
		double e = exp(input.Get(i) - amax);
		esum += e;
		es[i] = e;
	}
	
	// normalize and output to sum to one
	for (int i = 0; i < output_depth; i++) {
		es[i] /= esum;
		output_activation.Set(i, es[i]);
	}
	
	return output_activation;
}

double SoftmaxLayer::Backward(double y) {
	int yint = (int)y;
	
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	input.ZeroGradients(); // zero out the gradient of input Vol
	
	for (int i = 0; i < output_depth; i++) {
		double indicator = i == yint ? 1.0 : 0.0;
		double mul = -1.0 * (indicator - es[i]);
		input.SetGradient(i, mul);
	}
	
	// loss is the class negative log likelihood
	return -1.0 * log(es[yint]);
}

double SoftmaxLayer::Backward(const Vector<double>& y) {
	throw NotImplementedException();
}

void SoftmaxLayer::Backward() {
	throw NotImplementedException();
}

}
