#include "Layers.h"


namespace ConvNet {

SvmLayer::SvmLayer(int class_count) {
	this->class_count = class_count;
}

void SvmLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	// computed
	output_depth = input_width * input_height * input_depth;
	output_width = 1;
	output_height = 1;
}

Volume& SvmLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input; // nothing to do, output raw scores
	return input;
}

double SvmLayer::Backward(double yd) {
	int y = (int)yd;
	
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	
	input.ZeroGradients(); // zero out the gradient of input Vol
	
	// we're using structured loss here, which means that the score
	// of the ground truth should be higher than the score of any other
	// class, by a margin
	double yscore = input.Get(y); // score of ground truth
	const double margin = 1.0;
	double loss = 0.0;
	for (int i = 0; i < output_depth; i++) {
		if (y == i) {
			continue;
		}
		double ydiff = -1.0 * yscore + input.Get(i) + margin;
		if (ydiff > 0) {
			// violating dimension, apply loss
			input.SetGradient(i, input.GetGradient(i) + 1);
			input.SetGradient(y, input.GetGradient(y) - 1);
			loss += ydiff;
		}
	}
	
	return loss;
}

double SvmLayer::Backward(const Vector<double>& y) {
	throw NotImplementedException();
}

void SvmLayer::Backward() {
	throw NotImplementedException();
}

}
