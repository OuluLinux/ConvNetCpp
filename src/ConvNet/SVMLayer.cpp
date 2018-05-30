#include "LayerBase.h"


namespace ConvNet {

void LayerBase::InitSVM(int input_width, int input_height, int input_depth) {
	// computed
	output_depth = input_width * input_height * input_depth;
	output_width = 1;
	output_height = 1;
}

Volume& LayerBase::ForwardSVM(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input; // nothing to do, output raw scores
	return input;
}

double LayerBase::BackwardSVM(int pos, double yd) {
	
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	
	input.ZeroGradients(); // zero out the gradient of input Vol
	
	// we're using structured loss here, which means that the score
	// of the ground truth should be higher than the score of any other
	// class, by a margin
	double yscore = input.Get(pos); // score of ground truth
	const double margin = 1.0;
	double loss = 0.0;
	for (int i = 0; i < output_depth; i++) {
		if (pos == i) {
			continue;
		}
		double ydiff = -1.0 * yscore + input.Get(i) + margin;
		if (ydiff > 0) {
			// violating dimension, apply loss
			input.SetGradient(i, input.GetGradient(i) + 1);
			input.SetGradient(pos, input.GetGradient(pos) - 1);
			loss += ydiff;
		}
	}
	
	return loss;
}

String LayerBase::ToStringSVM() const {
	return Format("SVM: w:%d, h:%d, d:%d, classes:%d",
		output_width, output_height, output_depth, class_count);
}

}
