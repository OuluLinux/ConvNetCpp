#include "LayerBase.h"


namespace ConvNet {

void LayerBase::InitSoftmax(int input_width, int input_height, int input_depth) {
	int input_count = input_width * input_height * input_depth;
	
	output_depth = input_count;
	output_width = 1;
	output_height = 1;
}

Volume& LayerBase::ForwardSoftmax(Volume& input, bool is_training) {
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

double LayerBase::BackwardSoftmax(int pos, double y) {
	
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	input.ZeroGradients(); // zero out the gradient of input Vol
	
	for (int i = 0; i < output_depth; i++) {
		double indicator = i == pos ? 1.0 : 0.0;
		double mul = -1.0 * (indicator - es[i]);
		input.SetGradient(i, mul);
	}
	
	// loss is the class negative log likelihood
	return -1.0 * log(es[pos]);
}

String LayerBase::ToStringSoftmax() const {
	return Format("Softmax: w:%d, h:%d, d:%d classes:%d",
		output_width, output_height, output_depth, class_count);
}

}
