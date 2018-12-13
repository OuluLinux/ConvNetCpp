#include "LayerBase.h"


namespace ConvNet {


void LayerBase::InitDropOut(int input_width, int input_height, int input_depth) {
	
	// computed
	output_width = input_width;
	output_height = input_height;
	output_depth = input_depth;
	
	dropped.SetCount(0);
	dropped.SetCount(output_width * output_height * output_depth, false);
}

Volume& LayerBase::ForwardDropOut(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input;
	Volume& output = output_activation;
	
	int length = input.GetLength();
	
	if (is_training) {
		// do dropout
		for (int i = 0; i < length; i++) {
			if (Randomf() < drop_prob) {
				output.Set(i, 0);
				dropped[i] = true;
			} // drop!
			else {
				dropped[i] = false;
			}
		}
	}
	else {
		// scale the activations during prediction
		for (int i = 0; i < length; i++) {
			output.Set(i, output.Get(i) * (1 - drop_prob));
		}
	}
	
	return output_activation; // dummy identity function for now
}

void LayerBase::BackwardDropOut() {
	Volume& input = *input_activation; // we need to set dw of this
	Volume& output = output_activation;
	
	int length = input.GetLength();
	input.ZeroGradients(); // zero out gradient wrt data
	
	for (int i = 0; i < length; i++) {
		if (!dropped[i]) {
			input.SetGradient(i, output.GetGradient(i)); // copy over the gradient
		}
	}
}

String LayerBase::ToStringDropOut() const {
	return Format("Dropout: w:%d, h:%d, d:%d, probability:%2!,n",
		output_width, output_height, output_depth, drop_prob);
}

}
