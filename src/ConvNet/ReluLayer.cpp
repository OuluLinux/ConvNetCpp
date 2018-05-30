#include "LayerBase.h"


namespace ConvNet {

void LayerBase::InitRelu(int input_width, int input_height, int input_depth) {
	output_depth = input_depth;
	output_width = input_width;
	output_height = input_height;
}

Volume& LayerBase::ForwardRelu(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input;
	Volume& output = output_activation;
	
	for (int i = 0; i < input.GetLength(); i++) {
		if (output.Get(i) < 0) {
			output.Set(i, 0); // threshold at 0
		}
	}
	
	return output_activation;
}

void LayerBase::BackwardRelu() {
	Volume& input = *input_activation; // we need to set dw of this
	int length = input.GetLength();
	
	input.ZeroGradients(); // zero out gradient wrt data
	
	for (int i = 0; i < length; i++)
	{
		if (output_activation.Get(i) <= 0) {
			input.SetGradient(i, 0); // threshold
		}
		else {
			input.SetGradient(i, output_activation.GetGradient(i));
		}
	}
}

String LayerBase::ToStringRelu() const {
	return Format("Relu: w:%d, h:%d, d:%d",
		output_width, output_height, output_depth);
}

}
