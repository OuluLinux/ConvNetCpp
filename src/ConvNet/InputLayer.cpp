#include "Layers.h"

namespace ConvNet {

InputLayer::InputLayer(int input_width, int input_height, int input_depth) {
	if (!(input_width > 0 && input_height > 0 && input_depth > 0))
		throw ArgumentException("All volume components must be positive integers");
	
	Init(input_width, input_height, input_depth);
	
	output_width = input_width;
	output_height = input_height;
	output_depth = input_depth;
}

Volume& InputLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input;
	return output_activation; // simply identity function for now
}

void InputLayer::Backward() {
	
}

Volume& InputLayer::Forward(bool is_training) {
	return output_activation;
}

}
