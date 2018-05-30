#include "LayerBase.h"

namespace ConvNet {


Volume& LayerBase::ForwardInput(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input;
	return output_activation; // simply identity function for now
}

void LayerBase::BackwardInput() {
	
}

Volume& LayerBase::ForwardInput(bool is_training) {
	return output_activation;
}

String LayerBase::ToStringInput() const {
	return Format("Input: w:%d, h:%d, d:%d",
		output_width, output_height, output_depth);
}

}
