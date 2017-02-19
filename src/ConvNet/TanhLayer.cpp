#include "Layers.h"


namespace ConvNet {

TanhLayer::TanhLayer() {
	
}

void TanhLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	output_depth = input_depth;
	output_width = input_width;
	output_height = input_height;
}

Volume& TanhLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation.Init(input.GetWidth(), input.GetHeight(), input.GetDepth(), 0.0);
	int length = input.GetLength();
	
	for (int i = 0; i < length; i++) {
		output_activation.Set(i, tanh(input.Get(i)));
	}
	return output_activation;
}

void TanhLayer::Backward() {
	Volume& input = *input_activation; // we need to set dw of this
	Volume& output = output_activation;
	int length = input.GetLength();
	
	input.ZeroGradients(); // zero out gradient wrt data
	
	for (int i = 0; i < length; i++)
	{
		double v2wi = output.Get(i);
		input.SetGradient(i, (1.0 - v2wi * v2wi) * output.GetGradient(i));
	}
}

}
