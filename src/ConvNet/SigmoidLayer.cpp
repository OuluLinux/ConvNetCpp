#include "LayerBase.h"


namespace ConvNet {

void LayerBase::InitSigmoid(int input_width, int input_height, int input_depth) {
	output_depth = input_depth;
	output_width = input_width;
	output_height = input_height;
}

Volume& LayerBase::ForwardSigmoid(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation.Init(input.GetWidth(), input.GetHeight(), input.GetDepth(), 0.0);
	
	int length = input.GetLength();
	
	for (int i = 0; i < length; i++) {
		output_activation.Set(i, 1.0 / (1.0 + exp(-1.0 * input.Get(i))));
	}
	
	return output_activation;
}

double LayerBase::BackwardSigmoid() {
	Volume& input = *input_activation; // we need to set dw of this
	Volume& output = output_activation;
	
	input.ZeroGradients(); // zero out gradient wrt data
	
	double loss = 0.0;
	for (int i = 0; i < input.GetLength(); i++) {
		double v2wi = output.Get(i);
		double dy = v2wi * (1.0 - v2wi) * output.GetGradient(i);
		input.SetGradient(i, dy);
		
		loss += 0.5 * dy * dy;
	}
	
	return loss;
}

double LayerBase::BackwardSigmoid(const Vector<double>& y) {
	Volume& input = *input_activation; // we need to set dw of this
	Volume& output = output_activation;
	
	for(int i = 0; i < y.GetCount(); i++)
		output.SetGradient(i, y[i]);
	
	return BackwardSigmoid();
}

String LayerBase::ToStringSigmoid() const {
	return Format("Sigmoid: w:%d, h:%d, d:%d",
		output_width, output_height, output_depth);
}

}
