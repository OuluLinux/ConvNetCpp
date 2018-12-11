#include "LayerBase.h"


namespace ConvNet {

void LayerBase::InitRegression(int input_width, int input_height, int input_depth) {
	int input_count = input_width * input_height * input_depth;
	output_depth = input_count;
	output_width = 1;
	output_height = 1;
}

Volume& LayerBase::ForwardRegression(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input;
	return input; // identity function
}

double LayerBase::BackwardRegression(const Vector<double>& y) {
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	
	input.ZeroGradients(); // zero out the gradient of input Vol
	double loss = 0.0;
	
	for (int i = 0; i < output_depth; i++) {
		double dy = input.Get(i) - y[i];
		input.SetGradient(i,  dy);
		loss += 0.5 * dy * dy;
	}
	
	return loss;
}

double LayerBase::BackwardRegression(int pos, double y) {
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	input.ZeroGradients(); // zero out the gradient of input Vol
	double loss = 0.0;
	
	// lets hope that only one number is being regressed
	double dy = input.Get(pos) - y;
	if (!IsFin(dy)) return 0;
	input.SetGradient(pos, dy);
	loss += 0.5 * dy * dy;
	
	return loss;
}


double LayerBase::BackwardRegression(int cols, const Vector<int>& posv, const Vector<double>& yv) {
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	input.ZeroGradients(); // zero out the gradient of input Vol
	double loss = 0.0;
	
	ASSERT(posv.GetCount() == yv.GetCount());
	for(int i = 0; i < posv.GetCount(); i++) {
		int p = posv[i];
		ASSERT(p >= 0 && p < cols);
		int pos = i * cols + p;
		double y = yv[i];
		
		double dy = input.Get(pos) - y;
		input.SetGradient(pos, dy);
		loss += 0.5 * dy * dy;
	}
	
	return loss;
}


String LayerBase::ToStringRegression() const {
	return Format("Regression: w:%d, h:%d, d:%d",
		output_width, output_height, output_depth);
}

}
