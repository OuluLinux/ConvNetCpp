#include "LayerBase.h"


namespace ConvNet {

void LayerBase::InitHeteroscedasticRegression(int input_width, int input_height, int input_depth) {
	int input_count = input_width * input_height * input_depth;
	output_depth = input_count;
	output_width = 1;
	output_height = 1;
	
	filters.SetCount(output_depth);
	for(int i = 0; i < filters.GetCount(); i++) {
		filters[i].Init(1, 1, input_count);
	}
	biases.Init(1, 1, output_depth, bias_pref);
}

Volume& LayerBase::ForwardHeteroscedasticRegression(Volume& input, bool is_training) {
	input_activation = &input;
	
	output_activation.Init(1, 1, output_depth, 0.0);
	
	for (int i = 0; i < output_depth; i++) {
		double a = 0.0;
		const Vector<double>& wi = filters[i].GetWeights();
		for(int j = 0; j < output_depth; j++) {
			a += input.Get(j) * wi[j];
		}
		a += biases.Get(i);
		output_activation.Set(i, a);
	}
	
	return output_activation; // identity function
}

double LayerBase::BackwardHeteroscedasticRegression(const Vector<double>& y) {
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	
	input.ZeroGradients(); // zero out the gradient of input Vol
	//output_activation.ZeroGradients();
	
	double loss = 0.0;
	
	int output_depth_2 = output_depth / 2;
	
	for (int i = 0; i < output_depth_2; i++) {
		double dy = output_activation.Get(i) - y[i];
		double ls2 = output_activation.Get(i + output_depth_2);
		double prec = exp(-ls2);
		
		Volume& Wmd = filters[i];
		Volume& Wsd = filters[i + output_depth_2];
		for(int j = 0; j < output_depth; j++) {
			input.AddGradient(j, prec * dy * Wmd.Get(j));
			input.AddGradient(j, -0.5 * (prec * dy * dy - 1) * Wsd.Get(j));
			Wmd.AddGradient(j, prec * dy * input.Get(j));
			Wsd.AddGradient(j, -0.5 * (prec * dy * dy - 1) * input.Get(j));
		}
		biases.AddGradient(i, prec * dy);
		biases.AddGradient(i + output_depth_2, -0.5 * (prec * dy * dy - 1));
		
		loss += 0.5 * prec * dy * dy;
	}
	
	return loss;
}

double LayerBase::BackwardHeteroscedasticRegression(int pos, double y) {
	Panic("Not implemented");
	return 0.0;
}


double LayerBase::BackwardHeteroscedasticRegression(int cols, const Vector<int>& posv, const Vector<double>& yv) {
	Panic("Not implemented");
	return 0.0;
}


String LayerBase::ToStringHeteroscedasticRegression() const {
	return Format("Regression: w:%d, h:%d, d:%d",
		output_width, output_height, output_depth);
}

}
