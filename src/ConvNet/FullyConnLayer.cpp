#include "LayerBase.h"


namespace ConvNet {


void LayerBase::InitFullyConn(int input_width, int input_height, int input_depth) {
	
	// required
	// ok fine we will allow 'filters' as the word as well
	output_depth = neuron_count;
	
	// computed
	output_width = 1;
	output_height = 1;
	
	// initializations
	double bias = bias_pref;
	filters.SetCount(0);
	
	for (int i = 0; i < output_depth; i++) {
		filters.Add().Init(1, 1, input_count);
	}
	
	biases.Init(1, 1, output_depth, bias);
}

Volume& LayerBase::ForwardFullyConn(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation.Init(1, 1, output_depth, 0.0);
	
	for (int i = 0; i < output_depth; i++)
	{
		double a = 0.0;
		for (int d = 0; d < input_count; d++) {
			a += input.Get(d) * filters[i].Get(d); // for efficiency use Vols directly for now
		}
		
		a += biases.Get(i);
		output_activation.Set(i, a);
	}
	            
	return output_activation;
}

double LayerBase::BackwardFullyConn() {
	ASSERT(input_activation);
	Volume& input = *input_activation;
	ASSERT(output_activation.GetLength());
	
	input.ZeroGradients(); // zero out the gradient in input Vol
	
	// compute gradient wrt weights and data
	for (int i = 0; i < output_depth; i++)
	{
		Volume& tfi = filters[i];
		double chain_gradient_ = output_activation.GetGradient(i);
		
		for (int d = 0; d < input_count; d++) {
			input.SetGradient(d, input.GetGradient(d) + tfi.Get(d) * chain_gradient_); // grad wrt input data
			tfi.SetGradient(d, tfi.GetGradient(d) + input.Get(d) * chain_gradient_); // grad wrt params
		}
		biases.SetGradient(i, biases.GetGradient(i) + chain_gradient_);
	}
	
	double loss = 0;
	for(int i = 0; i < input_count; i++) {
		double dy = input.GetGradient(i);
		loss += 0.5 * dy * dy;
	}
	return loss;
}

double LayerBase::BackwardFullyConn(const Vector<double>& y) {
	Volume& input = *input_activation; // we need to set dw of this
	Volume& output = output_activation;
	
	for(int i = 0; i < y.GetCount(); i++)
		output.SetGradient(i, y[i]);
	
	return BackwardFullyConn();
}

String LayerBase::ToStringFullyConn() const {
	return Format("Fully Connected: w:%d, h:%d, d:%d, bias-pref:%2!,n, neurons:%d, l1-decay:%2!,n, l2-decay:%2!,n",
		output_width, output_height, output_depth, bias_pref, neuron_count, l1_decay_mul, l2_decay_mul);
}

}
