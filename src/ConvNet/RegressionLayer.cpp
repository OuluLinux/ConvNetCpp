#include "Layers.h"


namespace ConvNet {

RegressionLayer::RegressionLayer() {
	
}

void RegressionLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	int input_count = input_width * input_height * input_depth;
	output_depth = input_count;
	output_width = 1;
	output_height = 1;
}

Volume& RegressionLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input;
	return input; // identity function
}

void RegressionLayer::Backward() {
	throw NotImplementedException();
}

double RegressionLayer::Backward(const Vector<double>& y) {
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

double RegressionLayer::Backward(int pos, double y) {
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	input.ZeroGradients(); // zero out the gradient of input Vol
	double loss = 0.0;
	
	// lets hope that only one number is being regressed
	double dy = input.Get(pos) - y;
	input.SetGradient(pos, dy);
	loss += 0.5 * dy * dy;
	
	return loss;
}


#define STOREVAR(json, field) map.GetAdd(#json) = this->field;
#define LOADVAR(field, json) this->field = map.GetValue(map.Find(#json));
#define LOADVARDEF(field, json, def) {Value tmp = map.GetValue(map.Find(#json)); if (tmp.IsNull()) this->field = def; else this->field = tmp;}

void RegressionLayer::Store(ValueMap& map) const {
	STOREVAR(out_depth, output_depth);
	STOREVAR(out_sx, output_width);
	STOREVAR(out_sy, output_height);
	STOREVAR(layer_type, GetKey());
}

void RegressionLayer::Load(const ValueMap& map) {
	LOADVAR(output_depth, out_depth);
	LOADVAR(output_width, out_sx);
	LOADVAR(output_height, out_sy);
}

}
