#include "Layers.h"


namespace ConvNet {

SoftmaxLayer::SoftmaxLayer(int class_count) {
	if (class_count < 1)
		throw ArgumentException("SoftmaxLayer class_count must be a positive integer");
	this->class_count = class_count;
}

void SoftmaxLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	int input_count = input_width * input_height * input_depth;
	
	output_depth = input_count;
	output_width = 1;
	output_height = 1;
}

Volume& SoftmaxLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	
	output_activation.Init(1, 1, output_depth, 0.0);
	
	// compute max activation
	double amax = input.Get(0);
	for (int i = 1; i < output_depth; i++) {
		if (input.Get(i) > amax) {
			amax = input.Get(i);
		}
	}
	
	// compute exponentials (carefully to not blow up)
	es.SetCount(output_depth, 0.0);
	
	double esum = 0.0;
	
	for (int i = 0; i < output_depth; i++) {
		double e = exp(input.Get(i) - amax);
		esum += e;
		es[i] = e;
	}
	
	// normalize and output to sum to one
	for (int i = 0; i < output_depth; i++) {
		es[i] /= esum;
		output_activation.Set(i, es[i]);
	}
	
	return output_activation;
}

double SoftmaxLayer::Backward(int pos, double y) {
	
	// compute and accumulate gradient wrt weights and bias of this layer
	Volume& input = *input_activation;
	input.ZeroGradients(); // zero out the gradient of input Vol
	
	for (int i = 0; i < output_depth; i++) {
		double indicator = i == pos ? 1.0 : 0.0;
		double mul = -1.0 * (indicator - es[i]);
		input.SetGradient(i, mul);
	}
	
	// loss is the class negative log likelihood
	return -1.0 * log(es[pos]);
}

double SoftmaxLayer::Backward(const VolumeDataBase& y) {
	throw NotImplementedException();
}

void SoftmaxLayer::Backward() {
	throw NotImplementedException();
}

#define STOREVAR(json, field) map.GetAdd(#json) = this->field;
#define LOADVAR(field, json) this->field = map.GetValue(map.Find(#json));
#define LOADVARDEF(field, json, def) {Value tmp = map.GetValue(map.Find(#json)); if (tmp.IsNull()) this->field = def; else this->field = tmp;}

void SoftmaxLayer::Store(ValueMap& map) const {
	STOREVAR(out_depth, output_depth);
	STOREVAR(out_sx, output_width);
	STOREVAR(out_sy, output_height);
	STOREVAR(layer_type, GetKey());
	STOREVAR(num_inputs, class_count);
}

void SoftmaxLayer::Load(const ValueMap& map) {
	LOADVAR(output_depth, out_depth);
	LOADVAR(output_width, out_sx);
	LOADVAR(output_height, out_sy);
	LOADVAR(class_count, num_inputs);
}

}
