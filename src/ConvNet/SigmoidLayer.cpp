#include "Layers.h"


namespace ConvNet {

SigmoidLayer::SigmoidLayer() {
	
}

void SigmoidLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	output_depth = input_depth;
	output_width = input_width;
	output_height = input_height;
}

Volume& SigmoidLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation.Init(input.GetWidth(), input.GetHeight(), input.GetDepth(), 0.0);
	
	int length = input.GetLength();
	
	for (int i = 0; i < length; i++) {
		output_activation.Set(i, 1.0 / (1.0 + exp(-1.0 * input.Get(i))));
	}
	
	return output_activation;
}

void SigmoidLayer::Backward() {
	Volume& input = *input_activation; // we need to set dw of this
	Volume& output = output_activation;
	
	input.ZeroGradients(); // zero out gradient wrt data
	
	for (int i = 0; i < input.GetLength(); i++) {
		double v2wi = output.Get(i);
		input.SetGradient(i, v2wi * (1.0 - v2wi) * output.GetGradient(i));
	}
}

#define STOREVAR(json, field) map.GetAdd(#json) = this->field;
#define LOADVAR(field, json) this->field = map.GetValue(map.Find(#json));
#define LOADVARDEF(field, json, def) {Value tmp = map.GetValue(map.Find(#json)); if (tmp.IsNull()) this->field = def; else this->field = tmp;}

void SigmoidLayer::Store(ValueMap& map) const {
	STOREVAR(out_depth, output_depth);
	STOREVAR(out_sx, output_width);
	STOREVAR(out_sy, output_height);
	STOREVAR(layer_type, GetKey());
}

void SigmoidLayer::Load(const ValueMap& map) {
	LOADVAR(output_depth, out_depth);
	LOADVAR(output_width, out_sx);
	LOADVAR(output_height, out_sy);
}

String SigmoidLayer::ToString() const {
	return Format("Sigmoid: w:%d, h:%d, d:%d",
		output_width, output_height, output_depth);
}

}
