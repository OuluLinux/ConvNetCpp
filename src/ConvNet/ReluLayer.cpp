#include "Layers.h"


namespace ConvNet {

ReluLayer::ReluLayer() {
	
}

void ReluLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	output_depth = input_depth;
	output_width = input_width;
	output_height = input_height;
}

Volume& ReluLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input;
	Volume& output = output_activation;
	
	for (int i = 0; i < input.GetLength(); i++) {
		if (output.Get(i) < 0) {
			output.Set(i, 0); // threshold at 0
		}
	}
	
	return output_activation;
}

void ReluLayer::Backward() {
	Volume& input = *input_activation; // we need to set dw of this
	int length = input.GetLength();
	
	input.ZeroGradients(); // zero out gradient wrt data
	
	for (int i = 0; i < length; i++)
	{
		if (output_activation.Get(i) <= 0) {
			input.SetGradient(i, 0); // threshold
		}
		else {
			input.SetGradient(i, output_activation.GetGradient(i));
		}
	}
}

#define STOREVAR(json, field) map.GetAdd(#json) = this->field;
#define LOADVAR(field, json) this->field = map.GetValue(map.Find(#json));
#define LOADVARDEF(field, json, def) {Value tmp = map.GetValue(map.Find(#json)); if (tmp.IsNull()) this->field = def; else this->field = tmp;}

void ReluLayer::Store(ValueMap& map) const {
	STOREVAR(out_depth, output_depth);
	STOREVAR(out_sx, output_width);
	STOREVAR(out_sy, output_height);
	STOREVAR(layer_type, GetKey());
}

void ReluLayer::Load(const ValueMap& map) {
	LOADVAR(output_depth, out_depth);
	LOADVAR(output_width, out_sx);
	LOADVAR(output_height, out_sy);
}

String ReluLayer::ToString() const {
	return Format("Relu: w:%d, h:%d, d:%d",
		output_width, output_height, output_depth);
}

}
