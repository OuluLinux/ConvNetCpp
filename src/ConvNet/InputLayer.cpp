#include "Layers.h"

namespace ConvNet {

InputLayer::InputLayer(int input_width, int input_height, int input_depth) {
	if (!(input_width > 0 && input_height > 0 && input_depth > 0))
		throw ArgumentException("All volume components must be positive integers");
	
	Init(input_width, input_height, input_depth);
	
	output_width = input_width;
	output_height = input_height;
	output_depth = input_depth;
}

Volume& InputLayer::Forward(Volume& input, bool is_training) {
	input_activation = &input;
	output_activation = input;
	return output_activation; // simply identity function for now
}

void InputLayer::Backward() {
	
}

Volume& InputLayer::Forward(bool is_training) {
	return output_activation;
}

#define STOREVAR(json, field) map.GetAdd(#json) = this->field;
#define LOADVAR(field, json) this->field = map.GetValue(map.Find(#json));
#define LOADVARDEF(field, json, def) {Value tmp = map.GetValue(map.Find(#json)); if (tmp.IsNull()) this->field = def; else this->field = tmp;}

void InputLayer::Store(ValueMap& map) const {
	STOREVAR(out_depth, output_depth);
	STOREVAR(out_sx, output_width);
	STOREVAR(out_sy, output_height);
	STOREVAR(layer_type, GetKey());
}

void InputLayer::Load(const ValueMap& map) {
	LOADVAR(output_depth, out_depth);
	LOADVAR(output_width, out_sx);
	LOADVAR(output_height, out_sy);
}

String InputLayer::ToString() const {
	return Format("Input: w:%d, h:%d, d:%d",
		output_width, output_height, output_depth);
}

}
