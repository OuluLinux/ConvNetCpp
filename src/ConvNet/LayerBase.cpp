#include "Layers.h"


namespace ConvNet {
	
LayerBase::LayerBase() {
	output_depth = 0;
	output_width = 0;
	output_height = 0;
	input_depth = 0;
	input_width = 0;
	input_height = 0;
	input_activation = NULL;
}

LayerBase::~LayerBase() {
	
}

Volume& LayerBase::Forward(bool is_training) {
	throw NotImplementedException();
}

Volume& LayerBase::Forward(Volume& input, bool is_training) {
	throw NotImplementedException();
}

void LayerBase::Init(int input_width, int input_height, int input_depth) {
	this->input_width = input_width;
	this->input_height = input_height;
	this->input_depth = input_depth;
}

Vector<ParametersAndGradients>& LayerBase::GetParametersAndGradients() {
	return response;
}

}
