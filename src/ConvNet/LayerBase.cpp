#include "LayerBase.h"


namespace ConvNet {
	
LayerBase::LayerBase() {
	
}

LayerBase::~LayerBase() {
	
}

void LayerBase::Serialize(Stream& s) {
	s % output_activation
	  % output_depth
	  % output_width
	  % output_height
	  % input_depth
	  % input_width
	  % input_height
	  % input_count
	  % layer_type
	  % class_count
	  % bias_pref
	  % biases
	  % filters
	  % l1_decay_mul
	  % l2_decay_mul
	  % neuron_count
	  % S_cache
	  % k % alpha % beta
	  % n
	  % dropped
	  % drop_prob
	  % es
	  % width
	  % height
	  % filter_count
	  % stride
	  % pad
	  % switches
	  % group_size
	  % switchx
	  % switchy;
}

Volume& LayerBase::Forward(Volume& input, bool is_training) {
	switch (layer_type) {
		case NULL_LAYER:		Panic("Invalid null layer"); break;
		case FULLYCONN_LAYER:	return ForwardFullyConn(input, is_training); break;
		case LRN_LAYER:			return ForwardLrn(input, is_training); break;
		case DROPOUT_LAYER:		return ForwardDropOut(input, is_training); break;
		case INPUT_LAYER:		return ForwardInput(input, is_training); break;
		case SOFTMAX_LAYER:		return ForwardSoftmax(input, is_training); break;
		case REGRESSION_LAYER:	return ForwardRegression(input, is_training); break;
		case CONV_LAYER:		return ForwardConv(input, is_training); break;
		case POOL_LAYER:		return ForwardPool(input, is_training); break;
		case RELU_LAYER:		return ForwardRelu(input, is_training); break;
		case SIGMOID_LAYER:		return ForwardSigmoid(input, is_training); break;
		case TANH_LAYER:		return ForwardTanh(input, is_training); break;
		case MAXOUT_LAYER:		return ForwardMaxout(input, is_training); break;
		case SVM_LAYER:			return ForwardSVM(input, is_training); break;
		default: Panic("Type not implemented");
	}
	throw Exc();
}

Volume& LayerBase::Forward(bool is_training) {
	switch (layer_type) {
		case NULL_LAYER:		Panic("Invalid null layer"); break;
		case INPUT_LAYER:		return ForwardInput(is_training); break;
		default: Panic("Type not implemented");
	}
	throw Exc();
}

double LayerBase::Backward() {
	switch (layer_type) {
		case NULL_LAYER:		Panic("Invalid null layer"); return 0;
		case FULLYCONN_LAYER:	BackwardFullyConn(); return 0;
		case LRN_LAYER:			BackwardLrn(); return 0;
		case DROPOUT_LAYER:		BackwardDropOut(); return 0;
		case INPUT_LAYER:		BackwardInput(); return 0;
		//case SOFTMAX_LAYER:		BackwardSoftmax(); break;
		//case REGRESSION_LAYER:	BackwardRegression(); break;
		case CONV_LAYER:		BackwardConv(); return 0;
		case POOL_LAYER:		BackwardPool(); return 0;
		case RELU_LAYER:		BackwardRelu(); return 0;
		case SIGMOID_LAYER:		BackwardSigmoid(); return 0;
		case TANH_LAYER:		BackwardTanh(); return 0;
		case MAXOUT_LAYER:		BackwardMaxout(); return 0;
		//case SVM_LAYER:			BackwardSVM(); break;
		default: Panic("Type not implemented");
	}
	throw Exc();
}

double LayerBase::Backward(int pos, double y) {
	switch (layer_type) {
		case SOFTMAX_LAYER:		return BackwardSoftmax(pos, y); break;
		case REGRESSION_LAYER:	return BackwardRegression(pos, y); break;
		case SVM_LAYER:			return BackwardSVM(pos, y); break;
		default: Panic("Type not implemented");
	}
	throw Exc();
}

double LayerBase::Backward(const Vector<double>& y) {
	switch (layer_type) {
		case REGRESSION_LAYER:	return BackwardRegression(y); break;
		default: Panic("Type not implemented");
	}
	throw Exc();
}

double LayerBase::Backward(int cols, const Vector<int>& pos, const Vector<double>& y) {
	switch (layer_type) {
		case REGRESSION_LAYER:	return BackwardRegression(cols, pos, y); break;
		default: Panic("Type not implemented");
	}
	throw Exc();
}

String LayerBase::ToString() const {
	switch (layer_type) {
		case NULL_LAYER:		Panic("Invalid null layer"); break;
		case FULLYCONN_LAYER:	return ToStringFullyConn(); break;
		case LRN_LAYER:			return ToStringLrn(); break;
		case DROPOUT_LAYER:		return ToStringDropOut(); break;
		case INPUT_LAYER:		return ToStringInput(); break;
		case SOFTMAX_LAYER:		return ToStringSoftmax(); break;
		case REGRESSION_LAYER:	return ToStringRegression(); break;
		case CONV_LAYER:		return ToStringConv(); break;
		case POOL_LAYER:		return ToStringPool(); break;
		case RELU_LAYER:		return ToStringRelu(); break;
		case SIGMOID_LAYER:		return ToStringSigmoid(); break;
		case TANH_LAYER:		return ToStringTanh(); break;
		case MAXOUT_LAYER:		return ToStringMaxout(); break;
		case SVM_LAYER:			return ToStringSVM(); break;
		default: Panic("Type not implemented");
	}
	throw Exc();
}

String LayerBase::GetKey() const {
	switch (layer_type) {
		case NULL_LAYER: Panic("Invalid null layer"); break;
		case FULLYCONN_LAYER: return "fc"; break;
		case LRN_LAYER: return "lrn"; break;
		case DROPOUT_LAYER: return "dropout"; break;
		case INPUT_LAYER: return "input"; break;
		case SOFTMAX_LAYER: return "softmax"; break;
		case REGRESSION_LAYER: return "regression"; break;
		case CONV_LAYER: return "conv"; break;
		case POOL_LAYER: return "pool"; break;
		case RELU_LAYER: return "relu"; break;
		case SIGMOID_LAYER: return "sigmoid"; break;
		case TANH_LAYER: return "tanh"; break;
		case MAXOUT_LAYER: return "maxout"; break;
		case SVM_LAYER: return "svm"; break;
		default: Panic("Type not implemented");
	}
	throw Exc();
}

	
void LayerBase::Init(int input_width, int input_height, int input_depth) {
	this->input_width = input_width;
	this->input_height = input_height;
	this->input_depth = input_depth;
	input_count = input_width * input_height * input_depth;
	
	switch (layer_type) {
		case NULL_LAYER: Panic("Invalid null layer"); break;
		case FULLYCONN_LAYER:
			InitFullyConn(input_width, input_height, input_depth); break;
		case LRN_LAYER:
			InitLrn(input_width, input_height, input_depth); break;
		case DROPOUT_LAYER:
			InitDropOut(input_width, input_height, input_depth); break;
		case INPUT_LAYER:
			break;
		case SOFTMAX_LAYER:
			InitSoftmax(input_width, input_height, input_depth); break;
		case REGRESSION_LAYER:
			InitRegression(input_width, input_height, input_depth); break;
		case CONV_LAYER:
			InitConv(input_width, input_height, input_depth); break;
		case POOL_LAYER:
			InitPool(input_width, input_height, input_depth); break;
		case RELU_LAYER:
			InitRelu(input_width, input_height, input_depth); break;
		case SIGMOID_LAYER:
			InitSigmoid(input_width, input_height, input_depth); break;
		case TANH_LAYER:
			InitTanh(input_width, input_height, input_depth); break;
		case MAXOUT_LAYER:
			InitMaxout(input_width, input_height, input_depth); break;
		case SVM_LAYER:
			InitSVM(input_width, input_height, input_depth); break;
		default: Panic("Type not implemented");
	}
}


Vector<ParametersAndGradients>& LayerBase::GetParametersAndGradients() {
	
	if (!filters.IsEmpty()) {
		response.SetCount(output_depth + 1);
		
		for (int i = 0; i < output_depth; i++) {
			ParametersAndGradients& pag = response[i];
			pag.volume = &this->filters[i];
			pag.l2_decay_mul = &this->l2_decay_mul;
			pag.l1_decay_mul = &this->l1_decay_mul;
		}
		ParametersAndGradients& pag = response[output_depth];
		pag.volume = &this->biases;
		pag.l1_decay_mul = 0;
		pag.l2_decay_mul = 0;
	}
	
	return response;
}

}
