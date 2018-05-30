#include "ConvNet.h"

namespace ConvNet {

	
void Net::CheckLayer() {
	LayerBase& layer = layers.Top();
	
	int input_width = 0, input_height = 0, input_depth = 0;
	LayerBase* last_layer = NULL;
	
	if (layers.GetCount() > 1) {
		last_layer   = &layers[layers.GetCount() - 2];
		input_width  = last_layer->output_width;
		input_height = last_layer->output_height;
		input_depth  = last_layer->output_depth;
	}
	else if (!layer.IsInputLayer()) {
		throw ArgumentException("First layer should be an InputLayer");
	}
	
	
	if (layer.IsClassificationLayer()) {
		if (!last_layer->IsFullyConnLayer()) {
			throw ArgumentException("Previously added layer should be a FullyConnLayer with {classification_layer.class_count} Neurons");
		}
		
		if (last_layer->neuron_count != layer.class_count) {
			throw ArgumentException("Previous FullyConnLayer should have {classification_layer.class_count} Neurons");
		}
	}
	
	if (layer.IsRegressionLayer()) {
		if (!last_layer->IsFullyConnLayer()) {
			throw ArgumentException("Previously added layer should be a FullyConnLayer");
		}
	}
	
	if (layer.IsReluLayer()) {
		if (last_layer->IsDotProductLayer()) {
			last_layer->bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
			// otherwise it's technically possible that a relu unit will never turn on (by chance)
			// and will never get any gradient and never contribute any computation. Dead relu.
		}
	}
	
	layer.Init(input_width, input_height, input_depth);
}

Volume& Net::Forward(const Vector<VolumePtr>& inputs, bool is_training) {
	return Forward(*inputs[0], is_training);
}

Volume& Net::Forward(Volume& input, bool is_training) {
	Volume* activation = &layers[0].Forward(input, is_training);
	for (int i = 1; i < layers.GetCount(); i++) {
		LayerBase& layer_base = layers[i];
		activation = &layer_base.Forward(*activation, is_training);
	}
	return *activation;
}

double Net::GetCostLoss(Volume& input, int pos, double y) {
	Forward(input);
	
	LayerBase& last_layer = layers.Top();
	if (last_layer.IsLastLayer()) {
		double loss = last_layer.Backward(pos, y);
		return loss;
	}
	
	throw Exception("Last layer doesnt implement ILastLayer interface");
}

double Net::GetCostLoss(Volume& input, const Vector<double>& y) {
	Forward(input);
	
	LayerBase& last_layer = layers.Top();
	if (last_layer.IsLastLayer()) {
		double loss = last_layer.Backward(y);
		return loss;
	}
	
	throw Exception("Last layer doesnt implement ILastLayer interface");
}

double Net::Backward(int pos, double y) {
	int n = layers.GetCount();
	LayerBase& last_layer = layers.Top();
	if (last_layer.IsLastLayer()) {
		double loss = last_layer.Backward(pos, y); // last layer assumed to be loss layer
		for (int i = n - 2; i >= 0; i--) {
			// first layer assumed input
			layers[i].Backward();
		}
		return loss;
	}
	
	throw Exception("Last layer doesnt implement ILastLayer interface");
}

double Net::Backward(const Vector<double>& y) {
	int n = layers.GetCount();
	LayerBase& last_layer = layers.Top();
	if (last_layer.IsLastLayer()) {
		double loss = last_layer.Backward(y); // last layer assumed to be loss layer
		for (int i = n - 2; i >= 0; i--) {
			// first layer assumed input
			layers[i].Backward();
		}
		return loss;
	}
	
	throw Exception("Last layer doesnt implement ILastLayer interface");
}

double Net::Backward(int cols, const Vector<int>& pos, const Vector<double>& y) {
	int n = layers.GetCount();
	LayerBase& last_layer = layers.Top();
	if (last_layer.IsLastLayer()) {
		double loss = last_layer.Backward(cols, pos, y); // last layer assumed to be loss layer
		for (int i = n - 2; i >= 0; i--) {
			// first layer assumed input
			layers[i].Backward();
		}
		return loss;
	}
	
	throw Exception("Last layer doesnt implement ILastLayer interface");
}

int Net::GetPrediction() {
	// this is a convenience function for returning the argmax
	// prediction, assuming the last layer of the net is a softmax
	LayerBase& last_layer = layers.Top();
	if (!last_layer.IsSoftMaxLayer()) {
		throw Exception("GetPrediction function assumes softmax as last layer of the net!");
	}
	
	double maxv = last_layer.output_activation.Get(0);
	int maxi = 0;
	
	for (int i = 1; i < last_layer.output_activation.GetLength(); i++) {
		double d = last_layer.output_activation.Get(i);
		if (d > maxv) {
			maxv = d;
			maxi = i;
		}
	}
	
	return maxi; // return index of the class with highest class probability
}

Vector<ParametersAndGradients>& Net::GetParametersAndGradients() {
	int count = 0;
	
	int k = 0;
	for(int i = 0; i < layers.GetCount(); i++) {
		LayerBase& layer = layers[i];
		Vector<ParametersAndGradients>& pag = layer.GetParametersAndGradients();
		count += pag.GetCount();
		if (count > response.GetCount()) response.SetCount(count);
		for(int j = 0; j < pag.GetCount(); j++) {
			response[k] = pag[j];
			k++;
		}
	}
	
	response.SetCount(count);
	
	return response;
}

String Net::ToString() const {
	String s;
	for(int i = 0; i < layers.GetCount(); i++)
		s << layers[i].ToString() << "\n";
	return s;
}

}
