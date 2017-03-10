#include "ConvNet.h"

namespace ConvNet {

	
void Net::AddLayer(LayerBase& layer) {
	
	int input_width = 0, input_height = 0, input_depth = 0;
	LayerBase* last_layer = NULL;
	
	if (layers.GetCount() > 0) {
		input_width = layers[layers.GetCount() - 1]->output_width;
		input_height = layers[layers.GetCount() - 1]->output_height;
		input_depth = layers[layers.GetCount() - 1]->output_depth;
		last_layer = layers[layers.GetCount() - 1];
	}
	else if (!dynamic_cast<InputLayer*>(&layer)) {
		throw ArgumentException("First layer should be an InputLayer");
	}
	
	IClassificationLayer* classification_layer = dynamic_cast<IClassificationLayer*>(&layer);
	
	
	if (classification_layer != NULL) {
		FullyConnLayer* fullcon_layer = dynamic_cast<FullyConnLayer*>(last_layer);
		if (fullcon_layer == NULL) {
			throw ArgumentException("Previously added layer should be a FullyConnLayer with {classification_layer.class_count} Neurons");
		}
		
		if (fullcon_layer->neuron_count != classification_layer->class_count) {
			throw ArgumentException("Previous FullyConnLayer should have {classification_layer.class_count} Neurons");
		}
	}
	
	RegressionLayer* regression_layer = dynamic_cast<RegressionLayer*>(&layer);
	if (regression_layer != NULL) {
		FullyConnLayer* fullcon_layer = dynamic_cast<FullyConnLayer*>(last_layer);
		if (fullcon_layer == NULL) {
			throw ArgumentException("Previously added layer should be a FullyConnLayer");
		}
	}
	
	ReluLayer* relu_layer = dynamic_cast<ReluLayer*>(&layer);
	if (relu_layer != NULL) {
		IDotProductLayer* dot_product_layer = dynamic_cast<IDotProductLayer*>(last_layer);
		if (dot_product_layer != NULL) {
			dot_product_layer->bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
			// otherwise it's technically possible that a relu unit will never turn on (by chance)
			// and will never get any gradient and never contribute any computation. Dead relu.
		}
	}
	
	if (layers.GetCount() > 0) {
		layer.Init(input_width, input_height, input_depth);
	}
	
	layers.Add(&layer);
}

Volume& Net::Forward(const Vector<VolumePtr>& inputs, bool is_training) {
	return Forward(*inputs[0], is_training);
}

Volume& Net::Forward(Volume& input, bool is_training) {
	Volume* activation = &layers[0]->Forward(input, is_training);
	for (int i = 1; i < layers.GetCount(); i++) {
		LayerBase& layer_base = *layers[i];
		activation = &layer_base.Forward(*activation, is_training);
	}
	return *activation;
}

double Net::GetCostLoss(Volume& input, int pos, double y) {
	Forward(input);
	
	LastLayerBase* last_layer = dynamic_cast<LastLayerBase*>(&*layers[layers.GetCount() - 1]);
	if (last_layer != NULL) {
		double loss = last_layer->Backward(pos, y);
		return loss;
	}
	
	throw Exception("Last layer doesnt implement ILastLayer interface");
}

double Net::GetCostLoss(Volume& input, const VolumeDataBase& y) {
	Forward(input);
	
	LastLayerBase* last_layer = dynamic_cast<LastLayerBase*>(&*layers[layers.GetCount() - 1]);
	if (last_layer != NULL) {
		double loss = last_layer->Backward(y);
		return loss;
	}
	
	throw Exception("Last layer doesnt implement ILastLayer interface");
}

double Net::Backward(int pos, double y) {
	int n = layers.GetCount();
	LastLayerBase* last_layer = dynamic_cast<LastLayerBase*>(&*layers[n - 1]);
	if (last_layer != NULL) {
		double loss = last_layer->Backward(pos, y); // last layer assumed to be loss layer
		for (int i = n - 2; i >= 0; i--) {
			// first layer assumed input
			layers[i]->Backward();
		}
		return loss;
	}
	
	throw Exception("Last layer doesnt implement ILastLayer interface");
}

double Net::Backward(const VolumeDataBase& y) {
	int n = layers.GetCount();
	LastLayerBase* last_layer = dynamic_cast<LastLayerBase*>(&*layers[n - 1]);
	if (last_layer != NULL) {
		double loss = last_layer->Backward(y); // last layer assumed to be loss layer
		for (int i = n - 2; i >= 0; i--) {
			// first layer assumed input
			layers[i]->Backward();
		}
		return loss;
	}
	
	throw Exception("Last layer doesnt implement ILastLayer interface");
}

int Net::GetPrediction() {
	// this is a convenience function for returning the argmax
	// prediction, assuming the last layer of the net is a softmax
	SoftmaxLayer* softmax_layer = dynamic_cast<SoftmaxLayer*>(&*layers[layers.GetCount() - 1]);
	if (softmax_layer == NULL) {
		throw Exception("GetPrediction function assumes softmax as last layer of the net!");
	}
	
	double maxv = softmax_layer->output_activation.Get(0);
	int maxi = 0;
	
	for (int i = 1; i < softmax_layer->output_activation.GetLength(); i++) {
		double d = softmax_layer->output_activation.Get(i);
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
		LayerBasePtr layer = layers[i];
		Vector<ParametersAndGradients>& pag = layer->GetParametersAndGradients();
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
		s << layers[i]->ToString() << "\n";
	return s;
}

}
