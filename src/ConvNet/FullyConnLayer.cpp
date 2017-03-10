#include "Layers.h"


namespace ConvNet {

FullyConnLayer::FullyConnLayer(int neuron_count) : neuron_count(neuron_count) {
	if (neuron_count < 1)
		throw ArgumentException("Neuron count must be more than 0");
	
	input_count = 0;
	
	l1_decay_mul = 0.0;
	l2_decay_mul = 1.0;
}

void FullyConnLayer::Init(int input_width, int input_height, int input_depth) {
	LayerBase::Init(input_width, input_height, input_depth);
	
	// required
	// ok fine we will allow 'filters' as the word as well
	output_depth = neuron_count;
	
	// computed
	input_count = input_width * input_height * input_depth;
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

Volume& FullyConnLayer::Forward(Volume& input, bool is_training) {
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

void FullyConnLayer::Backward() {
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
}

Vector<ParametersAndGradients>& FullyConnLayer::GetParametersAndGradients() {
	
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
	
	return response;
}

#define STOREVAR(json, field) map.GetAdd(#json) = this->field;
#define LOADVAR(field, json) this->field = map.GetValue(map.Find(#json));
#define LOADVARDEF(field, json, def) {Value tmp = map.GetValue(map.Find(#json)); if (tmp.IsNull()) this->field = def; else this->field = tmp;}

void FullyConnLayer::Store(ValueMap& map) const {
	STOREVAR(out_depth, output_depth);
	STOREVAR(out_sx, output_width);
	STOREVAR(out_sy, output_height);
	STOREVAR(layer_type, GetKey());
	STOREVAR(num_inputs, input_count);
	STOREVAR(l1_decay_mul, l1_decay_mul);
	STOREVAR(l2_decay_mul, l2_decay_mul);
	
	Value filters;
	for (int i = 0; i < this->filters.GetCount(); i++) {
		ValueMap map;
		this->filters[i].Store(map);
		filters.Add(map);
	}
	map.GetAdd("filters") = filters;
	
	ValueMap biases;
	this->biases.Store(biases);
	map.GetAdd("biases") = biases;
}

void FullyConnLayer::Load(const ValueMap& map) {
	LOADVAR(output_depth, out_depth);
	LOADVAR(output_width, out_sx);
	LOADVAR(output_height, out_sy);
	LOADVAR(input_count, num_inputs);
	//LOADVARDEF(l1_decay_mul, l1_decay_mul, 1.0);
	//LOADVARDEF(l2_decay_mul, l2_decay_mul, 1.0);
	LOADVAR(l1_decay_mul, l1_decay_mul);
	LOADVAR(l2_decay_mul, l2_decay_mul);
	
	neuron_count = output_depth;
	
	filters.Clear();
	Value filters = map.GetValue(map.Find("filters"));
	
	for (int i = 0; i < output_depth; i++) {
		Volume& v = this->filters.Add();
		ValueMap values = filters[i];
		v.Load(values);
	}
	ValueMap values = map.GetValue(map.Find("biases"));
	biases.Load(values);
}

String FullyConnLayer::ToString() const {
	return Format("Fully Connected: w:%d, h:%d, d:%d, bias-pref:%2!,n, neurons%d, l1-decay:%2!,n, l2-decay:%2!,n",
		output_width, output_height, output_depth, bias_pref, neuron_count, l1_decay_mul, l2_decay_mul);
}

}
