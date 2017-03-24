#include "ConvNet.h"

namespace ConvNet {



double sig(double x) {
	// helper function for computing sigmoid
	return 1.0 / (1.0 + exp(-x));
}

RecurrentBase::RecurrentBase() {
	input1 = NULL;
	input2 = NULL;
}

RecurrentBase::RecurrentBase(Volume& input) {
	input1 = &input;
	input2 = NULL;
}

RecurrentBase::RecurrentBase(Volume& input1, Volume& input2) {
	this->input1 = &input1;
	this->input2 = &input2;
}

RecurrentBase::~RecurrentBase() {
	
}

Volume& RecurrentBase::Forward(Volume& input) {
	input1 = &input;
	input2 = NULL;
	return Forward();
}

Volume& RecurrentBase::Forward(Volume& input1, Volume& input2) {
	this->input1 = &input1;
	this->input2 = &input2;
	return Forward();
}









Volume& RecurrentRowPluck::Forward() {
	Volume& input = *input1;
	
	// pluck a row of input with index ix and return it as col vector
	ASSERT(*ix >= 0 && *ix < input.GetLength());
	int w = input.GetWidth();
	
	output.Init(1, w, 1, 0);
	for (int i = 0, h = w; i < h; i++) {
		output.Set(0, i, 0, input.Get(i, *ix, 0)); // copy over the data
	}
	return output;
}

void RecurrentRowPluck::Backward() {
	int w = input1->GetWidth();
	
	for (int i = 0; i < w; i++) {
		input1->AddGradient(i, *ix, 0, output.GetGradient(0, i, 0));
	}
}







Volume& RecurrentTanh::Forward() {
	Volume& input = *input1;
	
	// tanh nonlinearity
	output.Init(input.GetWidth(), input.GetHeight(), input.GetDepth(), 0.0);
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		output.Set(i, tanh(input.Get(i)));
	}
	
	return output;
}

void RecurrentTanh::Backward() {
	int n = input1->GetLength();
	for (int i = 0; i < n; i++) {
		// grad for z = Tanh(x) is (1 - z^2)
		double mwi = output.Get(i);
		double d = (1.0 - mwi * mwi) * output.GetGradient(i);
		input1->AddGradient(i, d);
	}
}








Volume& RecurrentSigmoid::Forward() {
	Volume& input = *input1;
	
	// sigmoid nonlinearity
	output.Init(input);
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		output.Set(i, sig(input.Get(i)));
	}
	
	return output;
}

void RecurrentSigmoid::Backward() {
	int n = input1->GetLength();
	for (int i = 0; i < n; i++) {
		// grad for z = Tanh(x) is (1 - z^2)
		double mwi = output.Get(i);
		input1->AddGradient(i, mwi * (1.0 - mwi) * output.GetGradient(i));
	}
}








Volume& RecurrentRelu::Forward() {
	Volume& input = *input1;
	
	output.Init(input);
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		output.Set(i, max(0.0, input.Get(i))); // relu
	}
	
	return output;
}

void RecurrentRelu::Backward() {
	int n = input1->GetLength();
	for (int i = 0; i < n; i++) {
		input1->AddGradient(i, input1->Get(i) > 0 ? output.GetGradient(i) : 0.0);
	}
}








Volume& RecurrentMul::Forward() {
	Volume& input1 = *this->input1;
	Volume& input2 = *this->input2;
	
	// multiply matrices input1 * input2
	ASSERT_(input1.GetWidth() == input2.GetHeight(), "matmul dimensions misaligned");
	
	int h = input1.GetHeight();
	int w = input2.GetWidth();
	output.Init(w, h, 1, 0);
	
	// loop over rows of input1
	for (int i = 0; i < h; i++) {
		
		// loop over cols of input2
		for (int j = 0; j < w; j++) {
			
			// dot product loop
			double dot = 0.0;
			for (int k = 0; k < input1.GetWidth(); k++) {
				dot += input1.Get(k, i, 0) * input2.Get(j, k, 0);
			}
			output.Set(j, i, 0, dot);
		}
	}
	return output;
}

void RecurrentMul::Backward() {
	
	// loop over rows of m1
	for (int i = 0; i < input1->GetHeight(); i++) {
		
		// loop over cols of m2
		for (int j = 0; j < input2->GetWidth(); j++) {
			
			// dot product loop
			for (int k = 0; k < input1->GetWidth(); k++) {
				double b = output.GetGradient(j, i, 0);
				input1->AddGradient(k, i, 0, input2->Get(j, k, 0) * b);
				input2->AddGradient(j, k, 0, input1->Get(k, i, 0) * b);
			}
		}
	}
}








Volume& RecurrentAdd::Forward() {
	Volume& input1 = *this->input1;
	Volume& input2 = *this->input2;
	
	ASSERT(input1.GetLength() == input2.GetLength());
	
	output.Init(input1);
	for (int i = 0; i < input1.GetLength(); i++) {
		output.Set(i, input1.Get(i) + input2.Get(i));
	}
	
	return output;
}

void RecurrentAdd::Backward() {
	for (int i = 0; i < input1->GetLength(); i++) {
		input1->AddGradient(i, output.GetGradient(i));
		input2->AddGradient(i, output.GetGradient(i));
	}
}









Volume& RecurrentDot::Forward() {
	Volume& input1 = *this->input1;
	Volume& input2 = *this->input2;
	
	// input1 and input2 are both column vectors
	ASSERT(input1.GetLength() == input2.GetLength());
	
	output.Init(1, 1, 1, 0);
	
	double dot = 0.0;
	for (int i = 0; i < input1.GetLength(); i++) {
		dot += input1.Get(i) * input2.Get(i);
	}
	
	output.Set(0, dot);
	return output;
}

void RecurrentDot::Backward() {
	for (int i = 0; i < input1->GetLength(); i++) {
		input1->AddGradient(i, input2->Get(i) * output.GetGradient(0));
		input2->AddGradient(i, input1->Get(i) * output.GetGradient(0));
	}
}








Volume& RecurrentEltMul::Forward() {
	Volume& input1 = *this->input1;
	Volume& input2 = *this->input2;
	
	ASSERT(input1.GetLength() == input2.GetLength());
	ASSERT(input1.GetLength() > 0);
	
	output.Init(input1);
	
	for (int i = 0; i < input1.GetLength(); i++) {
		output.Set(i, input1.Get(i) * input2.Get(i));
	}
	
	return output;
}

void RecurrentEltMul::Backward() {
	for (int i = 0; i < input1->GetLength(); i++) {
		input1->AddGradient(i, input2->Get(i) * output.GetGradient(i));
		input2->AddGradient(i, input1->Get(i) * output.GetGradient(i));
	}
}




Volume& RecurrentCopy::Forward() {
	*input2 = *input1;
	return *input2;
}

void RecurrentCopy::Backward() {
	*input1 = *input2;
}













Graph::Graph() {
	
}

Graph::~Graph() {
	Clear();
}

void Graph::Clear() {
	while (layers.GetCount()) {
		RecurrentBase* l = layers[0];
		layers.Remove(0);
		extra_args.Remove(0);
		delete l;
	}
}

Volume& Graph::Forward(Volume& input) {
	Volume* v = &input;
	for (int i = 0; i < layers.GetCount(); i++) {
		RecurrentBase& b = *layers[i];
		int args = b.GetArgCount();
		if (args == 1) {
			v = &b.Forward(*v);
		}
		else {
			v = &b.Forward(*extra_args[i], *v);
		}
	}
	return *v;
}

void Graph::Backward() {
	for (int i = layers.GetCount()-1; i >= 0; i--) {
		layers[i]->Backward();
	}
}


Volume& Graph::AddRowPluck(int* row) {
	extra_args.Add(0);
	return layers.Add(new RecurrentRowPluck(row))->output;
}

Volume& Graph::AddTanh() {
	extra_args.Add(0);
	return layers.Add(new RecurrentTanh())->output;
}

Volume& Graph::AddSigmoid() {
	extra_args.Add(0);
	return layers.Add(new RecurrentSigmoid())->output;
}

Volume& Graph::AddRelu() {
	extra_args.Add(0);
	return layers.Add(new RecurrentRelu())->output;
}

Volume& Graph::AddMul(Volume& multiplier) {
	extra_args.Add(&multiplier);
	return layers.Add(new RecurrentMul())->output;
}

Volume& Graph::AddAdd(Volume& addition) {
	extra_args.Add(&addition);
	return layers.Add(new RecurrentAdd())->output;
}

Volume& Graph::AddDot(Volume& v) {
	extra_args.Add(&v);
	return layers.Add(new RecurrentDot())->output;
}

Volume& Graph::AddEltMul(Volume& v) {
	extra_args.Add(&v);
	return layers.Add(new RecurrentEltMul())->output;
}













GraphTree::GraphTree() {
	
}

GraphTree::~GraphTree() {
	Clear();
}

void GraphTree::Clear() {
	while (layers.GetCount()) {
		RecurrentBase* l = layers[0];
		layers.Remove(0);
		delete l;
	}
}

Volume& GraphTree::Forward() {
	Volume* v = 0;
	for (int i = 0; i < layers.GetCount(); i++) {
		v = &layers[i]->Forward();
	}
	return *v;
}

void GraphTree::Backward() {
	for (int i = layers.GetCount()-1; i >= 0; i--) {
		layers[i]->Backward();
	}
}

Volume& GraphTree::AddRowPluck(int* row, Volume& in) {
	return layers.Add(new RecurrentRowPluck(row, in))->output;
}

Volume& GraphTree::AddTanh(Volume& in) {
	return layers.Add(new RecurrentTanh(in))->output;
}

Volume& GraphTree::AddSigmoid(Volume& in) {
	return layers.Add(new RecurrentSigmoid(in))->output;
}

Volume& GraphTree::AddRelu(Volume& in) {
	return layers.Add(new RecurrentRelu(in))->output;
}

Volume& GraphTree::AddMul(Volume& in1, Volume& in2) {
	return layers.Add(new RecurrentMul(in1, in2))->output;
}

Volume& GraphTree::AddAdd(Volume& in1, Volume& in2) {
	return layers.Add(new RecurrentAdd(in1, in2))->output;
}

Volume& GraphTree::AddDot(Volume& in1, Volume& in2) {
	return layers.Add(new RecurrentDot(in1, in2))->output;
}

Volume& GraphTree::AddEltMul(Volume& in1, Volume& in2) {
	return layers.Add(new RecurrentEltMul(in1, in2))->output;
}

Volume& GraphTree::AddCopy(Volume& src, Volume& dst) {
	return layers.Add(new RecurrentCopy(src, dst))->output;
}




Volume Softmax(const Volume& m) {
	Volume out(m.GetWidth(), m.GetHeight(), m.GetDepth(), 0.0); // probability volume
	double maxval = -DBL_MAX;
	
	for (int i = 0; i < m.GetLength(); i++) {
		if (m.Get(i) > maxval)
			maxval = m.Get(i);
	}
	
	double s = 0.0;
	
	for (int i = 0; i < m.GetLength(); i++) {
		out.Set(i, exp(m.Get(i) - maxval));
		s += out.Get(i);
	}
	
	for (int i = 0; i < m.GetLength(); i++) {
		out.Set(i, out.Get(i) / s);
	}
	
	// no backward pass here needed
	// since we will use the computed probabilities outside
	// to set gradients directly on m
	return out;
}

}


