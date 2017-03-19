#include "ConvNet.h"

namespace ConvNet {

double sig(double x) {
	// helper function for computing sigmoid
	return 1.0 / (1.0 + exp(-x));
}

ReinforceBase::ReinforceBase() {
	input1 = NULL;
	input2 = NULL;
}

ReinforceBase::~ReinforceBase() {
	
}

Volume& ReinforceBase::Forward(Volume& input) {
	input1 = &input;
	input2 = NULL;
	return output;
}

Volume& ReinforceBase::Forward(Volume& input1, Volume& input2) {
	this->input1 = &input1;
	this->input2 = &input2;
	return output;
}

void ReinforceBase::Backward() {
	
}








ReinforceRowPluck::ReinforceRowPluck(int row) {
	ix = row;
}

ReinforceRowPluck::~ReinforceRowPluck() {
	
}

Volume& ReinforceRowPluck::Forward(Volume& input) {
	ReinforceBase::Forward(input);
	
	// pluck a row of input with index ix and return it as col vector
	ASSERT(ix >= 0 && ix < input.GetLength());
	int w = input.GetWidth();
	
	output.Init(1, w, 1, 0);
	for (int i = 0, h = w; i < h; i++) {
		output.Set(i, 0, 0, input.Get(ix, i, 0)); // copy over the data
	}
	return output;
}

void ReinforceRowPluck::Backward() {
	int w = input1->GetWidth();
	
	for (int i = 0, h = w; i < h; i++) {
		input1->AddGradient(i, ix, 0, output.GetGradient(i, 0, 0));
	}
}







ReinforceTanh::ReinforceTanh() {
	
}

ReinforceTanh::~ReinforceTanh() {
	
}

Volume& ReinforceTanh::Forward(Volume& input) {
	ReinforceBase::Forward(input);
	
	// tanh nonlinearity
	output.Init(input.GetWidth(), input.GetHeight(), input.GetDepth(), 0.0);
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		output.Set(i, tanh(input.Get(i)));
	}
	
	return output;
}

void ReinforceTanh::Backward() {
	int n = input1->GetLength();
	for (int i = 0; i < n; i++) {
		// grad for z = Tanh(x) is (1 - z^2)
		double mwi = output.Get(i);
		double d = (1.0 - mwi * mwi) * output.GetGradient(i);
		input1->AddGradient(i, d);
	}
}







ReinforceSigmoid::ReinforceSigmoid() {
	
}

ReinforceSigmoid::~ReinforceSigmoid() {
	
}

Volume& ReinforceSigmoid::Forward(Volume& input) {
	ReinforceBase::Forward(input);
	
	// sigmoid nonlinearity
	output.Init(input);
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		output.Set(i, sig(input.Get(i)));
	}
	
	return output;
}

void ReinforceSigmoid::Backward() {
	int n = input1->GetLength();
	for (int i = 0; i < n; i++) {
		// grad for z = Tanh(x) is (1 - z^2)
		double mwi = output.Get(i);
		input1->AddGradient(i, mwi * (1.0 - mwi) * output.GetGradient(i));
	}
}







ReinforceRelu::ReinforceRelu() {
	
}

ReinforceRelu::~ReinforceRelu() {
	
}

Volume& ReinforceRelu::Forward(Volume& input) {
	ReinforceBase::Forward(input);
	
	output.Init(input);
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		output.Set(i, max(0.0, input.Get(i))); // relu
	}
	
	return output;
}

void ReinforceRelu::Backward() {
	int n = input1->GetLength();
	for (int i = 0; i < n; i++) {
		input1->AddGradient(i, input1->Get(i) > 0 ? output.GetGradient(i) : 0.0);
	}
}







ReinforceMul::ReinforceMul() {
	
}

ReinforceMul::~ReinforceMul() {
	
}

Volume& ReinforceMul::Forward(Volume& input1, Volume& input2) {
	ReinforceBase::Forward(input1, input2);
	
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

void ReinforceMul::Backward() {
	
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







ReinforceAdd::ReinforceAdd() {
	
}

ReinforceAdd::~ReinforceAdd() {
	
}

Volume& ReinforceAdd::Forward(Volume& input1, Volume& input2) {
	ReinforceBase::Forward(input1, input2);
	
	ASSERT(input1.GetLength() == input2.GetLength());
	
	output.Init(input1);
	for (int i = 0; i < input1.GetLength(); i++) {
		output.Set(i, input1.Get(i) + input2.Get(i));
	}
	
	return output;
}

void ReinforceAdd::Backward() {
	for (int i = 0; i < input1->GetLength(); i++) {
		input1->AddGradient(i, output.GetGradient(i));
		input2->AddGradient(i, output.GetGradient(i));
	}
}







ReinforceDot::ReinforceDot() {
	
}

ReinforceDot::~ReinforceDot() {
	
}

Volume& ReinforceDot::Forward(Volume& input1, Volume& input2) {
	ReinforceBase::Forward(input1, input2);
	
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

void ReinforceDot::Backward() {
	for (int i = 0; i < input1->GetLength(); i++) {
		input1->AddGradient(i, input2->Get(i) * output.GetGradient(0));
		input2->AddGradient(i, input1->Get(i) * output.GetGradient(0));
	}
}







ReinforceEltMul::ReinforceEltMul() {
	
}

ReinforceEltMul::~ReinforceEltMul() {
	
}

Volume& ReinforceEltMul::Forward(Volume& input1, Volume& input2) {
	ReinforceBase::Forward(input1, input2);
	
	ASSERT(input1.GetLength() == input2.GetLength());
	
	output.Init(input1);
	
	for (int i = 0; i < input1.GetLength(); i++) {
		output.Set(i, input1.Get(i) * input2.Get(i));
	}
	
	return output;
}

void ReinforceEltMul::Backward() {
	for (int i = 0; i < input1->GetLength(); i++) {
		input1->AddGradient(i, input2->Get(i) * output.GetGradient(i));
		input2->AddGradient(i, input1->Get(i) * output.GetGradient(i));
	}
}

















Graph::Graph() {
	
}

Graph::~Graph() {
	Clear();
}

void Graph::Clear() {
	while (layers.GetCount()) {
		ReinforceBase* l = layers[0];
		layers.Remove(0);
		extra_args.Remove(0);
		delete l;
	}
}

Volume& Graph::Forward(Volume& input) {
	Volume* v = &input;
	for (int i = 0; i < layers.GetCount(); i++) {
		ReinforceBase& b = *layers[i];
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


void Graph::AddRowPluck(int row) {
	extra_args.Add(0);
	layers.Add(new ReinforceRowPluck(row));
}

void Graph::AddTanh() {
	extra_args.Add(0);
	layers.Add(new ReinforceTanh());
}

void Graph::AddSigmoid() {
	extra_args.Add(0);
	layers.Add(new ReinforceSigmoid());
}

void Graph::AddRelu() {
	extra_args.Add(0);
	layers.Add(new ReinforceRelu());
}

void Graph::AddMul(Volume& multiplier) {
	extra_args.Add(&multiplier);
	layers.Add(new ReinforceMul());
}

void Graph::AddAdd(Volume& addition) {
	extra_args.Add(&addition);
	layers.Add(new ReinforceAdd());
}

void Graph::AddDot(Volume& v) {
	extra_args.Add(&v);
	layers.Add(new ReinforceDot());
}

void Graph::AddEltMul(Volume& v) {
	extra_args.Add(&v);
	layers.Add(new ReinforceEltMul());
}

}
