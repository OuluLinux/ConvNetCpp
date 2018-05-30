#include "ConvNet.h"

namespace ConvNet {



double sig(double x) {
	// helper function for computing sigmoid
	return 1.0 / (1.0 + exp(-x));
}

RecurrentBase::RecurrentBase() {
	
}

RecurrentBase& RecurrentBase::InitOutput(MatPool& pool) {
	this->pool = &pool;
	ASSERT(this->pool);
	pool.InitMat(output);
	return *this;
}

RecurrentBase& RecurrentBase::SetPool(MatPool& pool) {
	this->pool = &pool;
	ASSERT(this->pool);
	return *this;
}

RecurrentBase& RecurrentBase::SetInput(MatId input) {
	input1 = input;
	return *this;
}

RecurrentBase& RecurrentBase::SetInput(MatId input1, MatId input2) {
	this->input1 = input1;
	this->input2 = input2;
	return *this;
}

MatId RecurrentBase::Forward(MatId input) {
	input1 = input;
	input2.value = -1;
	
	switch (recurrent_type) {
		case RECURRENT_NULL:		Panic("Invalid recurrent null layer");
		case RECURRENT_ROWPLUCK:	return ForwardRowPluck();
		case RECURRENT_TANH:		return ForwardTanh();
		case RECURRENT_SIGMOID:		return ForwardSigmoid();
		case RECURRENT_RELU:		return ForwardRelu();
		case RECURRENT_MUL:			return ForwardMul();
		case RECURRENT_ADD:			return ForwardAdd();
		case RECURRENT_DOT:			return ForwardDot();
		case RECURRENT_ELTMUL:		return ForwardEltMul();
		case RECURRENT_COPY:		return ForwardCopy();
		case RECURRENT_ADDCONST:	return ForwardAddConst();
		case RECURRENT_MULCONST:	return ForwardMulConst();
	}
	throw Exc("Never");
}

MatId RecurrentBase::Forward(MatId input1, MatId input2) {
	this->input1 = input1;
	this->input2 = input2;
	
	switch (recurrent_type) {
		case RECURRENT_NULL:		Panic("Invalid recurrent null layer");
		case RECURRENT_ROWPLUCK:	return ForwardRowPluck();
		case RECURRENT_TANH:		return ForwardTanh();
		case RECURRENT_SIGMOID:		return ForwardSigmoid();
		case RECURRENT_RELU:		return ForwardRelu();
		case RECURRENT_MUL:			return ForwardMul();
		case RECURRENT_ADD:			return ForwardAdd();
		case RECURRENT_DOT:			return ForwardDot();
		case RECURRENT_ELTMUL:		return ForwardEltMul();
		case RECURRENT_COPY:		return ForwardCopy();
		case RECURRENT_ADDCONST:	return ForwardAddConst();
		case RECURRENT_MULCONST:	return ForwardMulConst();
	}
	throw Exc("Never");
}

MatId RecurrentBase::Forward() {
	switch (recurrent_type) {
		case RECURRENT_NULL:		Panic("Invalid recurrent null layer");
		case RECURRENT_ROWPLUCK:	return ForwardRowPluck();
		case RECURRENT_TANH:		return ForwardTanh();
		case RECURRENT_SIGMOID:		return ForwardSigmoid();
		case RECURRENT_RELU:		return ForwardRelu();
		case RECURRENT_MUL:			return ForwardMul();
		case RECURRENT_ADD:			return ForwardAdd();
		case RECURRENT_DOT:			return ForwardDot();
		case RECURRENT_ELTMUL:		return ForwardEltMul();
		case RECURRENT_COPY:		return ForwardCopy();
		case RECURRENT_ADDCONST:	return ForwardAddConst();
		case RECURRENT_MULCONST:	return ForwardMulConst();
	}
	throw Exc("Never");
}

void RecurrentBase::Backward() {
	switch (recurrent_type) {
		case RECURRENT_NULL:		Panic("Invalid recurrent null layer");
		case RECURRENT_ROWPLUCK:	BackwardRowPluck(); return;
		case RECURRENT_TANH:		BackwardTanh(); return;
		case RECURRENT_SIGMOID:		BackwardSigmoid(); return;
		case RECURRENT_RELU:		BackwardRelu(); return;
		case RECURRENT_MUL:			BackwardMul(); return;
		case RECURRENT_ADD:			BackwardAdd(); return;
		case RECURRENT_DOT:			BackwardDot(); return;
		case RECURRENT_ELTMUL:		BackwardEltMul(); return;
		case RECURRENT_COPY:		BackwardCopy(); return;
		case RECURRENT_ADDCONST:	BackwardAddConst(); return;
		case RECURRENT_MULCONST:	BackwardMulConst(); return;
	}
	throw Exc("Never");
}

String RecurrentBase::GetKey() const {
	switch (recurrent_type) {
		case RECURRENT_NULL:		Panic("Invalid recurrent null layer");
		case RECURRENT_ROWPLUCK:	return "RowPluck";
		case RECURRENT_TANH:		return "Tanh";
		case RECURRENT_SIGMOID:		return "Sigmoid";
		case RECURRENT_RELU:		return "Relu";
		case RECURRENT_MUL:			return "Mul";
		case RECURRENT_ADD:			return "Add";
		case RECURRENT_DOT:			return "Dot";
		case RECURRENT_ELTMUL:		return "EltMul";
		case RECURRENT_COPY:		return "Copy";
		case RECURRENT_ADDCONST:	return "AddConst";
		case RECURRENT_MULCONST:	return "MulConst";
	}
	throw Exc("Never");
}

int RecurrentBase::GetArgCount() {
	switch (recurrent_type) {
		case RECURRENT_NULL:		Panic("Invalid recurrent null layer");
		case RECURRENT_ROWPLUCK:	return 0;
		case RECURRENT_TANH:		return 1;
		case RECURRENT_SIGMOID:		return 1;
		case RECURRENT_RELU:		return 1;
		case RECURRENT_MUL:			return 2;
		case RECURRENT_ADD:			return 2;
		case RECURRENT_DOT:			return 2;
		case RECURRENT_ELTMUL:		return 2;
		case RECURRENT_COPY:		return 2;
		case RECURRENT_ADDCONST:	return 1;
		case RECURRENT_MULCONST:	return 1;
	}
	throw Exc("Never");
}









MatId RecurrentBase::ForwardRowPluck() {
	MatPool& ses = *pool;
	Mat& input = ses.Get(input1);
	Mat& output = ses.Get(this->output);
	int chr = ses.GetInput(this->ix);
	
	// pluck a row of input with index ix and return it as col vector
	ASSERT(chr >= 0 && chr < input.GetLength());
	int w = input.GetWidth();
	
	output.Init(1, w, 0);
	for (int i = 0, h = w; i < h; i++) {
		output.Set(0, i, input.Get(i, chr)); // copy over the data
	}
	return this->output;
}

void RecurrentBase::BackwardRowPluck() {
	MatPool& ses = *pool;
	Mat& input = ses.Get(input1);
	Mat& output = ses.Get(this->output);
	int chr = ses.GetInput(this->ix);
	
	int w = input.GetWidth();
	
	for (int i = 0; i < w; i++) {
		input.AddGradient(i, chr, output.GetGradient(0, i));
	}
}







MatId RecurrentBase::ForwardTanh() {
	MatPool& ses = *pool;
	Mat& input = ses.Get(input1);
	Mat& output = ses.Get(this->output);
	
	// tanh nonlinearity
	output.Init(input.GetWidth(), input.GetHeight(), 0.0);
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		output.Set(i, tanh(input.Get(i)));
	}
	
	return this->output;
}

void RecurrentBase::BackwardTanh() {
	MatPool& ses = *pool;
	Mat& input = ses.Get(input1);
	Mat& output = ses.Get(this->output);
	
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		// grad for z = Tanh(x) is (1 - z^2)
		double mwi = output.Get(i);
		double d = (1.0 - mwi * mwi) * output.GetGradient(i);
		input.AddGradient(i, d);
	}
}








MatId RecurrentBase::ForwardSigmoid() {
	MatPool& ses = *pool;
	Mat& input = ses.Get(input1);
	Mat& output = ses.Get(this->output);
	
	// sigmoid nonlinearity
	output.Init(input.GetWidth(), input.GetHeight(), 0.0);
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		output.Set(i, sig(input.Get(i)));
	}
	
	return this->output;
}

void RecurrentBase::BackwardSigmoid() {
	MatPool& ses = *pool;
	Mat& input = ses.Get(input1);
	Mat& output = ses.Get(this->output);
	
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		// grad for z = Tanh(x) is (1 - z^2)
		double mwi = output.Get(i);
		input.AddGradient(i, mwi * (1.0 - mwi) * output.GetGradient(i));
	}
}








MatId RecurrentBase::ForwardRelu() {
	MatPool& ses = *pool;
	Mat& input = ses.Get(input1);
	Mat& output = ses.Get(this->output);
	
	output.Init(input.GetWidth(), input.GetHeight(), 0.0);
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		output.Set(i, max(0.0, input.Get(i))); // relu
	}
	
	return this->output;
}

void RecurrentBase::BackwardRelu() {
	MatPool& ses = *pool;
	Mat& input = ses.Get(input1);
	Mat& output = ses.Get(this->output);
	
	int n = input.GetLength();
	for (int i = 0; i < n; i++) {
		input.AddGradient(i, input.Get(i) > 0 ? output.GetGradient(i) : 0.0);
	}
}








MatId RecurrentBase::ForwardMul() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	Mat& output = ses.Get(this->output);
	
	// multiply matrices input1 * input2
	ASSERT_(input1.GetWidth() == input2.GetHeight(), "matmul dimensions misaligned");
	
	int h = input1.GetHeight();
	int w = input2.GetWidth();
	output.Init(w, h, 0.0);
	
	// loop over rows of input1
	for (int i = 0; i < h; i++) {
		
		// loop over cols of input2
		for (int j = 0; j < w; j++) {
			
			// dot product loop
			double dot = 0.0;
			for (int k = 0; k < input1.GetWidth(); k++) {
				dot += input1.Get(k, i) * input2.Get(j, k);
			}
			output.Set(j, i, dot);
		}
	}
	return this->output;
}

void RecurrentBase::BackwardMul() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	Mat& output = ses.Get(this->output);
	
	// loop over rows of m1
	for (int i = 0; i < input1.GetHeight(); i++) {
		
		// loop over cols of m2
		for (int j = 0; j < input2.GetWidth(); j++) {
			
			// dot product loop
			for (int k = 0; k < input1.GetWidth(); k++) {
				double b = output.GetGradient(j, i);
				input1.AddGradient(k, i, input2.Get(j, k) * b);
				input2.AddGradient(j, k, input1.Get(k, i) * b);
			}
		}
	}
}








MatId RecurrentBase::ForwardAdd() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	Mat& output = ses.Get(this->output);
	
	ASSERT(input1.GetLength() == input2.GetLength());
	
	output.Init(input1.GetWidth(), input1.GetHeight(), 0.0);
	for (int i = 0; i < input1.GetLength(); i++) {
		output.Set(i, input1.Get(i) + input2.Get(i));
	}
	
	return this->output;
}

void RecurrentBase::BackwardAdd() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	Mat& output = ses.Get(this->output);
	
	for (int i = 0; i < input1.GetLength(); i++) {
		input1.AddGradient(i, output.GetGradient(i));
		input2.AddGradient(i, output.GetGradient(i));
	}
}









MatId RecurrentBase::ForwardDot() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	Mat& output = ses.Get(this->output);
	
	// input1 and input2 are both column vectors
	ASSERT(input1.GetLength() == input2.GetLength());
	
	output.Init(1, 1, 0);
	
	double dot = 0.0;
	for (int i = 0; i < input1.GetLength(); i++) {
		dot += input1.Get(i) * input2.Get(i);
	}
	
	output.Set(0, dot);
	return this->output;
}

void RecurrentBase::BackwardDot() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	Mat& output = ses.Get(this->output);
	
	for (int i = 0; i < input1.GetLength(); i++) {
		input1.AddGradient(i, input2.Get(i) * output.GetGradient(0));
		input2.AddGradient(i, input1.Get(i) * output.GetGradient(0));
	}
}








MatId RecurrentBase::ForwardEltMul() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	Mat& output = ses.Get(this->output);
	
	ASSERT(input1.GetLength() == input2.GetLength());
	ASSERT(input1.GetLength() > 0);
	
	output.Init(input1.GetWidth(), input1.GetHeight(), 0.0);
	
	for (int i = 0; i < input1.GetLength(); i++) {
		output.Set(i, input1.Get(i) * input2.Get(i));
	}
	
	return this->output;
}

void RecurrentBase::BackwardEltMul() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	Mat& output = ses.Get(this->output);
	
	for (int i = 0; i < input1.GetLength(); i++) {
		input1.AddGradient(i, input2.Get(i) * output.GetGradient(i));
		input2.AddGradient(i, input1.Get(i) * output.GetGradient(i));
	}
}




MatId RecurrentBase::ForwardCopy() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	
	input2 = input1;
	return this->input2;
}

void RecurrentBase::BackwardCopy() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	
	input1 = input2;
}






MatId RecurrentBase::ForwardAddConst() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	Mat& output = ses.Get(this->output);
	
	output.Init(input1.GetWidth(), input1.GetHeight(), 0.0);
	for (int i = 0; i < input1.GetLength(); i++) {
		output.Set(i, input1.Get(i) + d);
	}
	
	return this->output;
}

void RecurrentBase::BackwardAddConst() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& input2 = ses.Get(this->input2);
	Mat& output = ses.Get(this->output);
	
	for (int i = 0; i < input1.GetLength(); i++) {
		input1.AddGradient(i, output.GetGradient(i));
	}
}









MatId RecurrentBase::ForwardMulConst() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& output = ses.Get(this->output);
	
	output.Init(input1.GetWidth(), input1.GetHeight(), 0.0);
	for (int i = 0; i < input1.GetLength(); i++) {
		output.Set(i, input1.Get(i) * d);
	}
	
	return this->output;
}

void RecurrentBase::BackwardMulConst() {
	MatPool& ses = *pool;
	Mat& input1 = ses.Get(this->input1);
	Mat& output = ses.Get(this->output);
	
	for (int i = 0; i < input1.GetLength(); i++) {
		input1.AddGradient(i, output.GetGradient(i));
	}
}













Graph::Graph() {
	
}

Graph::~Graph() {
	Clear();
}

void Graph::Clear() {
	layers.Clear();
	extra_args.Clear();
}

MatId Graph::Forward(MatId input) {
	MatId v = input;
	for (int i = 0; i < layers.GetCount(); i++) {
		RecurrentBase& b = layers[i];
		int args = b.GetArgCount();
		if (args == 1) {
			v = b.Forward(v);
		}
		else {
			v = b.Forward(extra_args[i], v);
		}
	}
	return v;
}

void Graph::Backward() {
	for (int i = layers.GetCount()-1; i >= 0; i--) {
		layers[i].Backward();
	}
}


MatId Graph::RowPluck(int row) {
	extra_args.Add();
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_ROWPLUCK).SetInt(row).output;
}

MatId Graph::Tanh() {
	extra_args.Add();
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_TANH).output;
}

MatId Graph::Sigmoid() {
	extra_args.Add();
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_SIGMOID).output;
}

MatId Graph::Relu() {
	extra_args.Add();
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_RELU).output;
}

MatId Graph::Mul(MatId multiplier) {
	ASSERT(multiplier.value != -1);
	extra_args.Add(multiplier);
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_MUL).output;
}

MatId Graph::Add(MatId addition) {
	ASSERT(addition.value != -1);
	extra_args.Add(addition);
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_ADD).output;
}

MatId Graph::Dot(MatId v) {
	ASSERT(v.value != -1);
	extra_args.Add(v);
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_DOT).output;
}

MatId Graph::EltMul(MatId v) {
	ASSERT(v.value != -1);
	extra_args.Add(v);
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_ELTMUL).output;
}












GraphTree::GraphTree() {
	
}

GraphTree::~GraphTree() {
	
}

void GraphTree::Clear() {
	layers.Clear();
}

MatId GraphTree::Forward() {
	MatId v;
	for (int i = 0; i < layers.GetCount(); i++) {
		v = layers[i].Forward();
	}
	return v;
}

void GraphTree::Backward() {
	for (int i = layers.GetCount()-1; i >= 0; i--) {
		layers[i].Backward();
	}
}

MatId GraphTree::RowPluck(int row, MatId in) {
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_ROWPLUCK).SetInt(row).SetInput(in).output;
}

MatId GraphTree::Tanh(MatId in) {
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_TANH).SetInput(in).output;
}

MatId GraphTree::Sigmoid(MatId in) {
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_SIGMOID).SetInput(in).output;
}

MatId GraphTree::Relu(MatId in) {
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_RELU).SetInput(in).output;
}

MatId GraphTree::Mul(MatId in1, MatId in2) {
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_MUL).SetInput(in1, in2).output;
}

MatId GraphTree::Add(MatId in1, MatId in2) {
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_ADD).SetInput(in1, in2).output;
}

MatId GraphTree::Dot(MatId in1, MatId in2) {
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_DOT).SetInput(in1, in2).output;
}

MatId GraphTree::EltMul(MatId in1, MatId in2) {
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_ELTMUL).SetInput(in1, in2).output;
}

MatId GraphTree::Copy(MatId src, MatId dst) {
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_COPY).SetInput(src, dst).output;
}

MatId GraphTree::AddConstant(double d, MatId in) {
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_ADDCONST).SetInput(in).SetDouble(d).output;
}

MatId GraphTree::MulConstant(double d, MatId in) {
	return layers.Add().InitOutput(*pool).SetType(RECURRENT_MULCONST).SetInput(in).SetDouble(d).output;
}


void Softmax(const Mat& m, Mat& out) {
	out.Init(m.GetWidth(), m.GetHeight(), 0.0); // probability volume
	double maxval = -DBL_MAX;
	
	for (int i = 0; i < m.GetLength(); i++) {
		if (m.Get(i) > maxval)
			maxval = m.Get(i);
	}
	
	double s = 0.0;
	
	for (int i = 0; i < m.GetLength(); i++) {
		double d = exp(m.Get(i) - maxval);
		out.Set(i, d);
		s += d;
	}
	
	for (int i = 0; i < m.GetLength(); i++) {
		out.Set(i, out.Get(i) / s);
	}
	
	// no backward pass here needed
	// since we will use the computed probabilities outside
	// to set gradients directly on m
}

}


