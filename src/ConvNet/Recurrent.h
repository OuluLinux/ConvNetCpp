#ifndef _ConvNet_Recurrent_h_
#define _ConvNet_Recurrent_h_

#include "Mat.h"

/*
	Recurrent.h is a C++ conversion of Recurrent.js by Andrej Karpathy.
	MIT License.

	Original source:	https://github.com/karpathy/reinforcejs (fork of recurrentjs)
*/

namespace ConvNet {

enum {
	RECURRENT_NULL, RECURRENT_ROWPLUCK, RECURRENT_TANH, RECURRENT_SIGMOID, RECURRENT_RELU,
	RECURRENT_MUL, RECURRENT_ADD, RECURRENT_DOT, RECURRENT_ELTMUL, RECURRENT_COPY,
	RECURRENT_ADDCONST, RECURRENT_MULCONST
};

class Graph;
class RecurrentSession;

class RecurrentBase : Moveable<RecurrentBase> {
	
public:
	
	MatPool* pool = NULL;
	
	MatId input1, input2;
	MatId output;
	double d = 0.0;
	int ix = -1;
	int recurrent_type = RECURRENT_NULL;
	
	RecurrentBase();
	RecurrentBase& SetInt(int i) {ix = i; return *this;}
	RecurrentBase& SetDouble(double d) {this->d = d; return *this;}
	RecurrentBase& SetType(int i) {recurrent_type = i; return *this;}
	RecurrentBase& SetInput(MatId input);
	RecurrentBase& SetInput(MatId input1, MatId input2);
	RecurrentBase& InitOutput(MatPool& pool);
	RecurrentBase& SetPool(MatPool& pool);
	
	void Serialize(Stream& s) {
		s % input1 % input2
		  % output
		  % d
		  % ix
		  % recurrent_type;
	}
	
	String GetKey() const;
	int GetArgCount();
	
	MatId Forward();
	MatId Forward(MatId input);
	MatId Forward(MatId input1, MatId input2);
	void Backward();
	
	MatId ForwardRowPluck();
	MatId ForwardTanh();
	MatId ForwardSigmoid();
	MatId ForwardRelu();
	MatId ForwardMul();
	MatId ForwardAdd();
	MatId ForwardDot();
	MatId ForwardEltMul();
	MatId ForwardCopy();
	MatId ForwardAddConst();
	MatId ForwardMulConst();
	void BackwardRowPluck();
	void BackwardTanh();
	void BackwardSigmoid();
	void BackwardRelu();
	void BackwardMul();
	void BackwardAdd();
	void BackwardDot();
	void BackwardEltMul();
	void BackwardCopy();
	void BackwardAddConst();
	void BackwardMulConst();
	
	//Mat& operator() (Mat& a) {return Forward(a);}
	//Mat& operator() (Mat& a, Mat& b) {return Forward(a,b);}
	
};



class Graph {
	
protected:
	friend class RecurrentBase;
	
	Vector<RecurrentBase> layers;
	Vector<MatId> extra_args;
	MatPool* pool = NULL;
	
public:
	Graph();
	~Graph();
	
	void Serialize(Stream& s) {
		s % layers % extra_args;
	}
	
	void FixPool() {
		for(int i = 0; i < layers.GetCount(); i++)
			layers[i].SetPool(*pool);
	}
	
	void Clear();
	MatId Forward(MatId input);
	void Backward();
	
	MatId RowPluck(int row);
	MatId Tanh();
	MatId Sigmoid();
	MatId Relu();
	MatId Mul(MatId multiplier);
	MatId Add(MatId addition);
	MatId Dot(MatId v);
	MatId EltMul(MatId v);
	
	RecurrentBase& GetLayer(int i) {return layers[i];}
	int GetCount() const {return layers.GetCount();}
	
	void SetPool(MatPool& p) {pool = &p;}
	MatPool& GetPool() {return *pool;}
	
};


// GraphTree allows custom connections between layers.
class GraphTree {
	Vector<RecurrentBase> layers;
	MatPool* pool = NULL;
	
public:
	GraphTree();
	~GraphTree();
	
	void Serialize(Stream& s) {
		s % layers;
	}
	
	void FixPool() {
		for(int i = 0; i < layers.GetCount(); i++)
			layers[i].SetPool(*pool);
	}
	
	void Clear();
	MatId Forward();
	void Backward();
	
	MatId RowPluck(int row, MatId in);
	MatId Tanh(MatId in);
	MatId Sigmoid(MatId in);
	MatId Relu(MatId in);
	MatId Mul(MatId in1, MatId in2);
	MatId Add(MatId in1, MatId in2);
	MatId Dot(MatId in1, MatId in2);
	MatId EltMul(MatId in1, MatId in2);
	MatId Copy(MatId src, MatId dst);
	MatId AddConstant(double d, MatId in);
	MatId MulConstant(double d, MatId in);
	
	RecurrentBase& GetLayer(int i) {return layers[i];}
	int GetCount() const {return layers.GetCount();}
	
	RecurrentBase& Top() {return layers.Top();}
	
	void SetPool(MatPool& p) {pool = &p;}
	MatPool& GetPool() {return *pool;}
};




struct HighwayModel : Moveable<HighwayModel> {
	
	MatId noise_h[2];
	
	static int GetCount() {return 2;}
	
	MatId GetMat(int i) {
		ASSERT(i >= 0 && i < 6);
		switch (i) {
			case 0: return noise_h[0];
			case 1: return noise_h[1];
			default: return noise_h[1];
		}
	}
	
	void Serialize(Stream& s) {
		s % noise_h[0] % noise_h[1];
	}
	
};

struct LSTMModel : Moveable<LSTMModel> {
	
	MatId Wix, Wih, bi, Wfx, Wfh, bf, Wox, Woh, bo, Wcx, Wch, bc;
	
	static int GetCount() {return 12;}
	MatId GetMat(int i) {
		ASSERT(i >= 0 && i < 12);
		switch (i) {
			case 0: return Wix;
			case 1: return Wih;
			case 2: return bi;
			case 3: return Wfx;
			case 4: return Wfh;
			case 5: return bf;
			case 6: return Wox;
			case 7: return Woh;
			case 8: return bo;
			case 9: return Wcx;
			case 10: return Wch;
			case 11: return bc;
			default: return bc;
		}
	}
	
	void Serialize(Stream& s) {
		s % Wix % Wih % bi % Wfx % Wfh % bf % Wox % Woh % bo % Wcx % Wch % bc;
	}
	
};

struct RNNModel : Moveable<RNNModel> {
	
	MatId Wxh, Whh, bhh;
	
	static int GetCount() {return 3;}
	MatId GetMat(int i) {
		ASSERT(i >= 0 && i < 3);
		switch (i) {
			case 0: return Wxh;
			case 1: return Whh;
			case 2: return bhh;
			default: return bhh;
		}
	}
	
	void Serialize(Stream& s) {
		s % Wxh % Whh % bhh;
	}
};


void Softmax(const Mat& m, Mat& out);

}

#endif
