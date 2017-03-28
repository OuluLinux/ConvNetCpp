#ifndef _ConvNet_Recurrent_h_
#define _ConvNet_Recurrent_h_

/*
	Recurrent.h is a C++ conversion of Recurrent.js of Andrej Karpathy.
	MIT License.

	Original source:	https://github.com/karpathy/reinforcejs (fork of recurrentjs)
*/

namespace ConvNet {

class RecurrentBase {
	
protected:
	RecurrentBase();
	RecurrentBase(Volume& input);
	RecurrentBase(Volume& input1, Volume& input2);
	
public:
	
	Volume *input1, *input2;
	Volume output;
	
	virtual ~RecurrentBase();
	virtual Volume& Forward() = 0;
	virtual void Backward() = 0;
	virtual String GetKey() const {return "base";}
	virtual int GetArgCount() const = 0;
	
	Volume& Forward(Volume& input);
	Volume& Forward(Volume& input1, Volume& input2);
	
	Volume& operator() (Volume& a) {return Forward(a);}
	Volume& operator() (Volume& a, Volume& b) {return Forward(a,b);}
	
};

class RecurrentRowPluck : public RecurrentBase {
	int* ix;
	
public:
	RecurrentRowPluck(int* i) {ix = i;}
	RecurrentRowPluck(int* i, Volume& in) : RecurrentBase(in) {ix = i;}
	~RecurrentRowPluck() {}
	virtual Volume& Forward();
	virtual void Backward();
	virtual String GetKey() const {return "RowPluck";}
	virtual int GetArgCount() const {return 0;}
	
};

class RecurrentTanh : public RecurrentBase {
	
public:
	RecurrentTanh() {}
	RecurrentTanh(Volume& in) : RecurrentBase(in) {};
	~RecurrentTanh() {}
	virtual Volume& Forward();
	virtual void Backward();
	virtual String GetKey() const {return "Tanh";}
	virtual int GetArgCount() const {return 1;}
	
};

class RecurrentSigmoid : public RecurrentBase {
	
public:
	RecurrentSigmoid() {}
	RecurrentSigmoid(Volume& in) : RecurrentBase(in) {};
	~RecurrentSigmoid() {}
	virtual Volume& Forward();
	virtual void Backward();
	virtual String GetKey() const {return "Sigmoid";}
	virtual int GetArgCount() const {return 1;}
	
};

class RecurrentRelu : public RecurrentBase {
	
public:
	RecurrentRelu() {}
	RecurrentRelu(Volume& in) : RecurrentBase(in) {};
	~RecurrentRelu() {}
	virtual Volume& Forward();
	virtual void Backward();
	virtual String GetKey() const {return "Relu";}
	virtual int GetArgCount() const {return 1;}
	
};

class RecurrentMul : public RecurrentBase {
	
public:
	RecurrentMul() {}
	RecurrentMul(Volume& in1, Volume& in2) : RecurrentBase(in1, in2) {};
	~RecurrentMul() {}
	virtual Volume& Forward();
	virtual void Backward();
	virtual String GetKey() const {return "Mul";}
	virtual int GetArgCount() const {return 2;}
	
};

class RecurrentAdd : public RecurrentBase {
	
public:
	RecurrentAdd() {}
	RecurrentAdd(Volume& in1, Volume& in2) : RecurrentBase(in1, in2) {};
	~RecurrentAdd() {}
	virtual Volume& Forward();
	virtual void Backward();
	virtual String GetKey() const {return "Add";}
	virtual int GetArgCount() const {return 2;}
	
};

class RecurrentDot : public RecurrentBase {
	
public:
	RecurrentDot() {}
	RecurrentDot(Volume& in1, Volume& in2) : RecurrentBase(in1, in2) {};
	~RecurrentDot() {}
	virtual Volume& Forward();
	virtual void Backward();
	virtual String GetKey() const {return "Dot";}
	virtual int GetArgCount() const {return 2;}
	
};

class RecurrentEltMul : public RecurrentBase {
	
public:
	RecurrentEltMul() {}
	RecurrentEltMul(Volume& in1, Volume& in2) : RecurrentBase(in1, in2) {};
	~RecurrentEltMul() {}
	virtual Volume& Forward();
	virtual void Backward();
	virtual String GetKey() const {return "EltMul";}
	virtual int GetArgCount() const {return 2;}
	
};

class RecurrentCopy : public RecurrentBase {
	
public:
	RecurrentCopy(Volume& in1, Volume& in2) : RecurrentBase(in1, in2) {};
	~RecurrentCopy() {}
	virtual Volume& Forward();
	virtual void Backward();
	virtual String GetKey() const {return "Copy";}
	virtual int GetArgCount() const {return 2;}
	
};

class RecurrentAddConst : public RecurrentBase {
	double d;
	
public:
	RecurrentAddConst(double d, Volume& in) : RecurrentBase(in), d(d) {};
	~RecurrentAddConst() {}
	virtual Volume& Forward();
	virtual void Backward();
	virtual String GetKey() const {return "AddConst";}
	virtual int GetArgCount() const {return 1;}
	
};

class RecurrentMulConst : public RecurrentBase {
	double d;
	
public:
	RecurrentMulConst(double d, Volume& in) : RecurrentBase(in), d(d) {};
	~RecurrentMulConst() {}
	virtual Volume& Forward();
	virtual void Backward();
	virtual String GetKey() const {return "AddConst";}
	virtual int GetArgCount() const {return 1;}
	
};






// Graph follows Net class, and allow easy pipeline creation.
// This is not very useful in complex problems.
class Graph {
	Vector<RecurrentBase*> layers;
	Vector<Volume*> extra_args;
	
public:
	Graph();
	~Graph();
	
	void Clear();
	Volume& Forward(Volume& input);
	void Backward();
	
	Volume& RowPluck(int* row);
	Volume& Tanh();
	Volume& Sigmoid();
	Volume& Relu();
	Volume& Mul(Volume& multiplier);
	Volume& Add(Volume& addition);
	Volume& Dot(Volume& v);
	Volume& EltMul(Volume& v);
	
	RecurrentBase& GetLayer(int i) {return *layers[i];}
	int GetCount() const {return layers.GetCount();}
	
};


// GraphTree allows custom connections between layers.
class GraphTree {
	Vector<RecurrentBase*> layers;
	
public:
	GraphTree();
	~GraphTree();
	
	void Clear();
	Volume& Forward();
	void Backward();
	
	Volume& RowPluck(int* row, Volume& in);
	Volume& Tanh(Volume& in);
	Volume& Sigmoid(Volume& in);
	Volume& Relu(Volume& in);
	Volume& Mul(Volume& in1, Volume& in2);
	Volume& Add(Volume& in1, Volume& in2);
	Volume& Dot(Volume& in1, Volume& in2);
	Volume& EltMul(Volume& in1, Volume& in2);
	Volume& Copy(Volume& src, Volume& dst);
	Volume& AddConstant(double d, Volume& in);
	Volume& MulConstant(double d, Volume& in);
	
	RecurrentBase& GetLayer(int i) {return *layers[i];}
	int GetCount() const {return layers.GetCount();}
	
	RecurrentBase& Top() {return *layers.Top();}
	
};




struct HighwayModel : Moveable<HighwayModel> {
	
	Volume noise_i[2];
	Volume noise_h[2];
	Volume Wix, Wih;
	
	static int GetCount() {return 6;}
	
	Volume& GetVolume(int i) {
		ASSERT(i >= 0 && i < 6);
		switch (i) {
			case 0: return noise_i[0];
			case 1: return noise_i[1];
			case 2: return noise_h[0];
			case 3: return noise_h[1];
			case 4: return Wix;
			case 5: return Wih;
			default: return Wih;
		}
	}
};

struct LSTMModel : Moveable<LSTMModel> {
	
	Volume Wix, Wih, bi, Wfx, Wfh, bf, Wox, Woh, bo, Wcx, Wch, bc;
	
	static int GetCount() {return 12;}
	Volume& GetVolume(int i) {
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
};

struct RNNModel : Moveable<RNNModel> {
	
	Volume Wxh, Whh, bhh;
	
	static int GetCount() {return 3;}
	Volume& GetVolume(int i) {
		ASSERT(i >= 0 && i < 3);
		switch (i) {
			case 0: return Wxh;
			case 1: return Whh;
			case 2: return bhh;
			default: return bhh;
		}
	}
};


void Softmax(const Volume& m, Volume& out);

}

#endif
