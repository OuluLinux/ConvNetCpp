#ifndef _ConvNet_Reinforce_h_
#define _ConvNet_Reinforce_h_

#include "Session.h"

/*

	Reinforce.h is a C++ conversion of Reinforce.js of Andrej Karpathy.
	MIT License.

	Original source:	https://github.com/karpathy/reinforcejs
	LSTM backward:		https://gist.github.com/karpathy/587454dc0146a6ae21fc

*/

namespace ConvNet {

class ReinforceBase {
	
protected:
	ReinforceBase();
	
public:
	
	Volume *input, *input1, *input2;
	Volume output;
	
	virtual ~ReinforceBase();
	virtual Volume& Forward(Volume& input);
	virtual Volume& Forward(Volume& input1, Volume& input2);
	virtual void Backward() = 0;
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "base";}
	virtual int GetArgCount() const = 0;
	
	Volume& operator() (Volume& a) {return Forward(a);}
	Volume& operator() (Volume& a, Volume& b) {return Forward(a,b);}
	
};

class ReinforceRowPluck : public ReinforceBase {
	int ix;
	
public:
	ReinforceRowPluck();
	~ReinforceRowPluck();
	virtual Volume& Forward(Volume& input);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "RowPluck";}
	virtual int GetArgCount() const {return 1;}
	
	void SetRow(int i) {ix = i;}
	
};

class ReinforceTanh : public ReinforceBase {
	
public:
	ReinforceTanh();
	~ReinforceTanh();
	virtual Volume& Forward(Volume& input);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Tanh";}
	virtual int GetArgCount() const {return 1;}
	
};

class ReinforceSigmoid : public ReinforceBase {
	
public:
	ReinforceSigmoid();
	~ReinforceSigmoid();
	virtual Volume& Forward(Volume& input);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Sigmoid";}
	virtual int GetArgCount() const {return 1;}
	
};

class ReinforceRelu : public ReinforceBase {
	
public:
	ReinforceRelu();
	~ReinforceRelu();
	virtual Volume& Forward(Volume& input);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Relu";}
	virtual int GetArgCount() const {return 1;}
	
};

class ReinforceMul : public ReinforceBase {
	
public:
	ReinforceMul();
	~ReinforceMul();
	virtual Volume& Forward(Volume& input1, Volume& input2);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Mul";}
	virtual int GetArgCount() const {return 2;}
	
};

class ReinforceAdd : public ReinforceBase {
	
public:
	ReinforceAdd();
	~ReinforceAdd();
	virtual Volume& Forward(Volume& input1, Volume& input2);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Add";}
	virtual int GetArgCount() const {return 2;}
	
};

class ReinforceDot : public ReinforceBase {
	
public:
	ReinforceDot();
	~ReinforceDot();
	virtual Volume& Forward(Volume& input1, Volume& input2);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Dot";}
	virtual int GetArgCount() const {return 2;}
	
};

class ReinforceEltMul : public ReinforceBase {
	
public:
	ReinforceEltMul();
	~ReinforceEltMul();
	virtual Volume& Forward(Volume& input1, Volume& input2);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "EltMul";}
	virtual int GetArgCount() const {return 2;}
	
};

class Graph {
	
	
	Vector<ReinforceBase*> layers;
	Vector<Volume*> extra_args;
	
public:
	Graph();
	~Graph();
	
	void Clear();
	Volume& Forward(Volume& input);
	void Backward();
	
	void AddMul(Volume& multiplier);
	void AddAdd(Volume& addition);
	void AddTanh();
	
};









}

#endif
