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
	
	
public:
	
	Volume *input, *input1, *input2;
	Volume output;
	
	ReinforceBase();
	virtual ~ReinforceBase();
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual Volume& Forward(Volume& input1, Volume& input2, bool is_training = false);
	virtual void Backward() = 0;
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "base";}
	
	Volume& operator() (Volume& a) {return Forward(a);}
	Volume& operator() (Volume& a, Volume& b) {return Forward(a,b);}
	
};

class ReinforceRowPluck : public ReinforceBase {
	int ix;
	
public:
	ReinforceRowPluck();
	~ReinforceRowPluck();
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "RowPluck";}
	
	void SetRow(int i) {ix = i;}
	
};

class ReinforceTanh : public ReinforceBase {
	
public:
	ReinforceTanh();
	~ReinforceTanh();
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Tanh";}
	
};

class ReinforceSigmoid : public ReinforceBase {
	
public:
	ReinforceSigmoid();
	~ReinforceSigmoid();
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Sigmoid";}
	
};

class ReinforceRelu : public ReinforceBase {
	
public:
	ReinforceRelu();
	~ReinforceRelu();
	virtual Volume& Forward(Volume& input, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Relu";}
	
};

class ReinforceMul : public ReinforceBase {
	
public:
	ReinforceMul();
	~ReinforceMul();
	virtual Volume& Forward(Volume& input1, Volume& input2, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Mul";}
	
};

class ReinforceAdd : public ReinforceBase {
	
public:
	ReinforceAdd();
	~ReinforceAdd();
	virtual Volume& Forward(Volume& input1, Volume& input2, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Add";}
	
};

class ReinforceDot : public ReinforceBase {
	
public:
	ReinforceDot();
	~ReinforceDot();
	virtual Volume& Forward(Volume& input1, Volume& input2, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "Dot";}
	
};

class ReinforceEltMul : public ReinforceBase {
	
public:
	ReinforceEltMul();
	~ReinforceEltMul();
	virtual Volume& Forward(Volume& input1, Volume& input2, bool is_training = false);
	virtual void Backward();
	virtual void Init(int input_width, int input_height, int input_depth);
	virtual String GetKey() const {return "EltMul";}
	
};

class Graph {
	
	
	Vector<ReinforceBase*> layers;
	
public:
	Graph() {
		
	}
	
	void Backward() {
		
	}
	
};









}

#endif
