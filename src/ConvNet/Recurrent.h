#ifndef _ConvNet_Recurrent_h_
#define _ConvNet_Recurrent_h_

namespace ConvNet {



struct LSTMModel : Moveable<LSTMModel> {
	
	// LSTM
	Volume Wix, Wih, bi, Wfx, Wfh, bf, Wox, Woh, bo, Wcx, Wch, bc;
	
	// RNN
	Volume Wxh, Whh, bhh;
	
	LSTMModel() {
		
	}
	
	LSTMModel& operator=(const LSTMModel& m) {
		Panic("TODO");
		return *this;
	}
	
};

struct ModelVector : Moveable<ModelVector> {
	Vector<LSTMModel> model;
	Volume Whd, bd, Wil;
	
};

struct CellMemory {
	Vector<Volume> h;
	Vector<Volume> c;
	Volume o;
};





class LSTM {
	
	ReinforceTanh		tanh;
	ReinforceSigmoid	sigmoid;
	ReinforceRelu		relu;
	ReinforceMul		mul;
	ReinforceAdd		add;
	ReinforceDot		dot;
	ReinforceEltMul		eltmul;
	
public:
	LSTM();
	ModelVector Init(int input_size, Vector<int>& hidden_sizes, int output_size);
	CellMemory Forward(const Graph& G, ModelVector& vec, const Vector<int>& hidden_sizes, Volume& x, CellMemory* prev=NULL);
};

class RNN {
	
	ReinforceMul		mul;
	ReinforceAdd		add1, add2;
	ReinforceRelu		relu;
	
public:
	RNN();
	ModelVector Init(int input_size, Vector<int>& hidden_sizes, int output_size);
	CellMemory Forward(const Graph& G, ModelVector& vec, const Vector<int>& hidden_sizes, Volume& x, CellMemory* prev=NULL);
	
};







struct SolverStat {
	double ratio_clipped;
};

class Solver {
	
	double decay_rate;
	double smooth_eps;
	Vector<Volume> step_cache;
	
public:
	Solver();
	SolverStat Step(Vector<Volume>& model, int step_size, int regc, int clipval);
};


Volume Softmax(const Volume& m);

}

#endif
