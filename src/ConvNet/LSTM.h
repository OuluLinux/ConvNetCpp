#ifndef _ConvNet_LSTM_h_
#define _ConvNet_LSTM_h_

namespace ConvNet {



struct LSTMModel : Moveable<LSTMModel> {
	
	Volume Wix, Wih, bi, Wfx, Wfh, bf, Wox, Woh, bo, Wcx, Wch, bc;
	
	LSTMModel() {
		
	}
	
};

struct ModelVector : Moveable<ModelVector> {
	Vector<LSTMModel> model;
	Volume Whd, bd;
	
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
	ModelVector InitLSTM(int input_size, Vector<int>& hidden_sizes, int output_size);
	CellMemory ForwardLSTM(const Graph& G, ModelVector& vec, const Vector<int>& hidden_sizes, Volume& x/*, prev*/);
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

}

#endif
