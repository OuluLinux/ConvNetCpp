#ifndef _ConvNet_RecurrentSession_h_
#define _ConvNet_RecurrentSession_h_

#include "Recurrent.h"

namespace ConvNet {

#define M_LOG2E 1.44269504088896340736 //log2(e)

inline long double log2(const long double x){
    return  log(x) * M_LOG2E;
}

class RecurrentSession {
	
	
protected:
	
	// Vector
	Array<Array<GraphTree> > graphs;
	
	// ModelVector
	Vector<LSTMModel> lstm_model;
	Vector<RNNModel> rnn_model;
	Volume Whd, bd, Wil;
	
	// Solver
	Vector<Volume> step_cache;
	double decay_rate;
	double smooth_eps;
	
	// Session vars
	Vector<Vector<Volume> > hidden_prevs;
	Vector<Vector<Volume> > cell_prevs;
	Vector<int> hidden_sizes;
	Array<int> index_sequence; // Array instead of vector to allow resizing
	Volume* input;
	double ppl, cost;
	double regc;
	double learning_rate;
	double clipval;
	double ratio_clipped;
	int mode;
	int input_size;
	int output_size;
	int letter_size;
	int max_graphs;
	
	enum {MODE_RNN, MODE_LSTM};
	
	void InitRNN();
	void InitRNN(int i, int j, GraphTree& g);
	void InitLSTM();
	void InitLSTM(int i, int j, GraphTree& g);
	void Backward(int seq_end_cursor);
	void SolverStep();
	void ResetPrevs();
	int GetVolumeCount();
	Volume& GetVolume(int i);
public:
	typedef RecurrentSession CLASSNAME;
	RecurrentSession();
	~RecurrentSession();
	
	void Init();
	void InitGraphs();
	void Learn(const Vector<int>& index_sequence);
	void Predict(Vector<int>& index_sequence, bool samplei=false, double temperature=1.0);
	void Load(const ValueMap& js);
	void Store(ValueMap& js);
	
	double GetPerplexity() const {return ppl;}
	double GetCost() const {return cost;}
	double GetLearningRate() const {return learning_rate;}
	
	void SetInputSize(int i) {input_size = i;}
	void SetOutputSize(int i) {output_size = i;}
	void SetLearningRate(double d) {learning_rate = d;}
	
};

}

#endif
