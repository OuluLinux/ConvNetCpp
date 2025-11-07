#ifndef _ConvNet_RecurrentSession_h_
#define _ConvNet_RecurrentSession_h_

#include "Recurrent.h"

namespace ConvNet {

#ifndef M_LOG2E
#define M_LOG2E 1.44269504088896340736 //log2(e)
#endif

inline long double log2(const long double x){
    return  log(x) * M_LOG2E;
}

class RecurrentSession : public MatPool {
	
	
protected:
	friend class RecurrentBase;
	
	// Vector
	Array<Array<GraphTree> > graphs;
	
	// ModelVector
	Vector<HighwayModel> hw_model;
	Vector<LSTMModel> lstm_model;
	Vector<RNNModel> rnn_model;
	MatId Whd, bd, Wil;
	MatId noise_i[2];
	
	// Solver
	Vector<Mat> step_cache;
	double decay_rate;
	double smooth_eps;
	
	// Session vars
	Vector<MatId> first_hidden, first_cell;
	Vector<Vector<MatId> > hidden_prevs;
	Vector<Vector<MatId> > cell_prevs;
	Vector<int> hidden_sizes;
	MatId input;
	Mat probs;
	double ppl, cost;
	double regc;
	double learning_rate;
	double clipval;
	double ratio_clipped;
	double initial_bias;
	int mode;
	int input_size;
	int output_size;
	int letter_size;
	int max_graphs;
	
	enum {MODE_RNN, MODE_LSTM, MODE_HIGHWAY};
	
	void InitRNN();
	void InitRNN(int i, int j, GraphTree& g);
	void InitLSTM();
	void InitLSTM(int i, int j, GraphTree& g);
	void InitHighway();
	void InitHighway(int i, int j, GraphTree& g);
	void Backward(int seq_end_cursor);
	void SolverStep();
	void ResetPrevs();
	
	
public:
	typedef RecurrentSession CLASSNAME;
	RecurrentSession();
	~RecurrentSession();
	
	void Init();
	void InitGraphs();
	void Learn(const Vector<int>& index_sequence);
	void Predict(Vector<int>& index_sequence, bool samplei=false, double temperature=1.0, bool continue_sentence=false, int max_predictions=-1);
	void Load(const ValueMap& js);
	void Store(ValueMap& js);
	void Serialize(Stream& s);
	
	double GetPerplexity() const {return ppl;}
	double GetCost() const {return cost;}
	double GetLearningRate() const {return learning_rate;}
	int GetMatCount();
	MatId GetMat(int i);
	int GetGraphCount() {return graphs.GetCount();}
	
	void SetInputSize(int i) {input_size = i;}
	void SetOutputSize(int i) {output_size = i;}
	void SetLearningRate(double d) {learning_rate = d;}
	
};

}

#endif
