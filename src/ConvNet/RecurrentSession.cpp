#include "ConvNet.h"

namespace ConvNet {

RecurrentSession::RecurrentSession() {
	mode = MODE_RNN;
	learning_rate = 0.01;
	clipval = 5.0;
	regc = 0.000001;
	input_size = -1;
	output_size = -1;
	letter_size = -1;
	max_graphs = 100;
	
	// Solver
	decay_rate = 0.999;
	smooth_eps = 1e-8;
	
	index_sequence.SetCount(max_graphs, 0);
	graphs.SetCount(max_graphs);
	hidden_prevs.SetCount(max_graphs+1);
	cell_prevs.SetCount(max_graphs+1);
}

RecurrentSession::~RecurrentSession() {
	
}

int RecurrentSession::GetVolumeCount() {
	int count = 0;
	if (mode == MODE_RNN)
		count = rnn_model.GetCount() * RNNModel::GetCount();
	else if (mode == MODE_LSTM)
		count = lstm_model.GetCount() * LSTMModel::GetCount();
	else
		Panic("Invalid mode");
	count += 3;
	return count;
}

Volume& RecurrentSession::GetVolume(int i) {
	int count = 0, cols = 0, rows = 0;
	if (mode == MODE_RNN) {
		rows = rnn_model.GetCount();
		cols = RNNModel::GetCount();
	}
	else if (mode == MODE_LSTM) {
		rows = lstm_model.GetCount();
		cols = LSTMModel::GetCount();
	}
	else Panic("Invalid mode");
	
	int row = i / cols;
	if (row >= rows) {
		i = i - rows * cols;
		switch (i) {
			case 0: return Whd;
			case 1: return bd;
			case 2: return Wil;
			default: Panic("Invalid id " + IntStr(i));
		}
	} else {
		int col = i % cols;
		if (mode == MODE_RNN) {
			return rnn_model[row].GetVolume(col);
		}
		else if (mode == MODE_LSTM) {
			return lstm_model[row].GetVolume(col);
		}
		else Panic("Invalid mode");
	}
	
	return Wil;
}

void RecurrentSession::Init() {
	ASSERT_(input_size != -1, "Input size must be set");
	ASSERT_(output_size != -1, "Output size must be set");
	
	Wil = RandVolume(input_size, letter_size, 0, 0.08);
	
	if (mode == MODE_RNN) {
		InitRNN();
	}
	else if (mode == MODE_LSTM) {
		InitLSTM();
	}
	else Panic("Invalid RecurrentSession mode");
	
	InitGraphs();
}

void RecurrentSession::InitGraphs() {
	ASSERT(mode == MODE_RNN || mode == MODE_LSTM);
	
	step_cache.Clear();
	int hidden_count = hidden_sizes.GetCount();
	ASSERT_(hidden_count > 0, "Hidden sizes must be set");
	
	for (int i = 0; i < hidden_prevs.GetCount(); i++) {
		Vector<Volume>& hidden_prevs = this->hidden_prevs[i];
		hidden_prevs.SetCount(hidden_count);
		for (int d = 0; d < hidden_count; d++) {
			hidden_prevs[d].Init(1, hidden_sizes[d], 1, 0);
		}
		
		Vector<Volume>& cell_prevs = this->cell_prevs[i];
		cell_prevs.SetCount(hidden_count);
		for (int d = 0; d < hidden_count; d++) {
			cell_prevs[d].Init(1, hidden_sizes[d], 1, 0);
		}
	}
	
	for (int i = 0; i < graphs.GetCount(); i++) {
		Array<GraphTree>& hidden_graphs = graphs[i];
		hidden_graphs.SetCount(hidden_count);
		for (int j = 0; j < hidden_count; j++) {
			if (mode == MODE_RNN)
				InitRNN(i, j, hidden_graphs[j]);
			else
				InitLSTM(i, j, hidden_graphs[j]);
		}
	}
}

void RecurrentSession::InitRNN() {
	int hidden_size = 0;
	
	// loop over depths
	rnn_model.SetCount(hidden_sizes.GetCount());
	for (int d = 0; d < hidden_sizes.GetCount(); d++) {
		int prev_size = d == 0 ? letter_size : hidden_sizes[d - 1];
		hidden_size = hidden_sizes[d];
		RNNModel& m = rnn_model[d];
		m.Wxh = RandVolume(hidden_size, prev_size,		0, 0.08);
		m.Whh = RandVolume(hidden_size, hidden_size,	0, 0.08);
		m.bhh.Init(1, hidden_size, 1, 0);
	}
	
	// decoder params
	Whd = RandVolume(output_size, hidden_size, 0, 0.08);
	bd.Init(1, output_size, 1, 0);
}

void RecurrentSession::InitRNN(int i, int j, GraphTree& g) {
	RNNModel& m = rnn_model[j];
	
	g.Clear();
	
	Vector<Volume>& hidden_prevs = this->hidden_prevs[i];
	Vector<Volume>& hidden_nexts = this->hidden_prevs[i+1];
	
	if (j == 0) {
		input = &g.AddRowPluck(&index_sequence[i], Wil);
	}
	
	Volume& input_vector = j == 0 ? *input : hidden_nexts[j-1];
	Volume& hidden_prev = hidden_prevs[j];
	
	Volume& h0 = g.AddMul(m.Wxh, input_vector);
	Volume& h1 = g.AddMul(m.Whh, hidden_prev);
	Volume& hidden_d = g.AddRelu(g.AddAdd(g.AddAdd(h0, h1), m.bhh));
	
	g.AddCopy(hidden_d, hidden_nexts[j]);
	
	// one decoder to outputs at end
	if (j == hidden_prevs.GetCount() - 1) {
		g.AddAdd(g.AddMul(Whd, hidden_nexts[j]), bd);
	}
}

// hidden size should be a list
void RecurrentSession::InitLSTM() {
	int hidden_size = 0;
	
	// loop over depths
	lstm_model.SetCount(hidden_sizes.GetCount());
	for (int d = 0; d < hidden_sizes.GetCount(); d++) {
		// loop over depths
		LSTMModel& m = lstm_model[d];
		
		int prev_size = d == 0 ? letter_size : hidden_sizes[d - 1];
		hidden_size = hidden_sizes[d];
		
		// gates parameters
		m.Wix	= RandVolume(hidden_size, prev_size,	0, 0.08);
		m.Wih	= RandVolume(hidden_size, hidden_size,	0, 0.08);
		m.bi	.Init(1, hidden_size, 1, 0);
		m.Wfx	= RandVolume(hidden_size, prev_size,	0, 0.08);
		m.Wfh	= RandVolume(hidden_size, hidden_size,	0, 0.08);
		m.bf	.Init(1, hidden_size, 1, 0);
		m.Wox	= RandVolume(hidden_size, prev_size,	0, 0.08);
		m.Woh	= RandVolume(hidden_size, hidden_size,	0, 0.08);
		m.bo	.Init(1, hidden_size, 1, 0);
		
		// cell write params
		m.Wcx	= RandVolume(hidden_size, prev_size,	0, 0.08);
		m.Wch	= RandVolume(hidden_size, hidden_size,	0, 0.08);
		m.bc	.Init(1, hidden_size, 1, 0);
	}
	
	// decoder params
	Whd	= RandVolume(output_size, hidden_size, 0, 0.08);
	bd.Init(1, output_size, 1, 0);
}

void RecurrentSession::InitLSTM(int i, int j, GraphTree& g) {
	LSTMModel& m = lstm_model[j];
	
	g.Clear();
	
	// Graph uses hidden_prevs[j-1] so two hidden previous vectors are needed.
	// Otherwise the result could be written over hidden_prevs[j] immediately.
	Vector<Volume>& hidden_prevs = this->hidden_prevs[i];
	Vector<Volume>& hidden_nexts = this->hidden_prevs[i+1];
	Vector<Volume>& cell_prevs = this->cell_prevs[i];
	Vector<Volume>& cell_nexts = this->cell_prevs[i+1];
	
	if (j == 0) {
		input = &g.AddRowPluck(&index_sequence[i], Wil);
	}
	
	Volume& input_vector = j == 0 ? *input : hidden_nexts[j-1];
	Volume& hidden_prev = hidden_prevs[j];
	Volume& cell_prev = cell_prevs[j];
	
	// input gate
	Volume& h0 = g.AddMul(m.Wix, input_vector);
	Volume& h1 = g.AddMul(m.Wih, hidden_prev);
	Volume& input_gate = g.AddSigmoid(g.AddAdd(g.AddAdd(h0, h1), m.bi));
	
	// forget gate
	Volume& h2 = g.AddMul(m.Wfx, input_vector);
	Volume& h3 = g.AddMul(m.Wfh, hidden_prev);
	Volume& forget_gate = g.AddSigmoid(g.AddAdd(g.AddAdd(h2, h3), m.bf));
	
	// output gate
	Volume& h4 = g.AddMul(m.Wox, input_vector);
	Volume& h5 = g.AddMul(m.Woh, hidden_prev);
	Volume& output_gate = g.AddSigmoid(g.AddAdd(g.AddAdd(h4, h5), m.bo));
	
	// write operation on cells
	Volume& h6 = g.AddMul(m.Wcx, input_vector);
	Volume& h7 = g.AddMul(m.Wch, hidden_prev);
	Volume& cell_write = g.AddTanh(g.AddAdd(g.AddAdd(h6, h7), m.bc));
	
	// compute new cell activation
	Volume& retain_cell = g.AddEltMul(forget_gate, cell_prev); // what do we keep from cell
	Volume& write_cell = g.AddEltMul(input_gate, cell_write); // what do we write to cell
	Volume& cell_d = g.AddAdd(retain_cell, write_cell); // new cell contents
	
	// compute hidden state as gated, saturated cell activations
	Volume& hidden_d = g.AddEltMul(output_gate, g.AddTanh(cell_d));
	
	g.AddCopy(hidden_d,	hidden_nexts[j]);
	g.AddCopy(cell_d,	cell_nexts[j]);
	
	
	// one decoder to outputs at end
	if (j == hidden_prevs.GetCount() - 1) {
		g.AddAdd(g.AddMul(Whd, hidden_nexts[j]), bd);
	}
}

void RecurrentSession::Learn(const Vector<int>& input_sequence) {
	double log2ppl = 0.0;
	double cost = 0.0;
	
	ASSERT(input_sequence.GetCount() < graphs.GetCount());
	
	// Copy input sequence. Fixed index_sequence addresses are used in RowPluck.
	int n = input_sequence.GetCount();
	
	// start and end tokens are zeros
	index_sequence[0] = 0; // first step: start with START token
	for(int i = 0; i < n; i++)
		index_sequence[i+1] = input_sequence[i]; // this value is used in the RowPluck
	for(int i = n+1; i < index_sequence.GetCount(); i++)
		index_sequence[i] = -1; // for debugging
	
	ResetPrevs();
	
	for(int i = 0; i <= n; i++) {
		int ix_target = i == n ? 0 : index_sequence[i+1]; // last step: end with END token
		
		Array<GraphTree>& list = graphs[i];
		for(int j = 0; j < list.GetCount(); j++) {
			list[j].Forward();
		}
		
		Volume& logprobs = list.Top().Top().output;
		Volume probs = Softmax(logprobs); // compute the softmax probabilities
		
		log2ppl += -log2(probs.Get(ix_target)); // accumulate base 2 log prob and do smoothing
		cost += -log(probs.Get(ix_target));
		
		// write gradients into log probabilities
		int count = logprobs.GetLength();
		for(int j = 0; j < count; j++)
			logprobs.SetGradient(j, probs.Get(j));
		logprobs.AddGradient(ix_target, -1.0);
	}
	
	ppl = pow(2, log2ppl / (n - 1));
	cost = cost;
	
	Backward(n);
	
	SolverStep();
}

void RecurrentSession::Backward(int seq_end_cursor) {
	for (int i = seq_end_cursor; i >= 0; i--) {
		Array<GraphTree>& list = graphs[i];
		for (int j = list.GetCount()-1; j >= 0; j--) {
			list[j].Backward();
		}
	}
}

void RecurrentSession::SolverStep() {
	// perform parameter update
	int num_clipped = 0;
	int num_tot = 0;
	int n = GetVolumeCount();
	
	int step_cache_count = step_cache.GetCount();
	step_cache.SetCount(n);
	
	for (int k = 0; k < n; k++) {
		Volume& m = GetVolume(k);
		Volume& s = step_cache[k];
		
		if (k >= step_cache_count) {
			s.Init(m.GetWidth(), m.GetHeight(), m.GetDepth(), 0);
		}
		
		for (int i = 0; i < m.GetLength(); i++) {
			// rmsprop adaptive learning rate
			double mdwi = m.GetGradient(i);
			s.Set(i, s.Get(i) * decay_rate + (1.0 - decay_rate) * mdwi * mdwi);
			
			// gradient clip
			if (mdwi > +clipval) {
				mdwi = +clipval;
				num_clipped++;
			}
			else if (mdwi < -clipval) {
				mdwi = -clipval;
				num_clipped++;
			}
			
			num_tot++;
			
			// update (and regularize)
			m.Add(i, - learning_rate * mdwi / sqrt(s.Get(i) + smooth_eps) - regc * m.Get(i));
			m.SetGradient(i, 0); // reset gradients for next iteration
		}
	}
	ratio_clipped = num_clipped * 1.0 / num_tot;
}

void RecurrentSession::ResetPrevs() {
	int hidden_count = hidden_sizes.GetCount();
	
	Vector<Volume>& hidden_prevs = this->hidden_prevs[0];
	hidden_prevs.SetCount(hidden_count);
	for (int d = 0; d < hidden_count; d++) {
		hidden_prevs[d].Init(1, hidden_sizes[d], 1, 0);
	}
	
	Vector<Volume>& cell_prevs = this->cell_prevs[0];
	cell_prevs.SetCount(hidden_count);
	for (int d = 0; d < hidden_count; d++) {
		cell_prevs[d].Init(1, hidden_sizes[d], 1, 0);
	}
}

void RecurrentSession::Predict(Vector<int>& output_sequence, bool samplei, double temperature) {
	output_sequence.SetCount(0);
	
	index_sequence[0] = 0;
	for(int i = 1; i < index_sequence.GetCount(); i++)
		index_sequence[i] = -1; // for debugging
	
	ResetPrevs();
	
	for (int i = 0; ; i++) {
		Array<GraphTree>& list = graphs[i];
		for(int j = 0; j < list.GetCount(); j++) {
			GraphTree& g = list[j];
			g.Forward();
		}
		
		// sample predicted letter
		Volume& logprobs = list.Top().Top().output;
		
		if (temperature != 1.0 && samplei) {
			// scale log probabilities by temperature and renormalize
			// if temperature is high, logprobs will go towards zero
			// and the softmax outputs will be more diffuse. if temperature is
			// very low, the softmax outputs will be more peaky
			for (int q = 0; q < logprobs.GetLength(); q++) {
				logprobs.Set(q, logprobs.Get(q) / temperature);
			}
		}
		
		Volume probs = Softmax(logprobs);
		
		int ix = 0;
		if (samplei) {
			ix = probs.GetSampledColumn();
		} else {
			ix = probs.GetMaxColumn();
		}
		
		if (ix == 0) break; // END token predicted, break out
		if (i+1 >= max_graphs) break; // something is wrong
		
		output_sequence.Add(ix);
		
		// Set index to variable what RowPluck reads
		index_sequence[i+1] = ix;
	}
}

void RecurrentSession::Load(const ValueMap& js) {
	#define LOAD(x) if (js.Find(#x) != -1) {x = js.GetValue(js.Find(#x));}
	
	String generator;
	LOAD(generator);
	mode = generator == "lstm" ? MODE_LSTM : MODE_RNN;
	
	if (js.Find("hidden_sizes") != -1) {
		hidden_sizes.Clear();
		ValueMap hs = js.GetValue(js.Find("hidden_sizes"));
		for(int i = 0; i < hs.GetCount(); i++)
			hidden_sizes.Add(hs[i]);
	}
	
	LOAD(letter_size);
	LOAD(regc);
	LOAD(learning_rate);
	LOAD(clipval);
	
	if (js.Find("model") != -1) {
		ValueMap model = js.GetValue(js.Find("model"));
		
		#define LOADVOL(x) {ValueMap map = model.GetValue(model.Find(#x)); this->x.Load(map);}
		LOADVOL(Wil);
		LOADVOL(Whd);
		LOADVOL(bd);
		#undef LOADVOL
		
		if      (mode == MODE_LSTM) {lstm_model.SetCount(hidden_sizes.GetCount());}
		else if (mode == MODE_RNN)  {rnn_model.SetCount(hidden_sizes.GetCount());}
		else Panic("Invalid mode");
		
		for(int i = 0; i < hidden_sizes.GetCount(); i++) {
			if (mode == MODE_LSTM) {
				#define LOADMODVOL(x) {ValueMap map = model.GetValue(model.Find(#x + IntStr(i))); lstm_model[i].x.Load(map);}
				LOADMODVOL(Wix);
				LOADMODVOL(Wih);
				LOADMODVOL(bi);
				LOADMODVOL(Wfx);
				LOADMODVOL(Wfh);
				LOADMODVOL(bf);
				LOADMODVOL(Wox);
				LOADMODVOL(Woh);
				LOADMODVOL(bo);
				LOADMODVOL(Wcx);
				LOADMODVOL(Wch);
				LOADMODVOL(bc);
				#undef LOADMODVOL
			}
			else {
				#define LOADMODVOL(x) {ValueMap map = model.GetValue(model.Find(#x + IntStr(i))); rnn_model[i].x.Load(map);}
				LOADMODVOL(Wxh);
				LOADMODVOL(Whh);
				LOADMODVOL(bhh);
				#undef LOADMODVOL
			}
		}
	}
	#undef LOAD
}

void RecurrentSession::Store(ValueMap& js) {
	#define SAVE(x) js.GetAdd(#x) = x;
	
	String generator = mode == MODE_LSTM ? "lstm" : "rnn";
	SAVE(generator);
	
	ValueMap hs;
	for(int i = 0; i < hidden_sizes.GetCount(); i++)
		hs.Add(IntStr(i), hidden_sizes[i]);
	
	SAVE(letter_size);
	SAVE(regc);
	SAVE(learning_rate);
	SAVE(clipval);
	
	ValueMap model;
	
	#define SAVEVOL(x) {ValueMap map; this->x.Store(map); model.GetAdd(#x) = map;}
	SAVEVOL(Wil);
	SAVEVOL(Whd);
	SAVEVOL(bd);
	#undef SAVEVOL
	
	for(int i = 0; i < hidden_sizes.GetCount(); i++) {
		if (mode == MODE_LSTM) {
			#define SAVEMODVOL(x) {ValueMap map; lstm_model[i].x.Store(map); model.GetAdd(#x + IntStr(i)) = map;}
			SAVEMODVOL(Wix);
			SAVEMODVOL(Wih);
			SAVEMODVOL(bi);
			SAVEMODVOL(Wfx);
			SAVEMODVOL(Wfh);
			SAVEMODVOL(bf);
			SAVEMODVOL(Wox);
			SAVEMODVOL(Woh);
			SAVEMODVOL(bo);
			SAVEMODVOL(Wcx);
			SAVEMODVOL(Wch);
			SAVEMODVOL(bc);
			#undef SAVEMODVOL
		}
		else {
			#define SAVEMODVOL(x) {ValueMap map; rnn_model[i].x.Store(map); model.GetAdd(#x + IntStr(i)) = map;}
			SAVEMODVOL(Wxh);
			SAVEMODVOL(Whh);
			SAVEMODVOL(bhh);
			#undef SAVEMODVOL
		}
	}
	
	js.Add("model", model);
}

}
