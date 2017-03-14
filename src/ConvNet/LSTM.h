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
	
	ReinforceRowPluck	rowpluck;
	ReinforceTanh		tanh;
	ReinforceSigmoid	sigmoid;
	ReinforceRelu		relu;
	ReinforceMul		mul;
	ReinforceAdd		add;
	ReinforceDot		dot;
	ReinforceEltMul		eltmul;
	
public:
	LSTM() {
		
		
	}
	
	
	ModelVector InitLSTM(int input_size, Vector<int>& hidden_sizes, int output_size) {
		// hidden size should be a list
		
		ModelVector vec;
		vec.model.SetCount(hidden_sizes.GetCount());
		
		int hidden_size = 0;
		for (int d = 0; d < hidden_sizes.GetCount(); d++) { // loop over depths
			LSTMModel& m = vec.model[d];
			
			int prev_size = d == 0 ? input_size : hidden_sizes[d - 1];
			hidden_size = hidden_sizes[d];
			
			// gates parametersforward
			m.Wix	= RandVolume(hidden_size, prev_size , 0, 0.08);
			m.Wih	= RandVolume(hidden_size, hidden_size , 0, 0.08);
			m.bi	= Volume(hidden_size, 1);
			m.Wfx	= RandVolume(hidden_size, prev_size , 0, 0.08);
			m.Wfh	= RandVolume(hidden_size, hidden_size , 0, 0.08);
			m.bf	= Volume(hidden_size, 1);
			m.Wox	= RandVolume(hidden_size, prev_size , 0, 0.08);
			m.Woh	= RandVolume(hidden_size, hidden_size , 0, 0.08);
			m.bo	= Volume(hidden_size, 1);
			
			// cell write params
			m.Wcx	= RandVolume(hidden_size, prev_size , 0, 0.08);
			m.Wch	= RandVolume(hidden_size, hidden_size , 0, 0.08);
			m.bc	= Volume(hidden_size, 1);
		}
		
		// decoder params
		vec.Whd	= RandVolume(output_size, hidden_size, 0, 0.08);
		vec.bd	= Volume(output_size, 1);
		
		return vec;
	}
	
	CellMemory ForwardLSTM(const Graph& G, ModelVector& vec, const Vector<int>& hidden_sizes, Volume& x/*, prev*/) {
		// forward prop for a single tick of LSTM
		// G is graph to append ops to
		// model contains LSTM parameters
		// x is 1D column vector with observation
		// prev is a struct containing hidden and cell
		// from previous iteration
		
		Vector<Volume> hidden_prevs;
		Vector<Volume> cell_prevs;
		
		hidden_prevs.SetCount(hidden_sizes.GetCount());
		cell_prevs.SetCount(hidden_sizes.GetCount());
		
		/*if (prev == null || typeof prev.h == 'undefined') {*/
		for (int d = 0; d < hidden_sizes.GetCount(); d++) {
			hidden_prevs[d].Init(hidden_sizes[d], 1, 1);
			cell_prevs[d].Init(hidden_sizes[d], 1, 1);
		}
		/*} else {
			hidden_prevs <<= prev.h;
			cell_prevs <<= prev.c;
		}*/
		
		
		// return cell memory, hidden representation and output
		CellMemory cm;
		//return {'h':hidden, 'c':cell, 'o' : output};
		
		Vector<Volume>& hidden = cm.h;
		Vector<Volume>& cell = cm.c;
		for (int d = 0; d < hidden_sizes.GetCount(); d++) {
			LSTMModel& m = vec.model[d];
			
			Volume& input_vector = d == 0 ? x : hidden[d-1];
			Volume& hidden_prev = hidden_prevs[d];
			Volume& cell_prev = cell_prevs[d];
			
			// input gate
			/*Volume h0 = G.Mul(m.Wix, input_vector);
			Volume h1 = G.Mul(m.Wih, hidden_prev);
			Volume input_gate = G.Sigmoid(G.Add(G.Add(h0,h1),model['bi'+d]));
			
			// forget gate
			Volume h2 = G.Mul(m.Wfx, input_vector);
			Volume h3 = G.Mul(m.Wfh, hidden_prev);
			Volume forget_gate = G.Sigmoid(G.Add(G.Add(h2, h3), model['bf'+d]));
			
			// output gate
			Volume h4 = G.Mul(m.Wox, input_vector);
			Volume h5 = G.Mul(m.Woh, hidden_prev);
			Volume output_gate = G.sigmoid(G.Add(G.Add(h4, h5), m.bo));
			
			// write operation on cells
			Volume h6 = G.Mul(m.Wcx, input_vector);
			Volume h7 = G.Mul(m.Wch, hidden_prev);
			Volume cell_write = G.Tanh(G.Add(G.Add(h6, h7), m.bc));
			
			// compute new cell activation
			Volume retain_cell = G.EltMul(forget_gate, cell_prev); // what do we keep from cell
			Volume write_cell = G.EltMul(input_gate, cell_write); // what do we write to cell
			Volume cell_d = G.Add(retain_cell, write_cell); // new cell contents
			
			// compute hidden state as gated, saturated cell activations
			Volume hidden_d = G.EltMul(output_gate, G.Tanh(cell_d));*/
			
			// input gate
			Volume& h0 = mul(m.Wix, input_vector);
			Volume& h1 = mul(m.Wih, hidden_prev);
			Volume& input_gate = sigmoid(add(add(h0,h1), m.bi));
			
			// forget gate
			Volume& h2 = mul(m.Wfx, input_vector);
			Volume& h3 = mul(m.Wfh, hidden_prev);
			Volume& forget_gate = sigmoid(add(add(h2, h3), m.bf));
			
			// output gate
			Volume& h4 = mul(m.Wox, input_vector);
			Volume& h5 = mul(m.Woh, hidden_prev);
			Volume& output_gate = sigmoid(add(add(h4, h5), m.bo));
			
			// write operation on cells
			Volume& h6 = mul(m.Wcx, input_vector);
			Volume& h7 = mul(m.Wch, hidden_prev);
			Volume& cell_write = tanh(add(add(h6, h7), m.bc));
			
			// compute new cell activation
			Volume& retain_cell = eltmul(forget_gate, cell_prev); // what do we keep from cell
			Volume& write_cell = eltmul(input_gate, cell_write); // what do we write to cell
			Volume& cell_d = add(retain_cell, write_cell); // new cell contents
			
			// compute hidden state as gated, saturated cell activations
			Volume& hidden_d = eltmul(output_gate, tanh(cell_d));
			
			hidden.Add(hidden_d);
			cell.Add(cell_d);
		}
		
		// one decoder to outputs at end
		Volume& output = add(mul(vec.Whd, hidden.Top()), vec.bd);
		cm.o = output;
		
		return cm;
	}
};









struct SolverStat {
	double ratio_clipped;
};

class Solver {
	
	double decay_rate;
	double smooth_eps;
	Vector<Volume> step_cache;
	
public:
	Solver() {
		decay_rate = 0.999;
		smooth_eps = 1e-8;
	}
	
	SolverStat Step(Vector<Volume>& model, int step_size, int regc, int clipval) {
		// perform parameter update
		SolverStat solver_stats;
		int num_clipped = 0;
		int num_tot = 0;
		for(int k = 0; k < model.GetCount(); k++) {
			
			Volume& m = model[k]; // mat ref
			if (k == step_cache.GetCount()) {
				step_cache.Add().Init(m.GetWidth(), m.GetHeight(), m.GetDepth(), 0);
			}
			else {ASSERT(k < step_cache.GetCount());}
			
			/*if (!(k in step_cache)) {
				step_cache[k].Init(m.n, m.d);
			}*/
			
			Volume& s = step_cache[k];
			for (int i = 0; i < m.GetLength(); i++) {
				
				// rmsprop adaptive learning rate
				double mdwi = m.GetGradient(i);
				s.Set(i, s.Get(i) * decay_rate + (1.0 - decay_rate) * mdwi * mdwi);
				
				// gradient clip
				if (mdwi > clipval) {
					mdwi = clipval;
					num_clipped++;
				}
				if (mdwi < -clipval) {
					mdwi = -clipval;
					num_clipped++;
				}
				num_tot++;
				
				// update (and regularize)
				m.Add(i, - step_size * mdwi / sqrt(s.Get(i) + smooth_eps) - regc * m.Get(i));
				m.SetGradient(i, 0); // reset gradients for next iteration
			}
			
		}
		solver_stats.ratio_clipped = num_clipped * 1.0 / num_tot;
		return solver_stats;
	}
};

}

#endif
