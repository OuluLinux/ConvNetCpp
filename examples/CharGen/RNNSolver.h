#ifndef _CharGen_RNNSolver_h_
#define _CharGen_RNNSolver_h_



/*
	This was made for finding correct Highway RNN volumes, after failing to put them correctly
	from theory. The implementation shouldn't be considered as correct because of this.
	I still wanted to get it close as possible to working version.
	Any commits which does it correctly are welcome.
	
	Debugging volume sizes by brute-force:
	 - looping whole area is inefficient or impossible
	 - just should understand the theory, and not use this... this is for those who doesn't understand well enough
	 - this tries to resize volumes in combinations: 1,1 -> grid,1 -> 1,grid -> grid,grid
	 - this also tries to reverse order in the multiply operation arguments
	 1. set graph and list range to minimum (1)
	 2. be sure to add to inputs only custom volumes of the model
	 3. after finding successful combination, "freeze" variables by moving them from input to model
	 4. increase graph range after combination is small enough again
	 5. combinations should overlap different ranges: only freeze the most certain variables
*/

namespace RNNSolver {
	
enum {
	RNN_PIPE, RNN_IN1_1W, RNN_IN2_MUL, RNN_IN2_MUL_INV, RNN_IN2_LEN, RNN_IN2_LEN_1OUT
};

class Check : Moveable<Check> {
	
public:
	Check() {
		mode = -1;
		in1 = NULL;
		in2 = NULL;
	}
	
	Check& Init(Size& in, int mode) {
		in1 = &in;
		this->mode = mode;
		return *this;
	}
	
	Check& Init(Size& in1, Size& in2, int mode) {
		this->in1 = &in1;
		this->in2 = &in2;
		this->mode = mode;
		return *this;
	}
	
	bool Chk() {
		ASSERT(in1);
		ASSERT(in1->cx >= 1 && in1->cx <= 100);
		ASSERT(in1->cy >= 1 && in1->cy <= 100);
		if (mode == RNN_PIPE) {
			out = *in1;
		}
		else if (mode == RNN_IN1_1W) {
			out = Size(1, in1->cx);
		}
		else if (mode == RNN_IN2_MUL) {
			ASSERT(in2); ASSERT(in2->cx >= 1 && in2->cx <= 100); ASSERT(in2->cy >= 1 && in2->cy <= 100);
			if (in1->cx != in2->cy)
				return false;
			out = Size(in2->cx, in1->cy);
		}
		else if (mode == RNN_IN2_MUL_INV) {
			ASSERT(in2); ASSERT(in2->cx >= 1 && in2->cx <= 100); ASSERT(in2->cy >= 1 && in2->cy <= 100);
			if (in2->cx != in1->cy)
				return false;
			out = Size(in1->cx, in2->cy);
		}
		else if (mode == RNN_IN2_LEN) {
			ASSERT(in2); ASSERT(in2->cx >= 1 && in2->cx <= 100); ASSERT(in2->cy >= 1 && in2->cy <= 100);
			if (in1->cx * in1->cy != in2->cx * in2->cy)
				return false;
			out = Size(in1->cx, in1->cy);
		}
		else if (mode == RNN_IN2_LEN_1OUT) {
			ASSERT(in2); ASSERT(in2->cx >= 1 && in2->cx <= 100); ASSERT(in2->cy >= 1 && in2->cy <= 100);
			if (in1->cx * in1->cy != in2->cx * in2->cy)
				return false;
			out = Size(1,1);
		}
		return true;
	}
	
	int mode;
	Size *in1, *in2;
	Size out;
};

struct GraphTree {
	
	Array<Check> layers;
	
	
	GraphTree() {
		
	}
	
	void Clear() {layers.Clear();}
	
	Size& RowPluck(int* row, Size& in) {
		return layers.Add().Init(in, RNN_IN1_1W).out;
	}
	
	Size& Tanh(Size& in) {
		return layers.Add().Init(in, RNN_PIPE).out;
	}
	
	Size& Sigmoid(Size& in) {
		return layers.Add().Init(in, RNN_PIPE).out;
	}
	
	Size& Relu(Size& in) {
		return layers.Add().Init(in, RNN_PIPE).out;
	}
	
	Size& Mul(Size& in1, Size& in2) {
		return layers.Add().Init(in1, in2, RNN_IN2_MUL).out;
	}
	
	Size& Add(Size& in1, Size& in2) {
		return layers.Add().Init(in1, in2, RNN_IN2_LEN).out;
	}
	
	Size& Dot(Size& in1, Size& in2) {
		return layers.Add().Init(in1, in2, RNN_IN2_LEN_1OUT).out;
	}
	
	Size& EltMul(Size& in1, Size& in2) {
		return layers.Add().Init(in1, in2, RNN_IN2_LEN).out;
	}
	
	Size& Copy(Size& src, Size& dst) {
		return layers.Add().Init(src, dst, RNN_PIPE).out;
	}
	
	Size& AddConstant(double d, Size& in) {
		return layers.Add().Init(in, RNN_PIPE).out;
	}
	
	Size& MulConstant(double d, Size& in) {
		return layers.Add().Init(in, RNN_PIPE).out;
	}
	
	
};

class ModelSolver {
	
protected:
	Array<Array<GraphTree> > graphs;
	ArrayMap<String, Size> input, model;
	Vector<Vector<Size*> > hidden_prevs, cell_prevs;
	Vector<int> index_sequence, hidden_sizes;
	int max_graphs;
	int letter_size;
	int input_size, output_size;
	int grid;
public:
	
	ModelSolver() {
		
		max_graphs = 1;
		input_size = 33;
		output_size = 22;
		grid = 10;
		
		index_sequence.SetCount(max_graphs, 0);
		graphs.SetCount(max_graphs);
		hidden_prevs.SetCount(max_graphs+1);
		cell_prevs.SetCount(max_graphs+1);
	}
	
	void Init() {
		model.GetAdd("Wil", Size(grid, input_size));
		InitModel();
		InitGraphs();
		DUMPM(model);
	}
	
	void InitGraphs() {
		int hidden_count = hidden_sizes.GetCount();
		ASSERT_(hidden_count > 0, "Hidden sizes must be set");
		
		for (int i = 0; i < hidden_prevs.GetCount(); i++) {
			Vector<Size*>& hidden_prevs = this->hidden_prevs[i];
			Vector<Size*>& cell_prevs = this->cell_prevs[i];
			hidden_prevs.SetCount(hidden_count, NULL);
			cell_prevs.SetCount(hidden_count, NULL);
		}
		
		for(int i = 0; i < hidden_count; i++) {
			String is = IntStr(i);
			hidden_prevs[0][i]	= &model.GetAdd("first_hidden_" + is, Size(1,grid));
			cell_prevs[0][i]	= &model.GetAdd("first_cell_" + is, Size(1,grid));
		}
		
		for (int i = 0; i < graphs.GetCount(); i++) {
			Array<GraphTree>& hidden_graphs = graphs[i];
			hidden_graphs.SetCount(hidden_count);
			for (int j = 0; j < hidden_count; j++) {
				InitModel(i, j, hidden_graphs[j]);
			}
		}
	}
	
	virtual void InitModel() = 0;
	virtual void InitModel(int i, int j, GraphTree& g) = 0;
	
	void Solve() {
		for(int i = 0; i < input.GetCount(); i++) {
			Size sz = input[i];
			ASSERT(sz.cx >= 1 && sz.cx <= 100);
			ASSERT(sz.cy >= 1 && sz.cy <= 100);
		}
		for(int i = 0; i < model.GetCount(); i++) {
			Size sz = model[i];
			ASSERT(sz.cx >= 1 && sz.cx <= 100);
			ASSERT(sz.cy >= 1 && sz.cy <= 100);
		}
		
		Vector<int> comb, max;
		
		Vector<Check*> muls;
		for(int i = 0; i < graphs.GetCount(); i++) {
			for(int j = 0; j < graphs[i].GetCount(); j++) {
				GraphTree& gt = graphs[i][j];
				for(int k = 0; k < gt.layers.GetCount(); k++) {
					Check& c = gt.layers[k];
					if (c.mode == RNN_IN2_MUL)
						muls.Add(&c);
				}
			}
		}
		
		int count = input.GetCount() + muls.GetCount();
		
		if (count) {
			comb.SetCount(count, 0);
			max.SetCount(count, 0);
			for(int i = 0; i < input.GetCount(); i++)
				max[i] = 4;
			for(int i = input.GetCount(); i < count; i++)
				max[i] = 2;
			
			int64 total = 1;
			for(int i = 0; i < max.GetCount(); i++)
				total *= max[i];
			int64 counter = 0;
			
			VectorMap<int, int> blocks;
			
			bool running = true;
			while (running) {
				if (counter % 1000 == 0) {
					Cout() << counter << "/" << total << "\n";
				}
				
				for(int i = 0; i < count; i++) {
					int& c = comb[i];
					if (i < input.GetCount()) {
						Size& sz = input[i];
						switch (c) {
							case 0: sz = Size(1,grid); break;
							case 1: sz = Size(grid,1); break;
							case 2: sz = Size(1,1); break;
							case 3: sz = Size(grid,grid); break;
							default: Panic("Invalid value");
						}
					} else {
						Check& mul = *muls[i-input.GetCount()];
						switch (c) {
							case 0: mul.mode = RNN_IN2_MUL; break;
							case 1: mul.mode = RNN_IN2_MUL_INV; break;
							default: Panic("Invalid value");
						}
					}
				}
				
				bool success = true;
				for(int i = 0; i < graphs.GetCount() && success; i++) {
					Array<GraphTree>& list = graphs[i];
					for(int j = 0; j < list.GetCount() && success; j++) {
						GraphTree& gt = list[j];
						for(int k = 0; k < gt.layers.GetCount() && success; k++) {
							Check& c = gt.layers[k];
							/*if (k == 2) {
								LOG("mode: " << c.mode << " in1: " << c.in1->ToString() << " in2: " << c.in2->ToString());
								Cout() << "mode: " << c.mode << " in1: " << c.in1->ToString() << " in2: " << c.in2->ToString() << "\n";
							}*/
							
							success = c.Chk();
							
							if (!success) {
								blocks.GetAdd(k,0)++;
							}
						}
					}
				}
								
				if (success) {
					LOG("success");
					Cout() << "success\n";
					for(int i = 0; i < count; i++) {
						LOG("\t" << comb[i] << " / " << max[i]);
						Cout() << "\t" << comb[i] << " / " << max[i] << "\n";
					}
					//return;
				}
				
				for(int i = 0; i < count; i++) {
					int& c = comb[i];
					c++;
					if (c < max[i]) break;
					c = 0;
					if (i == count-1) running = false;
				}
				
				
				counter++;
			}
			ASSERT(counter == total);
			
			SortByValue(blocks, StdLess<int>());
			for(int i = 0; i < blocks.GetCount(); i++) {
				Cout() << "		" << blocks.GetKey(i) << " --> " << blocks[i] << "\n";
				LOG("		" << blocks.GetKey(i) << " --> " << blocks[i]);
			}
		}
		else {
			bool success = true;
			for(int i = 0; i < graphs.GetCount() && success; i++) {
				Array<GraphTree>& list = graphs[i];
				for(int j = 0; j < list.GetCount() && success; j++) {
					GraphTree& gt = list[j];
					for(int k = 0; k < gt.layers.GetCount() && success; k++) {
						success = gt.layers[k].Chk();
					}
				}
			}
			
			LOG("success = " << (int)success);
		}
	}


};


class HighwaySolver : public ModelSolver {
	
	Size* input_sz;
	double initial_bias;
	
public:

	HighwaySolver() {
		input_sz = NULL;
		
		hidden_sizes.Add(10);
		hidden_sizes.Add(10);
		
		letter_size		= hidden_sizes[0];
	}
	
	virtual void InitModel() {
		int hidden_size = 0;
		
		// loop over depths
		for (int d = 0; d < hidden_sizes.GetCount(); d++) {
			String is = IntStr(d);
			
			// loop over depths
			//int prev_size = d == 0 ? hidden_sizes[d] : 1;
			hidden_size = hidden_sizes[d];
			
			model.GetAdd("Wix_" + is, Size(1,1));
			model.GetAdd("Wih_" + is, Size(1,1));
			model.GetAdd("noise_i_" + is + "_0", Size(grid,1));
			model.GetAdd("noise_i_" + is + "_1", Size(1,1));
			model.GetAdd("noise_h_" + is + "_0", Size(grid,1));
			model.GetAdd("noise_h_" + is + "_1", Size(1,1));
		}
		
		// decoder params
		model.GetAdd("Whd", Size(grid, output_size));
		model.GetAdd("bd", Size(1, output_size));
	}
	
	virtual void InitModel(int i, int j, GraphTree& g) {
		String js = IntStr(j);
	
		g.Clear();
		
		Vector<Size*>& hidden_prevs = this->hidden_prevs[i];
		Vector<Size*>& hidden_nexts = this->hidden_prevs[i+1];
		
		if (j == 0) {
			input_sz = &g.RowPluck(&index_sequence[i], model.Get("Wil"));
		}
		
		Size& input_vector = j == 0 ? *input_sz : *hidden_nexts[j-1];
		Size& hidden_prev = *hidden_prevs[j];
		
		Size* i2h[2];
		Size* h2h_tab[2];
		
		if (j == 0) {
			{
				Size& dropped_x			= g.Mul(input_vector, model.Get("noise_i_" + js + "_0"));
				Size& dropped_h_tab		= g.Mul(hidden_prev, model.Get("noise_h_" + js + "_0"));
				i2h[0]					= &g.Mul(model.Get("Wix_" + js), dropped_x);
				h2h_tab[0]				= &g.Mul(model.Get("Wih_" + js), dropped_h_tab);
			}
			{
				Size& dropped_x			= g.Mul(input_vector, model.Get("noise_i_" + js + "_1"));
				Size& dropped_h_tab		= g.Mul(hidden_prev, model.Get("noise_h_" + js + "_1"));
				i2h[1]					= &g.Mul(model.Get("Wix_" + js), dropped_x);
				h2h_tab[1]				= &g.Mul(model.Get("Wih_" + js), dropped_h_tab);
			}
			Size& t_gate_tab			= g.Sigmoid(g.AddConstant(initial_bias, g.Add(*i2h[0], *h2h_tab[0])));
			Size& in_transform_tab		= g.Tanh(g.Add(*i2h[1], *h2h_tab[1]));
			Size& c_gate_tab			= g.AddConstant(1, g.MulConstant(-1, t_gate_tab));
			Size& hidden_d				= g.Add(
											g.Mul(c_gate_tab, hidden_prev),
											g.Mul(t_gate_tab, in_transform_tab));
			
			hidden_nexts[j] = &hidden_d;
		}
		else
		{
			{
				Size& dropped_h_tab		= g.Mul(input_vector, model.Get("noise_h_" + js + "_0"));
				h2h_tab[0]				= &g.Mul(model.Get("Wix_" + js), dropped_h_tab);
			}
			{
				Size& dropped_h_tab		= g.Mul(input_vector, model.Get("noise_h_" + js + "_1"));
				h2h_tab[1]				= &g.Mul(model.Get("Wix_" + js), dropped_h_tab);
			}
			Size& t_gate_tab			= g.Sigmoid(g.AddConstant(initial_bias, *h2h_tab[0]));
			Size& in_transform_tab		= g.Tanh(*h2h_tab[1]);
			Size& c_gate_tab			= g.AddConstant(1, g.MulConstant(-1, t_gate_tab));
			Size& hidden_d				= g.Add(
											g.Mul(c_gate_tab, input_vector),
											g.Mul(t_gate_tab, in_transform_tab));
			
			hidden_nexts[j] = &hidden_d;
		}
		
		
		// one decoder to outputs at end
		if (j == hidden_prevs.GetCount() - 1) {
			g.Add(g.Mul(model.Get("Whd"), *hidden_nexts.Top()), model.Get("bd"));
		}
	}
	
};


inline void SolveHighway() {
	HighwaySolver s;
	s.Init();
	s.Solve();
	LOG("Finished");
}

}

#endif
