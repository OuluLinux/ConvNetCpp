#include "CharGen.h"
#include "CharGen.brc"
#include <plugin/bz2/bz2.h>

#define M_LOG2E 1.44269504088896340736 //log2(e)

inline long double log2(const long double x){
    return  log(x) * M_LOG2E;
}

/*

// model parameters
generator = 'lstm'; // can be 'rnn' or 'lstm'
hidden_sizes = [20,20]; // list of sizes of hidden layers
letter_size = 5; // size of letter embeddings

// optimization
regc = 0.000001; // L2 regularization strength
learning_rate = 0.01; // learning rate
clipval = 5.0; // clip gradients at this value
  
 */
 
CharGen::CharGen() {
	Title("Character Generation Demo");
	Icon(CharGenImg::icon());
	MaximizeBox().MinimizeBox();
	
	MemReadStream input_mem(input_txt, input_txt_length);
	String input_str = BZ2Decompress(input_mem);
	
	running = false;
	stopped = true;
	
	model_str = "{\n"
			"\t\"generator\":\"lstm\",\n"
			"\t\"hidden_width\":20,\n"
			"\t\"hidden_height\":20,\n"
			"\t\"letter_size\":5,\n"
			"\t\"regc\":0.000001,\n"
			"\t\"learning_rate\":0.01,\n"
			"\t\"clipval\":5.0\n"
			"}";
	
	input.SetData(input_str);
	model_edit.SetData(model_str);
	
	learning_rate.MinMax(1,1000);
	learning_rate.SetData(1000);
	learning_rate <<= THISBACK(SetLearningRate);
	
	temp.MinMax(10, 900);
	temp.SetData(100);
	temp <<= THISBACK(SetSampleTemperature);
	save <<= THISBACK(Save);
	load <<= THISBACK(Load);
	pretrained <<= THISBACK(LoadPretrained);
	pause <<= THISBACK(Pause);
	
	CtrlLayout(*this);
	
	Size sz = Zsz(880, 430);
	SetRect(0,0,sz.cx,sz.cy);
	
	tick_iter = 0;
	max_chars_gen = 100;
	regc = 0.000001;
	clipval = 5.0;
	sample_softmax_temperature = 1.0;
	
	epoch_size = -1;
	input_size = -1;
	output_size = -1;
	
	PostCallback(THISBACK(Reload));
	PostCallback(THISBACK(Start));
}

CharGen::~CharGen() {
	Stop();
}

void CharGen::Start() {
	Stop();
	running = true;
	stopped = false;
	Thread::Start(THISBACK(Tick));
}

void CharGen::Stop() {
	running = false;
	while (!stopped) Sleep(100);
}

void CharGen::Ticking() {
	while (running) {
		Tick();
	}
	stopped = true;
}

void CharGen::Reset(bool init_reward, bool start) {
	
}

void CharGen::Reload() {
	// note: reinit writes global vars
	
	// eval options to set some globals
	//eval($("#newnet").val());
	
	//reinit_learning_rate_slider();
	SetLearningRate();
	/*
	solver = new R.Solver(); // reinit solver
	pplGraph = new Rvis.Graph();
	
	ppl_list.SetCount(0);
	tick_iter = 0;
	
	// process the input, filter out blanks
	Vector<WString> data_sents_raw = Split((WString)input.GetData(), "\n");
	data_sents.Clear();
	for (int i=0; i < data_sents_raw.GetCount(); i++) {
		WString sent = TrimBoth(data_sents_raw[i].ToString()).ToWString();
		if (sent.GetCount() > 0) {
			data_sents.Add(sent);
		}
	}
	
	InitVocab(data_sents, 1); // takes count threshold for characters
	model = InitModel();*/
}

void CharGen::Refresher() {
	
}

void CharGen::SetLearningRate() {
	
}

void CharGen::SetSampleTemperature() {
	
}

void CharGen::Save() {
	
}

void CharGen::Load() {
	
}

void CharGen::LoadPretrained() {
	
}

void CharGen::Pause() {
	
}

void CharGen::InitVocab(Vector<WString>& sents, int count_threshold) {
	// go over all characters and keep track of all unique ones seen
	WString txt = Join(sents, ""); // concat all
	
	// count up all characters
	VectorMap<int, int> d;
	for (int i = 0; i < txt.GetCount();i++) {
		int txti = txt[i];
		d.GetAdd(txti, 0)++;
	}
	
	// filter by count threshold and create pointers
	letterToIndex.Clear();
	indexToLetter.Clear();
	vocab.Clear();
	
	// NOTE: start at one because we will have START and END tokens!
	// that is, START token will be index 0 in model letter vectors
	// and END token will be index 0 in the next character softmax
	int q = 1;
	for (int i = 0; i < d.GetCount(); i++) {
		int ch = d.GetKey(i);
		if (d[i] >= count_threshold) {
			// add character to vocab
			letterToIndex.Add(ch, q);
			indexToLetter.Add(q, ch);
			/*WString ws;
			ws.Cat(ch);
			vocab.Add(ws);*/
			vocab.Add().Cat(ch);
			q++;
		}
	}
	
	// globals written: indexToLetter, letterToIndex, vocab (list), and:
	input_size	= vocab.GetCount() + 1;
	output_size	= vocab.GetCount() + 1;
	epoch_size	= sents.GetCount();
	
	input_stats.SetLabel(Format("found %d distinct characters: %s", vocab.GetCount(), Join(vocab, WString(""))));
}

void CharGen::UtilAddToModel(LSTMModel& modelto, LSTMModel& modelfrom) {
	/*for (int k = 0; k < modelfrom) {
		// copy over the pointer but change the key to use the append
		modelto[k] = modelfrom[k];
	}*/
	modelto = modelfrom;
}

ModelVector CharGen::InitModel() {
	// letter embedding vectors
	ModelVector model;
	model.Wil = RandVolume(input_size, letter_size, 0, 0.08);
	
	if (generator == "rnn") {
		rnn.Init(letter_size, hidden_sizes, output_size);
		//UtilAddToModel(model, rnn);
	} else {
		lstm.Init(letter_size, hidden_sizes, output_size);
		//UtilAddToModel(model, lstm);
	}
	Panic("TODO");
	
	return model;
}

CellMemory CharGen::ForwardIndex(Graph& G, ModelVector& model, int ix, CellMemory& prev) {
	rowPluck.SetRow(ix);
	Volume& x = rowPluck(model.Wil);
	// forward prop the sequence learner
	if (generator == "rnn") {
		return rnn.Forward(G, model, hidden_sizes, x, &prev);
	} else {
		return lstm.Forward(G, model, hidden_sizes, x, &prev);
	}
}

WString CharGen::PredictSentence(ModelVector& model, bool samplei, double temperature) {
	Graph G;
	WString s;
	CellMemory prev;
	while (true) {
		
		// RNN tick
		int ix = s.IsEmpty() ? 0 : letterToIndex[s[s.GetCount()-1]];
		CellMemory lh = ForwardIndex(G, model, ix, prev);
		prev = lh;
		
		// sample predicted letter
		logprobs = lh.o;
		if (temperature != 1.0 && samplei) {
			// scale log probabilities by temperature and renormalize
			// if temperature is high, logprobs will go towards zero
			// and the softmax outputs will be more diffuse. if temperature is
			// very low, the softmax outputs will be more peaky
			for (int q = 0; q < logprobs.GetLength(); q++) {
				logprobs.Set(q, logprobs.Get(q) / temperature);
			}
		}
		
		Volume& probs = Softmax(logprobs);
		ix = 0;
		if (samplei) {
			ix = probs.GetSampledColumn();
		} else {
			ix = probs.GetMaxColumn();
		}
		
		if (ix == 0) break; // END token predicted, break out
		if (s.GetCount() > max_chars_gen) break; // something is wrong
		
		int letter = indexToLetter[ix];
		s.Cat(letter);
	}
	return s;
}

Cost CharGen::CostFun(ModelVector& model, const WString& sent) {
	// takes a model and a sentence and
	// calculates the loss. Also returns the Graph
	// object which can be used to do backprop
	int n = sent.GetCount();
	Cost c;
	Graph& G = c.G;
	double log2ppl = 0.0;
	double cost = 0.0;
	CellMemory prev;
	for (int i = -1; i < n; i++) {
		// start and end tokens are zeros
		int ix_source = i ==  -1 ? 0 : letterToIndex[sent[ i ]]; // first step: start with START token
		int ix_target = i == n-1 ? 0 : letterToIndex[sent[i+1]]; // last step: end with END token
		
		CellMemory lh = ForwardIndex(G, model, ix_source, prev);
		prev = lh;
		
		// set gradients into logprobabilities
		logprobs = lh.o; // interpret output as logprobs
		Volume probs = Softmax(logprobs); // compute the softmax probabilities
		
		log2ppl += -log2(probs.Get(ix_target)); // accumulate base 2 log prob and do smoothing
		cost += -log(probs.Get(ix_target));
		
		// write gradients into log probabilities
		for(int j = 0; j < probs.GetLength(); j++)
			logprobs.SetGradient(j, probs.Get(j));
		logprobs.AddGradient(ix_target, -1);
	}
	
	c.ppl = pow(2, log2ppl / (n - 1));
	c.cost = cost;
	return c;
}

double CharGen::Median(Vector<double>& values) {
	Sort(values, StdGreater<double>());
	int half = values.GetCount() / 2;
	if (values.GetCount() % 2)
		return values[half];
	else
		return (values[half-1] + values[half]) / 2.0;
}

void CharGen::Tick() {
	
	// sample sentence fromd data
	int sentix = Random(data_sents.GetCount());
	const WString& sent = data_sents[sentix];
	
	TimeStop ts;  // log start timestamp
	
	// evaluate cost function on a sentence
	Cost cost_struct = CostFun(model, sent);
	
	// use built up graph to compute backprop (set .dw fields in mats)
	cost_struct.G.Backward();
	
	// perform param update
	SolverStat solver_stats;// = solver.Step(model, learning_rate, regc, clipval);
	Panic("TODO");
	
	//$("#gradclip").text('grad clipped ratio: ' + solver_stats.ratio_clipped)
	
	int tick_time = ts.Elapsed();
	
	ppl_list.Add(cost_struct.ppl); // keep track of perplexity
	
	// evaluate now and then
	tick_iter += 1;
	if (tick_iter % 50 == 0) {
		// draw samples
		WString samples;
		for (int q = 0; q < 5; q++) {
			if (q) samples += "\n\n";
			samples += PredictSentence(model, true, sample_softmax_temperature);
		}
		this->samples.SetData(samples);
	}
	if (tick_iter % 10 == 0) {
		// draw argmax prediction
		WString pred = PredictSentence(model, false);
		argmaxpred.SetData(pred);
		
		// keep track of perplexity
		lbl_epoch.SetLabel("epoch: " + FormatDoubleFix(tick_iter/epoch_size, 2));
		lbl_perp.SetLabel("perplexity: " + FormatDoubleFix(cost_struct.ppl, 2));
		lbl_time.SetLabel("forw/bwd time per example: " + FormatDoubleFix(tick_time, 1) + "ms");
		
		if (tick_iter % 100 == 0) {
			double median_ppl = Median(ppl_list);
			ppl_list.SetCount(0);
			perp.AddValue(median_ppl);
		}
	}
}

void CharGen::GradCheck() {
	Vector<Volume> model;// = InitModel();
	Panic("TODO");
	String sent = "^test sentence$";
	Cost cost_struct;
	// = CostFun(model, sent);
	Panic("TODO");
	cost_struct.G.Backward();
	double eps = 0.000001;
	double oldval = 0;
	
	for (int k = 0; k < model.GetCount(); k++) {
		Volume& m = model[k]; // mat ref
		for (int i = 0; i < m.GetLength(); i++) {
			oldval = m.Get(i);
			m.Set(i, oldval + eps);
			Cost c0;// = CostFun(model, sent);
			Panic("TODO");
			m.Set(i, oldval - eps);
			Cost c1;// = CostFun(model, sent);
			m.Set(i, oldval);
			
			double gnum = (c0.cost - c1.cost)/(2 * eps);
			double ganal = m.GetGradient(i);
			double relerr = (gnum - ganal)/(fabs(gnum) + fabs(ganal));
			if (relerr > 1e-1) {
				LOG(k << ": numeric: " << gnum << ", analytic: " << ganal << ", err: " << relerr);
			}
		}
	}
}
