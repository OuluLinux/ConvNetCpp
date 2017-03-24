#include "CharGen.h"
#include "CharGen.brc"
#include <plugin/bz2/bz2.h>



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
			"\t\"hidden_sizes\":[20,20],\n"
			"\t\"letter_size\":5,\n"
			"\t\"regc\":0.000001,\n"
			"\t\"learning_rate\":0.01,\n"
			"\t\"clipval\":5.0\n"
			"}";
	
	input.SetData(input_str);
	model_edit.SetData(model_str);
	
	learning_rate_slider.MinMax(1,1000);
	learning_rate_slider.SetData(1000);
	learning_rate_slider <<= THISBACK(SetLearningRate);
	
	temp.MinMax(10, 900);
	temp.SetData(100);
	temp <<= THISBACK(SetSampleTemperature);
	save <<= THISBACK(Save);
	load <<= THISBACK(Load);
	load_pretrained <<= THISBACK(LoadPretrained);
	pause <<= THISBACK(Pause);
	
	CtrlLayout(*this);
	
	Size sz = Zsz(880, 430);
	SetRect(0,0,sz.cx,sz.cy);
	
	tick_iter = 0;
	sample_softmax_temperature = 1.0;
	
	epoch_size = -1;
	input_size = -1;
	output_size = -1;
	
	PostCallback(THISBACK(Reload));
	//PostCallback(THISBACK(LoadPretrained));
	PostCallback(THISBACK(Start));
}

CharGen::~CharGen() {
	Stop();
}

void CharGen::Start() {
	Stop();
	running = true;
	stopped = false;
	Thread::Start(THISBACK(Ticking));
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
	
	perp.Clear();
	
	tick_iter = 0;
	
	// process the input, filter out blanks
	Vector<WString> data_sents_raw = Split((WString)input.GetData(), String("\n").ToWString());
	data_sents.Clear();
	for (int i=0; i < data_sents_raw.GetCount(); i++) {
		WString sent = TrimBoth(data_sents_raw[i].ToString()).ToWString();
		if (sent.GetCount() > 0) {
			data_sents.Add(sent);
		}
	}
	
	InitVocab(data_sents, 1); // takes count threshold for characters
	
	// eval options to set some globals
	ses.LoadJSON(model_edit.GetData());
	ses.SetInputSize(input_size);
	ses.SetOutputSize(output_size);
	ses.Init();
	
	learning_rate_slider.SetData(ses.GetLearningRate() / 0.00001);
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
	Stop();
	
	MemReadStream pretrained_mem(pretrained, pretrained_length);
	String pretrained_str = BZ2Decompress(pretrained_mem);
	
	ses.LoadJSON(pretrained_str);
	ses.SetInputSize(input_size);
	ses.SetOutputSize(output_size);
	ses.Init();
	ses.LoadJSON(pretrained_str);
	
	letterToIndex.Clear();
	indexToLetter.Clear();
	vocab.Clear();
	
	ValueMap map = ParseJSON(pretrained_str);
	
	ValueMap l2i = map.GetAdd("letterToIndex");
	for(int i = 0; i < l2i.GetCount(); i++) {
		String k = l2i.GetKey(i);
		int v = l2i.GetValue(i);
		letterToIndex.Add(k.ToWString()[0], v);
	}
	
	ValueMap i2l = map.GetAdd("indexToLetter");
	int count = i2l.GetCount();
	for(int i = 0; i < count; i++) {
		String k = i2l.GetKey(i);
		String v = i2l.GetValue(i);
		indexToLetter.Add(StrInt(k), v.ToWString()[0]);
	}
	
	ValueMap v = map.GetAdd("vocab");
	for(int i = 0; i < v.GetCount(); i++) {
		vocab.Add(String(v[i]).ToWString());
	}
	
	Start();
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

WString CharGen::PredictSentence(bool samplei, double temperature) {
	Vector<int> sequence;
	ses.Predict(sequence, samplei, temperature);
	
	WString s;
	for(int i = 0; i < sequence.GetCount(); i++) {
		int chr = indexToLetter.Get(sequence[i]);
		s.Cat(chr);
	}
	return s;
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
	TimeStop ts;  // log start timestamp
	
	// sample sentence fromd data
	int sentix = Random(data_sents.GetCount());
	const WString& sent = data_sents[sentix];
	LOG(sent.ToString());
	
	
	sequence.SetCount(sent.GetCount());
	for(int i = 0; i < sent.GetCount(); i++) {
		sequence[i] = letterToIndex.Get(sent[i]);
	}
	
	ses.Learn(sequence);
	
	int tick_time = ts.Elapsed();
	
	double ppl = ses.GetPerplexity();
	ppl_list.Add(ppl); // keep track of perplexity
	
	// evaluate now and then
	tick_iter += 1;
	if (tick_iter % 50 == 0) {
		// draw samples
		WString samples;
		for (int q = 0; q < 5; q++) {
			if (q) samples += "\n\n";
			samples += PredictSentence(true, sample_softmax_temperature);
		}
		
		GuiLock __;
		this->samples.SetData(samples);
	}
	if (tick_iter % 10 == 0) {
		// draw argmax prediction
		WString pred = PredictSentence(false);
		LOG("Predicted: " << pred);
		
		GuiLock __;
		argmaxpred.SetData(pred);
		
		// keep track of perplexity
		lbl_epoch.SetLabel("epoch: " + FormatDoubleFix((double)tick_iter/epoch_size, 2));
		lbl_perp.SetLabel("perplexity: " + FormatDoubleFix(ppl, 2));
		lbl_time.SetLabel("forw/bwd time per example: " + FormatDoubleFix(tick_time, 1) + "ms");
		
		if (tick_iter % 100 == 0) {
			double median_ppl = Median(ppl_list);
			ppl_list.SetCount(0);
			perp.AddValue(median_ppl);
		}
	}
}
