#include "CharGen.h"
#include "CharGen.brc"
#include <plugin/bz2/bz2.h>

CharGen::CharGen() {
	Title("Character Generation Demo");
	Icon(CharGenImg::icon());
	MaximizeBox().MinimizeBox();
	
	MemReadStream input_mem(input_txt, input_txt_length);
	String input_str = BZ2Decompress(input_mem);
	
	running = false;
	stopped = true;
	paused = false;
	
	input.SetData(input_str);
	
	learning_rate_slider.MinMax(1,1000);
	learning_rate_slider.SetData(1000);
	learning_rate_slider <<= THISBACK(SetLearningRate);
	
	temp.MinMax(10, 900);
	temp.SetData(100);
	temp <<= THISBACK(SetSampleTemperature);
	restart <<= THISBACK(Reset);
	save <<= THISBACK(Save);
	load <<= THISBACK(Load);
	load_pretrained <<= THISBACK(LoadPretrained);
	pause <<= THISBACK(Pause);
	resume <<= THISBACK(Resume);
	set_rnn <<= THISBACK1(SetPreset, 0);
	set_lstm <<= THISBACK1(SetPreset, 1);
	set_rhn <<= THISBACK1(SetPreset, 2);
	
	CtrlLayout(*this);
	
	network_view.SetRecurrentSession(ses);
	
	Size sz = Zsz(880, 544);
	SetRect(0,0,sz.cx,sz.cy);
	
	tick_iter = 0;
	sample_softmax_temperature = 1.0;
	
	epoch_size = -1;
	input_size = -1;
	output_size = -1;
	
	PostCallback(THISBACK1(SetPreset, 1));
	PostCallback(THISBACK(Reload));
	PostCallback(THISBACK(Start));
	
	PostCallback(THISBACK(Refresher));
}

CharGen::~CharGen() {
	Stop();
}

void CharGen::Refresher() {
	network_view.Refresh();
	
	PostCallback(THISBACK(Refresher));
}

void CharGen::SetPreset(int i) {
	if (i == 0) {
		model_str = "{\n"
	
			// model parameters
			"\t\"generator\":\"rnn\",\n" // can be 'rnn' or 'lstm' or 'highway'
			"\t\"hidden_sizes\":[20,20],\n" // list of sizes of hidden layers
			"\t\"letter_size\":5,\n" // size of letter embeddings
			
			// optimization
			"\t\"regc\":0.000001,\n" // L2 regularization strength
			"\t\"learning_rate\":0.01,\n" // learning rate
			"\t\"clipval\":5.0\n" // clip gradients at this value
			"}";
	}
	else if (i == 1) {
		model_str = "{\n"
	
			// model parameters
			"\t\"generator\":\"lstm\",\n" // can be 'rnn' or 'lstm' or 'highway'
			"\t\"hidden_sizes\":[20,20],\n" // list of sizes of hidden layers
			"\t\"letter_size\":5,\n" // size of letter embeddings
			
			// optimization
			"\t\"regc\":0.000001,\n" // L2 regularization strength
			"\t\"learning_rate\":0.01,\n" // learning rate
			"\t\"clipval\":5.0\n" // clip gradients at this value
			"}";
	}
	else if (i == 2) {
		model_str = "{\n"
	
			// model parameters
			"\t\"generator\":\"highway\",\n" // can be 'rnn' or 'lstm' or 'highway'
			"\t\"hidden_sizes\":[20,20],\n" // list of sizes of hidden layers
			
			// optimization
			"\t\"regc\":0.000001,\n" // L2 regularization strength
			"\t\"learning_rate\":0.01,\n" // learning rate
			"\t\"clipval\":5.0\n" // clip gradients at this value
			"}";
	}
	model_edit.SetData(model_str);
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
		while (running && paused) Sleep(100);
	}
	stopped = true;
}

void CharGen::Reset() {
	Stop();
	Reload();
	Start();
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
	ValueMap js = ParseJSON((String)model_edit.GetData());
	ses.Load(js);
	ses.SetInputSize(input_size);
	ses.SetOutputSize(output_size);
	ses.Init();
	
	learning_rate_slider.SetData(ses.GetLearningRate() / 0.00001);
	SetLearningRate();
}

void CharGen::SetLearningRate() {
	double value = learning_rate_slider.GetData();
	value *= 0.00001;
	ses.SetLearningRate(value);
	lbl_learning_rate.SetLabel(FormatDoubleFix(value, 5, FD_ZEROS));
}

void CharGen::SetSampleTemperature() {
	double value = temp.GetData();
	value *= 0.01;
	sample_softmax_temperature = value;
	lbl_temp.SetLabel(FormatDoubleFix(value, 2, FD_ZEROS));
}

void CharGen::Save() {
	String file = SelectFileSaveAs("JSON files\t*.json\nAll files\t*.*");
	if (file.IsEmpty()) return;
	
	// Save json
	String json;
	StoreJSON(json);
	
	FileOut fout(file);
	if (!fout.IsOpen()) {
		PromptOK("Error: could not open file " + file);
		return;
	}
	fout << json;
}

void CharGen::Load() {
	String file = SelectFileOpen("JSON files\t*.json\nAll files\t*.*");
	if (file.IsEmpty()) return;
	
	if (!FileExists(file)) {
		PromptOK("File does not exists");
		return;
	}
	
	// Load json
	String json = LoadFile(file);
	if (json.IsEmpty()) {
		PromptOK("File is empty");
		return;
	}
	
	LoadJSON(json);
}

void CharGen::LoadPretrained() {
	Stop();
	MemReadStream pretrained_mem(pretrained, pretrained_length);
	String pretrained_str = BZ2Decompress(pretrained_mem);
	LoadJSON(pretrained_str);
}

void CharGen::LoadJSON(const String& json) {
	Stop();
	
	{
		ValueMap map = ParseJSON(json);
		
		ses.Load(map);
		ses.SetInputSize(input_size);
		ses.SetOutputSize(output_size);
		ses.InitGraphs();
		
		letterToIndex.Clear();
		indexToLetter.Clear();
		vocab.Clear();
		
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
		
		perp.Clear();
		tick_iter = 0;
		learning_rate_slider.SetData(ses.GetLearningRate() / 0.00001);
		SetLearningRate();
	}
	
	Start();
}

void CharGen::StoreJSON(String& json) {
	Stop();
	
	{
		ValueMap js;
		
		ses.Store(js);
		
		ValueMap l2i;
		for(int i = 0; i < letterToIndex.GetCount(); i++) {
			int k = letterToIndex.GetKey(i);
			int v = letterToIndex[i];
			WString ws;
			ws.Cat(k);
			l2i.Add(ws.ToString(), v);
		}
		js.GetAdd("letterToIndex") = l2i;
	
		ValueMap i2l;
		int count = indexToLetter.GetCount();
		for(int i = 0; i < count; i++) {
			int k = indexToLetter.GetKey(i);
			int v = indexToLetter[i];
			WString ws;
			ws.Cat(v);
			i2l.Add(IntStr(k), ws.ToString());
		}
		js.GetAdd("indexToLetter") = i2l;
		
		ValueMap v;
		for(int i = 0; i < vocab.GetCount(); i++) {
			v.Add(IntStr(i), vocab[i].ToString());
		}
		js.GetAdd("vocab") = v;
		
		json = FixJsonComma(AsJSON(js, true));
	}
	
	Start();
}

void CharGen::Pause() {
	paused = true;
}

void CharGen::Resume() {
	paused = false;
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
		
		PostCallback(THISBACK1(SetSamples, samples));
	}
	if (tick_iter % 10 == 0) {
		// draw argmax prediction
		WString pred = PredictSentence(false);
		
		PostCallback(THISBACK1(SetArgMaxSample, pred));
		
		// keep track of perplexity
		PostCallback(THISBACK3(SetStats, (double)tick_iter/epoch_size, ppl, tick_time));
		
		if (tick_iter % 100 == 0) {
			double median_ppl = Median(ppl_list);
			ppl_list.SetCount(0);
			perp.AddValue(median_ppl);
		}
	}
}

void CharGen::SetStats(double epoch, double ppl, int time) {
	lbl_epoch.SetLabel("epoch: " + FormatDoubleFix(epoch, 2, FD_ZEROS));
	lbl_perp.SetLabel("perplexity: " + FormatDoubleFix(ppl, 2, FD_ZEROS));
	lbl_time.SetLabel("forw/bwd time per example: " + FormatDoubleFix(time, 1) + "ms");
}
