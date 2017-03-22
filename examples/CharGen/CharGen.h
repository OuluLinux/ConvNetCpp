#ifndef _CharGen_CharGen_h
#define _CharGen_CharGen_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS CharGenImg
#define IMAGEFILE <CharGen/CharGen.iml>
#include <Draw/iml_header.h>

#define LAYOUTFILE <CharGen/CharGen.lay>
#include <CtrlCore/lay.h>

struct Cost {
	Graph G;
	double ppl;
	double cost;
	
	Cost() {}
	Cost(const Cost& c) {*this = c;}
	Cost& operator=(const Cost& c){
		Panic("TODO");
		return *this;
	}
};

class CharGen : public WithMainLayout<TopWindow> {
	
	ReinforceRowPluck rowPluck;
	
	RNN rnn;
	LSTM lstm;
	Solver solver;
	VectorMap<int, int> letterToIndex;
	VectorMap<int, int> indexToLetter;
	Vector<WString> vocab;
	Vector<WString> data_sents;
	Vector<double> ppl_list;
	Vector<int> hidden_sizes;
	ModelVector model;
	Volume logprobs;
	String model_str;
	String generator;
	double regc;
	double clipval;
	double sample_softmax_temperature;
	int input_size;
	int output_size;
	int epoch_size;
	int letter_size;
	int tick_iter;
	int max_chars_gen;
	bool running, stopped;
	
public:
	typedef CharGen CLASSNAME;
	CharGen();
	~CharGen();
	
	void Start();
	void Stop();
	void Ticking();
	void Reset(bool init_reward, bool start);
	void Reload();
	void Refresher();
	
	void SetLearningRate();
	void SetSampleTemperature();
	void Save();
	void Load();
	void LoadPretrained();
	void Pause();
	void InitVocab(Vector<WString>& sents, int count_threshold);
	void UtilAddToModel(LSTMModel& modelto, LSTMModel& modelfrom);
	ModelVector InitModel();
	CellMemory ForwardIndex(Graph& G, ModelVector& model, int ix, CellMemory& prev);
	WString PredictSentence(ModelVector& model, bool samplei=false, double temperature=1.0);
	Cost CostFun(ModelVector& model, const WString& sent);
	double Median(Vector<double>& values);
	void Tick();
	void GradCheck();
};

#endif
