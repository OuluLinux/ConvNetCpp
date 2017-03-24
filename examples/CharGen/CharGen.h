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
	
	//RNN rnn;
	//LSTM lstm;
	//Solver solver;
	VectorMap<int, int> letterToIndex;
	VectorMap<int, int> indexToLetter;
	Vector<WString> vocab;
	Vector<WString> data_sents;
	Vector<double> ppl_list;
	Vector<int> hidden_sizes;
	Vector<int> sequence;
	//ModelVector model;
	RecurrentSession ses;
	Volume logprobs;
	String model_str;
	double sample_softmax_temperature;
	int input_size;
	int output_size;
	int epoch_size;
	int tick_iter;
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
	//ModelVector InitModel();
	//CellMemory ForwardIndex(Graph& G, ModelVector& model, int ix, CellMemory* prev);
	WString PredictSentence(bool samplei=false, double temperature=1.0);
	double Median(Vector<double>& values);
	void Tick();
	//void GradCheck();
};

#endif
