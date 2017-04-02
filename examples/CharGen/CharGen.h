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
	VectorMap<int, int> letterToIndex;
	VectorMap<int, int> indexToLetter;
	Vector<WString> vocab;
	Vector<WString> data_sents;
	Vector<double> ppl_list;
	Vector<int> hidden_sizes;
	Vector<int> sequence;
	RecurrentSession ses;
	Volume logprobs;
	String model_str;
	SpinLock lock;
	double sample_softmax_temperature;
	int input_size;
	int output_size;
	int epoch_size;
	int tick_iter;
	bool running, stopped, paused;
	
public:
	typedef CharGen CLASSNAME;
	CharGen();
	~CharGen();
	
	void Start();
	void Stop();
	void Ticking();
	void Reset();
	void Reload();
	
	void SetPreset(int i);
	void SetLearningRate();
	void SetSampleTemperature();
	void Save();
	void Load();
	void LoadPretrained();
	void LoadJSON(const String& json);
	void StoreJSON(String& json);
	void Pause();
	void Resume();
	void Refresher();
	void InitVocab(Vector<WString>& sents, int count_threshold);
	WString PredictSentence(bool samplei=false, double temperature=1.0);
	double Median(Vector<double>& values);
	void Tick();
	void SetSamples(WString s) {samples.SetData(s);}
	void SetArgMaxSample(WString s) {argmaxpred.SetData(s);}
	void SetStats(double epoch, double ppl, int time);
};

#endif
