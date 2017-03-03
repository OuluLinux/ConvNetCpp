#ifndef _ClassifyMNIST_ClassifyMNIST_h
#define _ClassifyMNIST_ClassifyMNIST_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
#include <PlotCtrl/PlotCtrl.h>
using namespace Upp;
using namespace ConvNet;


#define IMAGECLASS ClassifyMNISTImg
#define IMAGEFILE <ClassifyMNIST/ClassifyMNIST.iml>
#include <Draw/iml_header.h>

class ImageBank {
	
	
public:
	
	Vector<int> labels, test_labels;
	Vector<Image> images, test_images;
	
	void Serialize(Stream& s) {s % labels % test_labels % images % test_images;}
	
};

inline ImageBank& GetImageBank() {return Single<ImageBank>();}


class Loader : public TopWindow {
	
	Label lbl;
	ProgressIndicator prog, sub;
	Button cancel;
	int ret_value;
	
public:
	typedef Loader CLASSNAME;
	Loader();
	
	void Cancel() {ret_value = 1; Close();}
	void Load();
	void Progress(int actual, int total, String label);
	bool SubProgress(int actual, int total);
	void Close0() {Close();}
	bool IsFail() {return ret_value;}
	
	
};

struct Sample : Moveable<Sample> {
	Sample() {}
	Sample(const Sample& src) {*this = src;}
	Sample& operator=(const Sample& src) {
		x = src.x;
		label = src.label;
		img_id = src.img_id;
		isval = src.isval;
		return *this;
	}
	Volume x;
	int label, img_id;
	bool isval;
};

class ClassifyMNIST : public DockWindow {
	ParentCtrl settings;
	Label lrate, lmom, lbatch, ldecay;
	EditDouble rate, mom, decay;
	EditInt batch;
	Button apply, save_net, load_net;
	
	Label status;
	PlotCtrl loss_graph;
	ParentCtrl net_ctrl;
	DocEdit net_edit;
	Button reload_btn;
	SessionConvLayers layer_view;
	ImagePrediction pred_view;
	
	Splitter v_split;
	
	Session ses;
	Vector<String> classes;
	Window xLossWindow, wLossWindow, trainAccWindow, valAccWindow;
	String t;
	SpinLock ticking_lock;
	Vector<Sample> tmp_samples;
	Sample tmp_sample;
	Volume tmp_vol, aavg;
	double forward_time, backward_time;
	int step_num;
	int average_size;
	int max_diff_imgs;
	bool use_validation_data;
	bool is_training;
	bool running, stopped;
	bool ticking_running, ticking_stopped;
	
public:
	typedef ClassifyMNIST CLASSNAME;
	ClassifyMNIST();
	~ClassifyMNIST();
	
	virtual void DockInit();
	
	void Start();
	void Refresher();
	void ApplySettings();
	void OpenFile();
	void SaveFile();
	void Reload();
	void AddLoss();
	void Ticking();
	void Tick();
	void RefreshStatus();
	void StopRefresher() {running = false; while (!stopped) Sleep(100);}
	void StopTicking() {ticking_running = false; while (!ticking_stopped) Sleep(100);}
	void RefreshPredictions() {pred_view.Refresh();}
	
	void SampleTrainingInstance(Sample& sample);
	void SampleTestInstance(Vector<Sample>& samples);
	void TestPredict();
	void Step(Sample& sample);
	void UpdateNetParamDisplay();
	void ResetAll();
	
};

#endif
