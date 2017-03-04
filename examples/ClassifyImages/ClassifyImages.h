#ifndef _ClassifyImages_ClassifyImages_h
#define _ClassifyImages_ClassifyImages_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
#include <PlotCtrl/PlotCtrl.h>
using namespace Upp;
using namespace ConvNet;

#ifdef flagMNIST
	#include <plugin/LoaderMNIST/LoaderMNIST.h>
#elif defined flagCIFAR10
	#include <plugin/LoaderCIFAR10/LoaderCIFAR10.h>
#endif

#define IMAGECLASS ClassifyImagesImg
#define IMAGEFILE <ClassifyImages/ClassifyImages.iml>
#include <Draw/iml_header.h>


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

class ClassifyImages : public DockWindow {
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
	Window xLossWindow, wLossWindow, trainAccWindow, valAccWindow;
	String t;
	SpinLock ticking_lock;
	Vector<Sample> tmp_samples;
	Sample tmp_sample;
	Volume tmp_vol, aavg;
	Size img_sz;
	double forward_time, backward_time;
	int step_num;
	int average_size;
	int max_diff_imgs;
	int augmentation;
	bool use_validation_data;
	bool is_training;
	bool do_flip;
	bool has_colors;
	bool running, stopped;
	bool ticking_running, ticking_stopped;
	
public:
	typedef ClassifyImages CLASSNAME;
	ClassifyImages();
	~ClassifyImages();
	
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
