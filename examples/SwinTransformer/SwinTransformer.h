#ifndef _SwinTransformer_SwinTransformer_h
#define _SwinTransformer_SwinTransformer_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#include "LoaderCIFAR10.h"  // Using CIFAR-10 loader as per the transformer implementation

#define IMAGECLASS SwinTransformerImg
#define IMAGEFILE <SwinTransformer/SwinTransformer.iml>
#include <Draw/iml_header.h>

// Swin Transformer example application
class SwinTransformer : public DockWindow {
	ParentCtrl settings;
	Label lrate, lmom, lbatch, ldecay;
	EditDouble rate, mom, decay;
	EditInt batch;
	Button apply, save_net, load_net, pause;
	TrainingGraph graph;
	Label status;
	SessionConvLayers layer_view;
	ImagePrediction pred_view;

	// Network
	ParentCtrl net_ctrl;
	DocEdit net_edit;
	Button reload_btn;

	Splitter v_split;

	Session ses;
	String t;
	SpinLock ticking_lock;
	Size img_sz;
	int average_size;
	int max_diff_imgs;
	int augmentation;
	bool is_training;
	bool do_flip;
	bool has_colors;

public:
	typedef SwinTransformer CLASSNAME;
	SwinTransformer();
	~SwinTransformer();

	virtual void DockInit();

	Session& GetSession() {return ses;}

	void Refresher();
	void ApplySettings();
	void OpenFile();
	void SaveFile();
	void Reload();
	void Pause();
	void RefreshStatus();
	void RefreshPredictions() {pred_view.Refresh();}

	void UpdateNetParamDisplay();
	void ResetAll();
	void PostReload() {PostCallback(THISBACK(Reload));}
};

#endif