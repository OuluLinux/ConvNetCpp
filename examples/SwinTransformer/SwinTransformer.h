#include <Core/Core.h>
#ifndef _SwinTransformer_SwinTransformer_h
#define _SwinTransformer_SwinTransformer_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
#include <PlotCtrl/PlotCtrl.h>

using namespace Upp;
using namespace ConvNet;

#include "LoaderCIFAR10.h"

// Swin Transformer example application
class SwinTransformer : public DockWindow {
public:
    void Init() {}

	ParentCtrl settings;
	Label lrate, lmom, lbatch, ldecay;
	EditDouble rate, mom, decay;
	EditInt batch;
	Button apply, save_net, load_net, pause;
	TrainingGraph graph;
	Label status;
	SessionConvLayers layer_view;

	// Network
	ParentCtrl net_ctrl;
	DocEdit net_edit;
	Button reload_btn;

	Splitter v_split;

	Session ses;
	String t;
	SpinLock ticking_lock;
	int average_size;
	int max_diff_imgs;
	int augmentation;
	bool is_training;

	LoaderCIFAR10 loader;

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

	void UpdateNetParamDisplay();
	void ResetAll();
	void PostReload() {PostCallback(THISBACK(Reload));}

	String BuildSwinConfig();
};

#endif
