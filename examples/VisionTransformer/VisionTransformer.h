#include <Core/Core.h>
#ifndef _VisionTransformer_VisionTransformer_h
#define _VisionTransformer_VisionTransformer_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
#include <PlotCtrl/PlotCtrl.h>

using namespace Upp;
using namespace ConvNet;

#include "LoaderCIFAR10.h"

// Manual layout members
class VisionTransformer : public DockWindow {
public:
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
	typedef VisionTransformer CLASSNAME;
	VisionTransformer();
	~VisionTransformer();

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

	String BuildViTConfig();
};

#endif
