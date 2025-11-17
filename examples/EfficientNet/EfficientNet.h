#ifndef _EfficientNet_EfficientNet_h
#define _EfficientNet_EfficientNet_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>

using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS EfficientNetImg
#define IMAGEFILE <EfficientNet/EfficientNet.iml>
#include <Draw/iml_header.h>

// EfficientNet implementation using compound scaling
class EfficientNetApp : public DockWindow {
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

	// EfficientNet-specific controls
	ParentCtrl efficientnet_ctrl;
	DropList model_variant; // For selecting EfficientNet variant (B0, B1, B2, etc.)

public:
	typedef EfficientNetApp CLASSNAME;
	EfficientNetApp();
	~EfficientNetApp();

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

	// EfficientNet-specific functionality
	void OnModelVariantChanged();
	String BuildEfficientNetB0Config(); // Base configuration for EfficientNet-B0
	String BuildScaledConfig(int model_variant); // Build configuration for different EfficientNet variants (B1, B2, etc.)
};

#endif