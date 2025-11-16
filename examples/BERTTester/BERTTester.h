#ifndef _BERTTester_BERTTester_h
#define _BERTTester_BERTTester_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS BERTTesterImg
#define IMAGEFILE <BERTTester/BERTTester.iml>
#include <Draw/iml_header.h>

// BERTTester example application
class BERTTester : public DockWindow {
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

public:
	typedef BERTTester CLASSNAME;
	BERTTester();
	~BERTTester();

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
};

#endif