#ifndef _BERTTester_BERTTester_h
#define _BERTTester_BERTTester_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace Upp;
using namespace ConvNet;

// BERTTester example application with MLM and NSP functionality
class BERTTester : public DockWindow {
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

	// BERT-specific controls
	ParentCtrl bert_ctrl;
	EditField input_text1;
	EditField input_text2;
	EditField output_text;
	Button tokenize_btn;
	Button mask_btn;
	Button predict_btn;
	Button task_select; // Toggle between MLM and NSP

	// Task state
	int current_task; // 0=MLM, 1=NSP

public:
	typedef BERTTester CLASSNAME;
	BERTTester();
	~BERTTester();

	void Init() {}
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

	// BERT-specific functionality
	void OnTaskToggle();
	void OnTokenize();
	void OnMask();
	void OnPredict();
	String BuildBERTConfig();
};

#endif
