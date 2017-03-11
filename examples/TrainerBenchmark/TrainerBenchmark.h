#ifndef _TrainerBenchmark_TrainerBenchmark_h
#define _TrainerBenchmark_TrainerBenchmark_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
#include <Docking/Docking.h>
#include "LoaderMNIST.h"
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS TrainerBenchmarkImg
#define IMAGEFILE <TrainerBenchmark/TrainerBenchmark.iml>
#include <Draw/iml_header.h>

class TrainerBenchmark : public DockWindow {
	
	MetaSession mses;
	String t;
	bool trainer_running, trainer_stopped;
	
	// Set network
	ParentCtrl net_ctrl;
	DocEdit net_edit;
	Button reload_btn;
	
	// Main view
	Splitter split;
	MetaSessionGraph loss_vs_num_graph, testacc_vs_num_graph, trainacc_vs_num_graph;
	
	
public:
	typedef TrainerBenchmark CLASSNAME;
	TrainerBenchmark();
	~TrainerBenchmark();
	
	virtual void DockInit();
	
	void Reload();
	void PostReload() {PostCallback(THISBACK(Reload));}
	void Runner();
	void StartTrainer();
	void ContinueTrainer();
	void StopTrainer() {trainer_running = false; while (!trainer_stopped) Sleep(100);}
	
	SessionData& GetSessionData() {return mses.GetSessionData(0);}
	
};

#endif
